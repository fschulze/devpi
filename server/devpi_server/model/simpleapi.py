from __future__ import annotations

from attrs import field
from attrs import frozen
from contextlib import suppress
from devpi_common.metadata import BasenameMeta
from devpi_common.metadata import is_archive_of_project
from devpi_common.metadata import parse_version
from devpi_common.metadata import splitbasename
from devpi_common.url import URL
from devpi_server.filestore import Digests
from devpi_server.filestore import MutableFileEntry
from devpi_server.filestore import split_digest
from devpi_server.htmlpage import HTMLPage
from devpi_server.log import threadlog
from devpi_server.normalized import normalize_name
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.parse import unquote
import json
import re


if TYPE_CHECKING:
    from .links import RequiresPython
    from .links import Yanked
    from .schema import Schema
    from attrs import Attribute
    from devpi_server.keyfs_types import LocatedKey
    from devpi_server.keyfs_types import ULIDKey
    from devpi_server.normalized import NormalizedName
    from devpi_server.readonly import DictViewReadonly
    from typing import Any

    SimpleInfos = list["SimpleInfo"]


SIMPLE_API_V1_JSON = "application/vnd.pypi.simple.v1+json"
SIMPLE_API_V1_0_VERSION = parse_version("1.0")
SIMPLE_API_V1_1_VERSION = parse_version("1.1")
SIMPLE_API_V2_VERSION = parse_version("2.0")
SIMPLE_API_ACCEPT = (
    f"application/vnd.pypi.simple.v1+html;q=0.2, {SIMPLE_API_V1_JSON}, text/html;q=0.01"
)


@frozen(kw_only=True)
class SimpleInfo:
    basename: str
    hashes: Digests = field()
    metadata_hashes: Digests | None = field()
    requires_python: RequiresPython
    url: URL = field()
    yanked: Yanked

    @hashes.validator
    def _check_hashes(self, _attribute: Attribute, value: Digests) -> None:
        assert isinstance(value, Digests)

    @metadata_hashes.validator
    def _check_metadata_hashes(self, _attribute: Attribute, value: Digests) -> None:
        assert value is None or isinstance(value, Digests)

    @url.validator
    def _check_url(self, _attribute: Attribute, value: URL) -> None:
        if not value.is_valid_http_url():
            raise ValueError(f"Not a valid URL: {value}")
        if value.fragment:
            raise ValueError(f"URL contains fragment: {value}")

    @classmethod
    def from_url(cls, url: URL | str) -> SimpleInfo:
        url = URL(url)
        return cls(
            basename=url.basename,
            hashes=Digests.from_spec(url.hash_spec) if url.hash_spec else Digests(),
            metadata_hashes=None,
            requires_python=None,
            url=url.geturl_nofragment(),
            yanked=None,
        )

    def make_key(
        self,
        schema: Schema,
        key_index: LocatedKey[dict, DictViewReadonly] | ULIDKey[dict, DictViewReadonly],
    ) -> LocatedKey[dict, DictViewReadonly]:
        if (hash_value := self.hashes.best_available_value) is not None:
            # we can only create 32K entries per directory
            # so let's take the first 3 bytes which gives
            # us a maximum of 16^3 = 4096 entries in the root dir
            (a, b) = split_digest(hash_value)
            return schema.FILE.locate(
                parent_key=key_index, hashdir_a=a, hashdir_b=b, filename=self.basename
            )
        parts = URL(self.url).torelpath().split("/")
        assert parts
        dirname = "_".join(parts[:-1])
        dirname = re.sub("[^a-zA-Z0-9_.-]", "_", dirname)
        return schema.FILE_NOHASH.locate(
            parent_key=key_index, dirname=unquote(dirname), basename=self.basename
        )

    def make_mutable_entry(
        self, schema: Schema, user: str, index: str, project: str
    ) -> MutableFileEntry:
        key_index = schema.INDEX.locate(user=user, index=index).resolve(fetch=True)
        key = self.make_key(schema, key_index)
        entry = MutableFileEntry(key)
        entry.url = self.url.url
        if self.hashes:
            entry._hashes = self.hashes
        entry.project = project
        version = None
        with suppress(ValueError):
            (_projectname, version, _ext) = splitbasename(self.basename)
        # only store version on entry if we can determine it
        # since version is a meta property of FileEntry, it will return None
        # if not set, if we set it explicitly, it would waste space in the
        # database
        if version is not None:
            entry.version = version
        return entry


class ProjectHTMLParser(HTMLParser):
    def __init__(self, url: URL | str) -> None:
        HTMLParser.__init__(self)
        self.projects: set[str] = set()
        self.baseurl = URL(url)
        self.basehost = self.baseurl.hostname
        self.project: str | None = None

    def handle_data(self, data: str) -> None:
        if self.project:
            self.project = data.strip()

    def handle_starttag(
        self, tag: str, attrs: dict[str, Any] | list[tuple[str, Any]]
    ) -> None:
        if tag == "a":
            self.project = None
            attrs = dict(attrs)
            if "href" not in attrs:
                return
            href = attrs["href"]
            if "://" not in href:
                project = href.rstrip("/").rsplit("/", 1)[-1]
            else:
                newurl = self.baseurl.joinpath(href)
                # remove trailing slashes, so basename works correctly
                newurl = newurl.asfile()
                if not newurl.is_valid_http_url():
                    return
                if not newurl.path.startswith(self.baseurl.path):
                    return
                if self.basehost != newurl.hostname:
                    return
                project = newurl.basename
            self.project = project

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.project:
            self.projects.add(self.project)
            self.project = None


class ProjectJSONv1Parser:
    def __init__(self, url: URL | str) -> None:
        self.baseurl = URL(url)

    def feed(self, data: dict[str, Any]) -> None:
        meta = data["meta"]
        api_version = parse_version(meta.get("api-version", "1.0"))
        if not (SIMPLE_API_V1_0_VERSION <= api_version < SIMPLE_API_V2_VERSION):
            raise ValueError(
                f"Wrong API version {api_version!r} in remote json response."
            )
        self.projects = {x["name"] for x in data["projects"]}


class IndexParser:
    def __init__(self, project: NormalizedName | str) -> None:
        self.project = normalize_name(project)
        self.basename2link: dict[str, SimpleInfo] = {}

    def _mergelink_ifbetter(self, newlink: SimpleInfo) -> None:
        """
        Stores a link given it's better fit than an existing one (if any).

        A link with hashes replaces one w/o it, even if the latter got other
        non-empty attributes (like requires_python), unlike the former.
        As soon as the first link with hashes is encountered, those that
        appear later are ignored.
        """
        entry = self.basename2link.get(newlink.basename)
        if entry is None or (
            not entry.hashes
            and (
                newlink.hashes
                or (not entry.requires_python and newlink.requires_python)
            )
        ):
            self.basename2link[newlink.basename] = newlink
            threadlog.debug("indexparser: adding link %s", newlink)
        else:
            threadlog.debug("indexparser: ignoring candidate link %s", newlink)

    @property
    def releaselinks(self) -> SimpleInfos:
        # the BasenameMeta wrapping essentially does link validation
        return [BasenameMeta(x).obj for x in self.basename2link.values()]

    def parse_index(self, disturl: URL, html: str) -> None:
        p = HTMLPage(html, disturl.url)
        for link in p.links:
            url = URL(link.url)
            if not url.is_valid_http_url():
                continue
            newlink = SimpleInfo(
                basename=url.basename,
                hashes=Digests.from_spec(url.hash_spec) if url.hash_spec else Digests(),
                metadata_hashes=None
                if link.metadata_hash_spec is None
                else Digests.from_metadata_spec(link.metadata_hash_spec),
                requires_python=link.requires_python,
                url=url.geturl_nofragment(),
                yanked=link.yanked,
            )
            if is_archive_of_project(newlink.basename, self.project):
                self._mergelink_ifbetter(newlink)
                continue


def parse_index(disturl: URL | str, html: str) -> IndexParser:
    if not isinstance(disturl, URL):
        disturl = URL(disturl)
    project = disturl.basename or disturl.parentbasename
    parser = IndexParser(project)
    parser.parse_index(disturl, html)
    return parser


def parse_index_v1_json(disturl: URL | str, text: str) -> SimpleInfos:
    if not isinstance(disturl, URL):
        disturl = URL(disturl)
    data = json.loads(text)
    meta = data["meta"]
    api_version = parse_version(meta.get("api-version", "1.0"))
    if not (SIMPLE_API_V1_0_VERSION <= api_version < SIMPLE_API_V2_VERSION):
        raise ValueError(f"Wrong API version {api_version!r} in remote json response.")
    result = []
    for item in data["files"]:
        metadata_hashes = item.get("core-metadata")
        url = disturl.joinpath(item["url"])
        # the BasenameMeta wrapping essentially does link validation
        result.append(
            BasenameMeta(
                SimpleInfo(
                    basename=url.basename,
                    hashes=Digests(item["hashes"]),
                    metadata_hashes=(
                        None
                        if metadata_hashes is None or metadata_hashes is False
                        else Digests({} if metadata_hashes is True else metadata_hashes)
                    ),
                    requires_python=item.get("requires-python"),
                    url=url.geturl_nofragment(),
                    yanked=item.get("yanked"),
                )
            ).obj
        )
    return result
