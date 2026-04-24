from __future__ import annotations

from devpi_common.metadata import BasenameMeta
from devpi_common.metadata import is_archive_of_project
from devpi_common.metadata import parse_version
from devpi_common.url import URL
from devpi_server.htmlpage import HTMLPage
from devpi_server.log import threadlog
from devpi_server.normalized import normalize_name
from html.parser import HTMLParser
from typing import TYPE_CHECKING
import json


if TYPE_CHECKING:
    ReleaseLinks = list["Link"]


SIMPLE_API_V1_JSON = "application/vnd.pypi.simple.v1+json"
SIMPLE_API_V1_VERSION = parse_version("1.0")
SIMPLE_API_V2_VERSION = parse_version("2.0")
SIMPLE_API_ACCEPT = (
    f"application/vnd.pypi.simple.v1+html;q=0.2, {SIMPLE_API_V1_JSON}, text/html;q=0.01"
)


class Link(URL):
    def __init__(self, url="", *args, **kwargs):
        self.requires_python = kwargs.pop("requires_python", None)
        self.yanked = kwargs.pop("yanked", None)
        URL.__init__(self, url, *args, **kwargs)


class ProjectHTMLParser(HTMLParser):
    def __init__(self, url):
        HTMLParser.__init__(self)
        self.projects = set()
        self.baseurl = URL(url)
        self.basehost = self.baseurl.hostname
        self.project = None

    def handle_data(self, data):
        if self.project:
            self.project = data.strip()

    def handle_starttag(self, tag, attrs):
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

    def handle_endtag(self, tag):
        if tag == "a" and self.project:
            self.projects.add(self.project)
            self.project = None


class ProjectJSONv1Parser:
    def __init__(self, url):
        self.baseurl = URL(url)

    def feed(self, data):
        meta = data["meta"]
        api_version = parse_version(meta.get("api-version", "1.0"))
        if not (SIMPLE_API_V1_VERSION <= api_version < SIMPLE_API_V2_VERSION):
            raise ValueError(
                f"Wrong API version {api_version!r} in mirror json response."
            )
        self.projects = {x["name"] for x in data["projects"]}


class IndexParser:
    def __init__(self, project):
        self.project = normalize_name(project)
        self.basename2link = {}

    def _mergelink_ifbetter(self, newlink):
        """
        Stores a link given it's better fit than an existing one (if any).

        A link with hash_spec replaces one w/o it, even if the latter got other
        non-empty attributes (like requires_python), unlike the former.
        As soon as the first link with hash_spec is encountered, those that
        appear later are ignored.
        """
        entry = self.basename2link.get(newlink.basename)
        if entry is None or (
            not entry.hash_spec
            and (
                newlink.hash_spec
                or (not entry.requires_python and newlink.requires_python)
            )
        ):
            self.basename2link[newlink.basename] = newlink
            threadlog.debug("indexparser: adding link %s", newlink)
        else:
            threadlog.debug("indexparser: ignoring candidate link %s", newlink)

    @property
    def releaselinks(self) -> ReleaseLinks:
        # the BasenameMeta wrapping essentially does link validation
        return [BasenameMeta(x).obj for x in self.basename2link.values()]

    def parse_index(self, disturl, html):
        p = HTMLPage(html, disturl.url)
        seen = set()
        for link in p.links:
            newurl = Link(
                link.url, requires_python=link.requires_python, yanked=link.yanked
            )
            if not newurl.is_valid_http_url():
                continue
            if is_archive_of_project(newurl, self.project):
                if not newurl.is_valid_http_url():
                    threadlog.warn("unparsable/unsupported url: %r", newurl)
                else:
                    seen.add(newurl.url)
                    self._mergelink_ifbetter(newurl)
                    continue


def parse_index(disturl, html):
    if not isinstance(disturl, URL):
        disturl = URL(disturl)
    project = disturl.basename or disturl.parentbasename
    parser = IndexParser(project)
    parser.parse_index(disturl, html)
    return parser


def parse_index_v1_json(disturl: URL | str, text: str) -> ReleaseLinks:
    if not isinstance(disturl, URL):
        disturl = URL(disturl)
    data = json.loads(text)
    meta = data["meta"]
    api_version = parse_version(meta.get("api-version", "1.0"))
    if not (SIMPLE_API_V1_VERSION <= api_version < SIMPLE_API_V2_VERSION):
        raise ValueError(f"Wrong API version {api_version!r} in mirror json response.")
    result = []
    for item in data["files"]:
        url = disturl.joinpath(item["url"])
        hashes = item["hashes"]
        if "sha256" in hashes:
            url = url.replace(fragment=f"sha256={hashes['sha256']}")
        elif hashes:
            url = url.replace(fragment="=".join(next(iter(hashes.items()))))
        # the BasenameMeta wrapping essentially does link validation
        result.append(
            BasenameMeta(
                Link(
                    url,
                    requires_python=item.get("requires-python"),
                    yanked=item.get("yanked"),
                )
            ).obj
        )
    return result
