"""

Implementation of the database layer for PyPI Package serving and
toxresult storage.

"""
from __future__ import annotations

from .config import hookimpl
from .exceptions import lazy_format_exception
from .filestore import BadGateway
from .filestore import RunningHashes
from .filestore import key_from_link
from .htmlpage import HTMLPage
from .log import threadlog
from .markers import absent
from .markers import deleted
from .markers import unknown
from .model import BaseStage
from .model import BaseStageCustomizer
from .model import Rel
from .model import ensure_boolean
from .model import join_links_data
from .normalized import NormalizedName
from .normalized import normalize_name
from .readonly import ensure_deeply_readonly
from .readonly import get_mutable_deepcopy
from contextlib import ExitStack
from devpi_common.metadata import BasenameMeta
from devpi_common.metadata import is_archive_of_project
from devpi_common.metadata import parse_version
from devpi_common.metadata import splitbasename
from devpi_common.types import cached_property
from devpi_common.url import URL
from functools import partial
from html.parser import HTMLParser
from pyramid.authentication import b64encode
from typing import TYPE_CHECKING
import asyncio
import json
import re
import threading
import time
import weakref


if TYPE_CHECKING:
    from .markers import Unknown
    from typing import Any

SIMPLE_API_V1_JSON = "application/vnd.pypi.simple.v1+json"
SIMPLE_API_V1_VERSION = parse_version("1.0")
SIMPLE_API_V2_VERSION = parse_version("2.0")
SIMPLE_API_ACCEPT = ", ".join((
    "application/vnd.pypi.simple.v1+html;q=0.2",
    SIMPLE_API_V1_JSON,
    "text/html;q=0.01"))


def _headers_from_response(r):
    content_type = r.headers.get("content-type")
    if not content_type:
        content_type = "application/octet-stream"
    if isinstance(content_type, tuple):
        content_type = content_type[0]
    headers = {
        "X-Accel-Buffering": "no",  # disable buffering in nginx
        "content-type": content_type,
    }
    if "last-modified" in r.headers:
        headers["last-modified"] = r.headers["last-modified"]
    if "content-length" in r.headers:
        headers["content-length"] = str(r.headers["content-length"])
    return headers


class FileStreamer:
    def __init__(self, f, entry, response):
        self.hash_type = entry.best_available_hash_type
        self.hash_types = entry.default_hash_types
        self._hashes = entry.hashes
        self.relpath = entry.relpath
        self.response = response
        self.error = None
        self.f = f

    def __iter__(self):
        filesize = 0
        running_hashes = RunningHashes(self.hash_type, *self.hash_types)
        running_hashes.start()
        content_size = self.response.headers.get("content-length")

        yield _headers_from_response(self.response)

        data_iter = self.response.iter_raw(10240)
        while 1:
            data = next(data_iter, None)
            if data is None:
                break
            filesize += len(data)
            for rh in running_hashes._running_hashes:
                rh.update(data)
            self.f.write(data)
            yield data

        self.hashes = running_hashes.digests

        if content_size and int(content_size) != filesize:
            raise ValueError(
                "%s: got %s bytes of %r from remote, expected %s"
                % (self.relpath, filesize, self.response.url, content_size)
            )
        if self._hashes:
            err = self.hashes.exception_for(self._hashes, self.relpath)
            if err is not None:
                raise err


def iter_cache_remote_file(stage, entry, url):
    # we get and cache the file and some http headers from remote
    xom = stage.xom
    url = URL(url)

    with ExitStack() as cstack:
        r = stage.http.stream(cstack, "GET", url, allow_redirects=True)
        if r.status_code != 200:
            r.close()
            msg = f"error {r.status_code} getting {url}"
            threadlog.error(msg)
            raise BadGateway(msg, code=r.status_code, url=url)
        f = cstack.enter_context(entry.file_new_open())
        file_streamer = FileStreamer(f, entry, r)
        threadlog.info("reading remote: %r, target %s", URL(r.url), entry.relpath)

        try:
            yield from file_streamer
        except Exception as err:
            threadlog.error(str(err))
            raise

        if not entry.has_existing_metadata():
            with xom.keyfs.write_transaction(allow_restart=True):
                if entry.readonly:
                    entry = xom.filestore.get_file_entry_from_key(entry.key)
                entry.file_set_content(
                    f,
                    last_modified=r.headers.get("last-modified", None),
                    hashes=file_streamer.hashes,
                )
                digest_key = entry.get_digest_key()
                with digest_key.update() as digest_paths:
                    digest_paths.add(entry.relpath)
                if entry.project:
                    stage = xom.model.getstage(entry.user, entry.index)
                    # for mirror indexes this makes sure the project is in the database
                    # as soon as a file was fetched
                    stage.add_project_name(entry.project)
                # on Windows we need to close the file
                # before the transaction closes
                f.close()
        else:
            # the file was downloaded before but locally removed, so put
            # it back in place without creating a new serial
            with xom.keyfs.filestore_transaction():
                entry.file_set_content_no_meta(f, hashes=file_streamer.hashes)
                threadlog.debug(
                    "put missing file back into place: %s", entry.file_path_info
                )
                # on Windows we need to close the file
                # before the transaction closes
                f.close()


def iter_remote_file_replica(stage, entry, url):
    xom = stage.xom
    replication_errors = xom.replica_thread.shared_data.errors
    # construct primary URL with param
    primary_url = xom.config.primary_url.joinpath(entry.relpath).url
    if not url:
        threadlog.warn("missing private file: %s" % entry.relpath)
    else:
        threadlog.info("replica doesn't have file: %s", entry.relpath)

    with ExitStack() as cstack:
        r = stage.http.stream(
            cstack,
            "GET",
            primary_url,
            allow_redirects=True,
            extra_headers={xom.replica_thread.H_REPLICA_FILEREPL: "YES"},
        )
        if r.status_code != 200:
            r.close()
            msg = f"{primary_url}: received {r.status_code} from primary"
            if not url:
                threadlog.error(msg)
                raise BadGateway(msg)
            # try to get from original location
            headers = {}
            url = URL(url)
            username = url.username or ""
            password = url.password or ""
            if username or password:
                url = url.replace(username=None, password=None)
                auth = f"{username}:{password}".encode()
                headers["Authorization"] = f"Basic {b64encode(auth).decode()}"
            r = xom.http.stream(
                cstack, "GET", url, allow_redirects=True, extra_headers=headers
            )
            if r.status_code != 200:
                r.close()
                msg = f"{msg}\n{url}: received {r.status_code}"
                threadlog.error(msg)
                raise BadGateway(msg)
        cstack.callback(r.close)
        f = cstack.enter_context(entry.file_new_open())
        file_streamer = FileStreamer(f, entry, r)

        try:
            yield from file_streamer
        except Exception as err:  # noqa: BLE001 - we have to convert all exceptions
            # the file we got is different, so we fail
            raise BadGateway(str(err)) from err

        with xom.keyfs.filestore_transaction():
            entry.file_set_content_no_meta(f, hashes=file_streamer.hashes)
            # on Windows we need to close the file
            # before the transaction closes
            f.close()
            threadlog.debug(
                "put missing file back into place: %s", entry.file_path_info
            )
        # in case there were errors before, we can now remove them
        replication_errors.remove(entry)


def iter_fetch_remote_file(stage, entry, url):
    if not stage.xom.is_replica():
        yield from iter_cache_remote_file(stage, entry, url)
    else:
        yield from iter_remote_file_replica(stage, entry, url)


class Link(URL):
    def __init__(self, url="", *args, **kwargs):
        self.requires_python = kwargs.pop('requires_python', None)
        self.yanked = kwargs.pop('yanked', None)
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
        if tag == 'a':
            self.project = None
            attrs = dict(attrs)
            if 'href' not in attrs:
                return
            href = attrs['href']
            if '://' not in href:
                project = href.rstrip('/').rsplit('/', 1)[-1]
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
        if tag == 'a' and self.project:
            self.projects.add(self.project)
            self.project = None


class ProjectJSONv1Parser:
    def __init__(self, url):
        self.baseurl = URL(url)

    def feed(self, data):
        meta = data['meta']
        api_version = parse_version(meta.get('api-version', '1.0'))
        if not (SIMPLE_API_V1_VERSION <= api_version < SIMPLE_API_V2_VERSION):
            raise ValueError(
                "Wrong API version %r in mirror json response."
                % api_version)
        self.projects = set(x['name'] for x in data['projects'])


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
        if entry is None or (not entry.hash_spec and (newlink.hash_spec or (
            not entry.requires_python and newlink.requires_python
        ))):
            self.basename2link[newlink.basename] = newlink
            threadlog.debug("indexparser: adding link %s", newlink)
        else:
            threadlog.debug("indexparser: ignoring candidate link %s", newlink)

    @property
    def releaselinks(self):
        # the BasenameMeta wrapping essentially does link validation
        return [BasenameMeta(x).obj for x in self.basename2link.values()]

    def parse_index(self, disturl, html):
        p = HTMLPage(html, disturl.url)
        seen = set()
        for link in p.links:
            newurl = Link(link.url, requires_python=link.requires_python, yanked=link.yanked)
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


def parse_index_v1_json(disturl, text):
    if not isinstance(disturl, URL):
        disturl = URL(disturl)
    data = json.loads(text)
    meta = data['meta']
    api_version = parse_version(meta.get('api-version', '1.0'))
    if not (SIMPLE_API_V1_VERSION <= api_version < SIMPLE_API_V2_VERSION):
        raise ValueError(
            "Wrong API version %r in mirror json response."
            % api_version)
    result = []
    for item in data['files']:
        url = disturl.joinpath(item['url'])
        hashes = item['hashes']
        if 'sha256' in hashes:
            url = url.replace(fragment=f"sha256={hashes['sha256']}")
        elif hashes:
            url = url.replace(fragment="=".join(next(iter(hashes.items()))))
        # the BasenameMeta wrapping essentially does link validation
        result.append(BasenameMeta(Link(
            url,
            requires_python=item.get('requires-python'),
            yanked=item.get('yanked'))).obj)
    return result


class HTTPClient:
    def __init__(self, http, get_extra_headers, update_auth_candidates):
        self.http = http
        self.get_extra_headers = get_extra_headers
        self.update_auth_candidates = update_auth_candidates

    async def async_get(
        self, url, *, allow_redirects, timeout=None, extra_headers=None
    ):
        extra_headers = self.get_extra_headers(extra_headers)
        response, text = await self.http.async_get(
            url=URL(url).url,
            allow_redirects=allow_redirects,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        # if we get an auth problem, see if we can try an alternative credential
        # to access the resource
        if response.status_code in (401, 403) and self.update_auth_candidates(
            response.headers.get("WWW-Authenticate", ""),
        ):
            return await self.async_get(
                url,
                allow_redirects=allow_redirects,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        return response, text

    def get(self, url, *, allow_redirects, timeout=None, extra_headers=None):
        extra_headers = self.get_extra_headers(extra_headers)
        response = self.http.get(
            url=URL(url).url,
            allow_redirects=allow_redirects,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        # if we get an auth problem, see if we can try an alternative credential
        # to access the resource
        if response.status_code in (401, 403) and self.update_auth_candidates(
            response.headers.get("WWW-Authenticate", ""),
        ):
            return self.get(
                url,
                allow_redirects=allow_redirects,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        return response

    def stream(
        self, cstack, method, url, *, allow_redirects, timeout=None, extra_headers=None
    ):
        extra_headers = self.get_extra_headers(extra_headers)
        response = self.http.stream(
            cstack,
            method,
            URL(url).url,
            allow_redirects=allow_redirects,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        # if we get an auth problem, see if we can try an alternative credential
        # to access the resource
        if response.status_code in (401, 403) and self.update_auth_candidates(
            response.headers.get("WWW-Authenticate", ""),
        ):
            return self.stream(
                cstack,
                method,
                url,
                allow_redirects=allow_redirects,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        return response


class MirrorLinks:
    def __init__(self, stage, project):
        self.stage = stage
        self.project = normalize_name(project)
        self.key_projectname = stage.key_projectname(self.project)

    def are_expired(self):
        if self.key_projectname.exists():
            return self.stage.cache_retrieve_times.is_expired(
                self.project, self.stage.cache_expiry
            )
        return True

    def delete(self):
        if (links := self.get_links()) is not None:
            _entry_from_href = self.stage._entry_from_href
            entries = (_entry_from_href(x[1]) for x in links)
            entries = (x for x in entries if x.file_exists())
            for entry in entries:
                entry.delete()
        key_mirrorfile = self.stage.key_mirrorfile(self.project)
        for key in list(key_mirrorfile.iter_ulidkeys(fill_cache=False)):
            key.delete()
        self.stage.cache_retrieve_times.expire(self.project)
        self.stage.key_projectcacheinfo(self.project).delete()

    def delete_version(self, version):
        # since this is a mirror, we only have the simple links and no
        # metadata, so only delete the files and keep the simple links
        # for the possibility to re-download a release
        if (links := self.get_links()) is not None:
            _entry_from_href = self.stage._entry_from_href
            entries = [_entry_from_href(x[1]) for x in links]
            entries = [x for x in entries if x.version == version and x.file_exists()]
            for entry in entries:
                entry.delete_file_only()

    def get_cache_info(self):
        return self.stage.key_projectcacheinfo(self.project).get()

    def get_fresh_links(self):
        if not self.key_projectname.exists() or not self.get_cache_info():
            return None
        key_mirrorfile = self.stage.key_mirrorfile(project=self.project)
        links_with_data = [
            (k.name, v["entrypath"], v.get("requires_python"), v.get("yanked"))
            for k, v in key_mirrorfile.iter_ulidkey_values()
        ]
        if self.stage.offline:
            entries = self.stage.get_entries_for_entrypaths(
                x[1] for x in links_with_data
            )
            links_with_data = ensure_deeply_readonly(
                [
                    link
                    for (link, entry) in zip(links_with_data, entries)
                    if entry is not None and entry.file_exists()
                ]
            )
        return links_with_data

    def get_links(self):
        # we might implement caching in get_links
        return self.get_fresh_links()

    def has_links(self):
        _is_file_cached = self.stage._is_file_cached
        if (links := self.get_links()) is not None:
            return any(_is_file_cached(x) for x in links)
        return False

    def _save_cache_links(self, links, requires_python, yanked, cache_info):
        if cache_info is None:
            cache_info = self.get_cache_info()
        cache_info = get_mutable_deepcopy(cache_info)
        cache_info.setdefault("serial", -1)
        assert links != ()  # we don't store the old "Not Found" marker anymore
        assert isinstance(cache_info["serial"], int), cache_info["serial"]
        tx = self.stage.keyfs.tx
        data = {}
        for fn, ep, rp, y in join_links_data(links, requires_python, yanked):
            link_data = data[fn] = dict(entrypath=ep)
            if rp is not None:
                link_data["requires_python"] = rp
            if y is not None:
                link_data["yanked"] = y
        self.key_projectname.set(self.project)
        if "etag" in cache_info and cache_info["etag"] is None:
            del cache_info["etag"]
        self.stage.key_projectcacheinfo(self.project).set(cache_info)
        key_mirrorfile = self.stage.key_mirrorfile(self.project)
        key_version = self.stage.key_version(self.project)
        name_key_map = {}
        old = {}
        for k, v in key_mirrorfile.iter_ulidkey_values():
            name_key_map[k.name] = k
            old[k.name] = v
            assert v["entrypath"].rsplit("#", 1)[0].endswith(f"/{k.name}"), (k, v)
        if old != data:
            threadlog.debug(
                "processing changed simplelinks for %s in %s",
                self.project,
                self.stage.index,
            )
            num_unchanged = 0
            num_changed = 0
            num_deleted = 0
            num_new = 0
            for name, v in old.items():
                new_value = data.pop(name, absent)
                if new_value == v:
                    num_unchanged += 1
                    continue
                key = name_key_map[name]
                if new_value is absent:
                    num_deleted += 1
                    key.delete()
                else:
                    num_changed += 1
                    key.set(new_value)
            new_keys = set()
            new_versions = {}
            old_versions = dict(key_version.iter_ulidkey_values())
            if old_versions:
                import pdb

                pdb.set_trace()
            for fn, value in data.items():
                new_keys.add(key_mirrorfile(fn))
                (project, version, _ext) = splitbasename(fn)
                assert project == self.project
                if version not in old_versions:
                    version_key = key_version(version)
                    new_version = new_versions[version_key] = {}
                    if "requires_python" in value:
                        new_version = RuntimeError
            for _k, ulid_key in tx.resolve_keys(
                new_keys, fetch=True, fill_cache=True, new_for_missing=True
            ):
                num_new += 1
                ulid_key.set(data[ulid_key.name])
            threadlog.debug(
                "%s new, %s deleted, %s changed, and %s unchanged simplelinks for %s in %s",
                num_new,
                num_deleted,
                num_changed,
                num_unchanged,
                self.project,
                self.stage.index,
            )
            # maintain list of currently cached project names to enable
            # deletion and offline mode
            self.stage.add_project_name(self.project)

        def on_commit():
            threadlog.debug("setting projects cache for %r", self.project)
            self.stage.cache_retrieve_times.refresh(
                self.project, cache_info.get("etag")
            )
            # make project appear in projects list even
            # before we next check up the full list with remote
            self.stage.cache_projectnames.add(self.project)

        tx.on_commit_success(on_commit)


class MirrorStage(BaseStage):
    _mirrorlinks: dict[NormalizedName, MirrorLinks]

    def __init__(self, xom, username, index, ixconfig, customizer_cls):
        super().__init__(
            xom, username, index, ixconfig, customizer_cls)
        self.xom = xom
        self.offline = self.xom.config.offline_mode
        self.timeout = xom.config.request_timeout
        # use a minimum of 30 seconds as timeout for remote server and
        # 60 seconds when running as replica, because the list can be
        # quite large and the primary might take a while to process it
        self.projects_timeout = max(self.timeout, 60 if self.xom.is_replica() else 30)
        # list of locally mirrored projects
        self.key_projectname = self.keyfs.PROJECTNAME(user=username, index=index)
        # used to log about stale projects only once
        self._offline_logging = set()
        self._mirrorlinks = {}

    def key_mirrorfile(self, project, filename=None):
        kw = {} if filename is None else dict(filename=filename)
        return self.keyfs.MIRRORFILE(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    def key_projectcacheinfo(self, project=None):
        kw = {} if project is None else dict(project=normalize_name(project))
        return self.keyfs.PROJECTCACHEINFO(user=self.username, index=self.index, **kw)

    @cached_property
    def http(self):
        if self.xom.is_replica():
            (uuid, primary_uuid) = self.xom.config.nodeinfo.make_uuid_headers()
            get_extra_headers = (
                self.xom.replica_thread.connection.http.get_extra_headers
            )
        else:

            def get_extra_headers(extra_headers):
                # make a copy of extra_headers
                extra_headers = {} if extra_headers is None else dict(extra_headers)
                auth = self.mirror_url_authorization_header
                if auth:
                    extra_headers["Authorization"] = auth
                return extra_headers

        return HTTPClient(
            self.xom.http, get_extra_headers, self._update_auth_candidates
        )

    def _update_auth_candidates(self, auth_header):
        # if we have any auth candidates, the first one has just failed, so
        # discard it, allowing any others in the list to be tried
        # if we didn't have any auth candidates, try and obtain some
        auth_candidates = self.xom.setdefault_singleton(
            self.name, "auth_candidates", factory=list)
        if auth_candidates:
            auth_candidates.pop(0)
        else:
            hook = self.xom.config.hook
            auth_candidates.extend(hook.devpiserver_get_mirror_auth(
                mirror_url=self.mirror_url,
                www_authenticate_header=auth_header))
        # return True if we have any new credentials to try
        return len(auth_candidates) > 0

    @property
    def cache_expiry(self):
        return self.ixconfig.get(
            'mirror_cache_expiry', self.xom.config.mirror_cache_expiry)

    @property
    def ignore_serial_header(self):
        return self.ixconfig.get("mirror_ignore_serial_header", False)

    @property
    def mirror_url(self):
        if self.xom.is_replica():
            url = self.xom.config.primary_url.joinpath(self.name, "+simple")
        else:
            url = URL(self.ixconfig['mirror_url'])
        return url.asdir()

    @property
    def mirror_url_without_auth(self):
        return self.mirror_url.replace(username=None, password=None)

    @property
    def mirror_url_auth(self):
        url = self.mirror_url
        return dict(username=url.username, password=url.password)

    @property
    def mirror_url_authorization_header(self):
        # prefer plugin generated credentials as they will only get generated
        # if the url embedded auth has previously failed
        auth_candidates = self.xom.setdefault_singleton(
            self.name, "auth_candidates", factory=list)
        if auth_candidates:
            return auth_candidates[0]
        url = self.mirror_url
        if url.username or url.password:
            auth = f"{url.username or ''}:{url.password or ''}".encode()
            return f"Basic {b64encode(auth).decode()}"
        return None

    @property
    def no_project_list(self):
        return self.ixconfig.get('mirror_no_project_list', False)

    @property
    def use_external_url(self):
        return self.ixconfig.get('mirror_use_external_urls', False)

    def get_possible_indexconfig_keys(self):
        return (
            *(k for k, v in self.get_default_config_items()),
            "custom_data",
            "description",
            "mirror_cache_expiry",
            "mirror_ignore_serial_header",
            "mirror_no_project_list",
            "mirror_url",
            "mirror_use_external_urls",
            "mirror_web_url_fmt",
            "title",
        )

    def get_default_config_items(self):
        return [("volatile", True)]

    def normalize_indexconfig_value(self, key, value):
        if key == "volatile":
            return ensure_boolean(value)
        if key == "mirror_url":
            if not value.startswith(("http://", "https://")):
                raise self.InvalidIndexconfig([
                    "'mirror_url' option must be a URL."])
            return value
        if key == "mirror_cache_expiry":
            try:
                value = int(value)
            except (TypeError, ValueError) as e:
                raise self.InvalidIndexconfig([
                    "'mirror_cache_expiry' option must be an integer"]) from e
            return value
        if key == "mirror_ignore_serial_header":
            return ensure_boolean(value)
        if key == "mirror_no_project_list":
            return ensure_boolean(value)
        if key == "mirror_use_external_urls":
            return ensure_boolean(value)
        if key in ("custom_data", "description", "mirror_web_url_fmt", "title"):
            return value
        return None

    def delete(self):
        # delete all projects on this index
        for key in list(self.key_projectname.iter_ulidkeys()):
            self.del_project(key.name)
            key.delete()
        BaseStage.delete(self)

    def add_project_name(self, project):
        project = normalize_name(project)
        self.key_projectname(project).set(project)

    def del_project(self, project):
        if not self.is_project_cached(project):
            raise KeyError("project not found")
        mirrorlinks = self._get_mirrorlinks(project)
        mirrorlinks.delete()
        self.key_projectname(project).delete()

    def del_versiondata(self, project, version, cleanup=True):
        project = normalize_name(project)
        if not self.has_project_perstage(project):
            raise self.NotFound("project %r not found on stage %r" %
                                (project, self.name))
        mirrorlinks = self._get_mirrorlinks(project)
        mirrorlinks.delete_version(version)

    def del_entry(self, entry, cleanup=True):
        project = entry.project
        if project is None:
            raise self.NotFound("no project set on entry %r" % entry)
        if not entry.file_exists():
            raise self.NotFound("entry has no file data %r" % entry)
        entry.delete()
        mirrorlinks = self._get_mirrorlinks(project)
        if not mirrorlinks.has_links() and cleanup:
            mirrorlinks.delete()
            self.key_projectname(project).delete()

    @property
    def _list_projects_perstage_lock(self):
        """ get server wide lock for this index """
        return self.xom.setdefault_singleton(
            self.name, "projects_update_lock", factory=threading.Lock)

    @property
    def cache_projectnames(self):
        """ cache for full list of projectnames. """
        # we could keep this info inside keyfs but pypi.org
        # produces a several MB list of names and it changes often which
        # would spam the database.
        return self.xom.setdefault_singleton(
            self.name, "projectnames", factory=ProjectNamesCache)

    @property
    def cache_retrieve_times(self):
        """ per-xom RAM cache for keeping track when we last updated simplelinks. """
        # we could keep this info in keyfs but it would lead to a write
        # for each remote check.
        return self.xom.setdefault_singleton(
            self.name, "project_retrieve_times", factory=ProjectUpdateCache)

    async def _get_remote_projects(self, projects_future: asyncio.Future) -> None:
        headers = {"Accept": SIMPLE_API_ACCEPT}
        etag = self.cache_projectnames.get_etag()
        if etag is not None:
            headers["If-None-Match"] = etag
        threadlog.debug(
            "fetching remote projects from %r with etag %r", self.mirror_url, etag
        )
        (response, text) = await self.http.async_get(
            self.mirror_url_without_auth,
            allow_redirects=True,
            extra_headers=headers,
        )
        if response.status_code == 304:
            raise self.UpstreamNotModified(
                f"{response.status_code} status on GET {self.mirror_url!r}", etag=etag
            )
        if response.status_code != 200:
            raise self.UpstreamError(
                "URL %r returned %s %s",
                self.mirror_url, response.status_code, response.reason)
        parser: ProjectHTMLParser | ProjectJSONv1Parser
        if response.headers.get('content-type') == SIMPLE_API_V1_JSON:
            parser = ProjectJSONv1Parser(response.url)
            parser.feed(json.loads(text))
        else:
            parser = ProjectHTMLParser(response.url)
            parser.feed(text)
        projects_future.set_result(
            ({normalize_name(x) for x in parser.projects}, response.headers.get("ETag"))
        )

    def _stale_list_projects_perstage(self):
        return {
            v for k, v in self.key_projectname.iter_ulidkey_values(fill_cache=False)
        }

    def _update_projects(self):
        projects_future = self.xom.create_future()
        try:
            self.xom.run_coroutine_threadsafe(
                self._get_remote_projects(projects_future),
                timeout=self.projects_timeout,
            )
        except asyncio.TimeoutError:
            threadlog.warn(
                "serving stale projects for %r, getting data timed out after %s seconds",
                self.index,
                self.projects_timeout,
            )
            return self._stale_list_projects_perstage()
        except self.UpstreamNotModified as e:
            # the etag might have changed
            self.cache_projectnames.mark_current(e.etag)
            return self._stale_list_projects_perstage()
        except self.UpstreamError as e:
            threadlog.warn("upstream error (%s): using stale projects list", e)
            return self._stale_list_projects_perstage()
        (projects, etag) = projects_future.result()
        old = self.cache_projectnames.get()
        if self.cache_projectnames.exists() and old == projects:
            # mark current without updating contents
            self.cache_projectnames.mark_current(etag)
        else:
            self.cache_projectnames.set(projects, etag)

            # trigger an initial-load event on primary
            if not self.xom.is_replica():
                # make sure we are at the current serial
                # this avoids setting the value again when
                # called from the notification thread
                if not self.keyfs.tx.write:
                    self.keyfs.restart_read_transaction()
                k = self.keyfs.MIRRORNAMESINIT(user=self.username, index=self.index)
                # when 0 it is new, when 1 it is pre 6.6.0 with
                # only normalized names
                if k.get() in (0, 1):
                    with self.keyfs.write_transaction(allow_restart=True):
                        k.set(2)
        return projects

    def _list_projects_perstage(self):
        """ Return the cached project names.

            Only for internal use which makes sure the data isn't modified.
        """
        if self.offline:
            threadlog.warn("offline mode: using stale projects list")
            return self._stale_list_projects_perstage()
        if self.no_project_list:
            # upstream of mirror configured as not having a project list
            # return only locally known projects
            return self._stale_list_projects_perstage()
        # try without lock first
        if not self.cache_projectnames.is_expired(self.cache_expiry):
            projects = self.cache_projectnames.get()
        else:
            with self._list_projects_perstage_lock:
                # retry in case it was updated in another thread
                if not self.cache_projectnames.is_expired(self.cache_expiry):
                    projects = self.cache_projectnames.get()
                else:
                    # no fresh projects or None at all, let's go remote
                    projects = self._update_projects()
        return projects

    def list_projects_perstage(self):
        """ Return the project names. """
        # return a read-only version of the cached data,
        # so it can't be modified accidentally and we avoid a copy
        return ensure_deeply_readonly(
            {v: v.original for v in self._list_projects_perstage()}
        )

    def is_project_cached(self, project):
        """ return True if we have some cached simpelinks information. """
        return self.key_projectcacheinfo(project).exists()

    def _entry_from_href(self, href):
        # extract relpath from href by cutting of the hash
        relpath = re.sub(r"#.*$", "", href)
        return self.filestore.get_file_entry(relpath)

    def _is_file_cached(self, link):
        entry = self._entry_from_href(link[1])
        return entry is not None and entry.file_exists()

    def clear_simplelinks_cache(self, project):
        # we have to set to an empty dict instead of removing the key, so
        # replicas behave correctly
        self.cache_retrieve_times.expire(project)
        self.key_projectcacheinfo(project).set({})
        threadlog.debug("cleared cache for %s", project)

    async def _async_fetch_releaselinks(
        self, newlinks_future, project, cache_info, _key_from_link
    ):
        # get the simple page for the project
        url = self.mirror_url.joinpath(project).asdir()
        get_url = self.mirror_url_without_auth.joinpath(project).asdir()
        threadlog.debug("reading index %r", url)
        headers = {"Accept": SIMPLE_API_ACCEPT}
        etag = self.cache_retrieve_times.get_etag(project) or cache_info.get("etag")
        if etag is not None:
            headers["If-None-Match"] = etag
        (response, text) = await self.http.async_get(
            get_url, allow_redirects=True, extra_headers=headers
        )
        if response.status_code == 304:
            raise self.UpstreamNotModified(
                "%s status on GET %r" % (response.status_code, url),
                etag=etag)
        elif response.status_code != 200:
            if response.status_code == 404:
                # immediately cache the not found with no ETag
                self.cache_retrieve_times.refresh(project, None)
                raise self.UpstreamNotFoundError(
                    "not found on GET %r" % url)

            # we don't have an old result and got a non-404 code.
            raise self.UpstreamError("%s status on GET %r" % (
                response.status_code, url))

        # pypi.org provides X-PYPI-LAST-SERIAL header in case of 200 returns.
        # devpi-primary may provide a 200 but not supply the header
        # (it's really a 404 in disguise and we should change
        # devpi-server behaviour since pypi.org serves 404
        # on non-existing projects for a longer time now).
        # Returning a 200 with "no such project" was originally meant to
        # provide earlier versions of easy_install/pip to request the full
        # simple page.
        try:
            serial = int(response.headers.get("X-PYPI-LAST-SERIAL"))
        except (TypeError, ValueError):
            # handle missing or invalid X-PYPI-LAST-SERIAL header
            serial = -1

        if serial < (cache_serial := cache_info.get("serial", -1)):
            if not self.ignore_serial_header:
                msg = (
                    f"serial mismatch on GET {url!r}, "
                    f"cache_serial {cache_serial} is newer than returned serial {serial}"
                )
                raise self.UpstreamError(msg)
            # reset serial, so when switching back we get correct data
            serial = -1

        threadlog.debug("%s: got response with serial %s", project, serial)

        # check returned url has the same normalized name
        assert project == normalize_name(url.asfile().basename)

        # make sure we don't store credential in the database
        response_url = URL(response.url).replace(username=None, password=None)
        # parse simple index's link
        if response.headers.get('content-type') == SIMPLE_API_V1_JSON:
            releaselinks = parse_index_v1_json(response_url, text)
        else:
            releaselinks = parse_index(response_url, text).releaselinks
        num_releaselinks = len(releaselinks)
        key_hrefs: list = [None] * num_releaselinks
        requires_python: list = [None] * num_releaselinks
        yanked: list = [None] * num_releaselinks
        for index, releaselink in enumerate(releaselinks):
            key = _key_from_link(releaselink)
            href = key.relpath
            if releaselink.hash_spec:
                href = f"{href}#{releaselink.hash_spec}"
            key_hrefs[index] = (releaselink.basename, href)
            requires_python[index] = releaselink.requires_python
            yanked[index] = None if releaselink.yanked is False else releaselink.yanked
        newlinks_future.set_result(
            dict(
                cache_info=dict(serial=serial, etag=response.headers.get("ETag")),
                releaselinks=releaselinks,
                key_hrefs=key_hrefs,
                requires_python=requires_python,
                yanked=yanked,
                devpi_serial=response.headers.get("X-DEVPI-SERIAL"),
            )
        )

    def _get_mirrorlinks(self, project: NormalizedName) -> MirrorLinks:
        if project not in self._mirrorlinks:
            self._mirrorlinks[project] = MirrorLinks(self, project)
        return self._mirrorlinks[project]

    def _update_simplelinks(self, project, info, links, newlinks):
        if self.xom.is_replica():
            # on the replica we wait for the changes to arrive (changes were
            # triggered by our http request above) because we have no direct
            # writeaccess to the db other than through the replication thread
            # and we need the current data of the new entries
            devpi_serial = int(info["devpi_serial"])
            threadlog.debug("get_simplelinks pypi: waiting for devpi_serial %r",
                            devpi_serial)
            links = None
            if self.keyfs.wait_tx_serial(devpi_serial, timeout=self.timeout):
                threadlog.debug("get_simplelinks pypi: finished waiting for devpi_serial %r",
                                devpi_serial)
                self.keyfs.restart_read_transaction()
                mirrorlinks = self._get_mirrorlinks(project)
                links = mirrorlinks.get_fresh_links()
            if links is not None:
                self.keyfs.tx.on_commit_success(
                    partial(
                        self.cache_retrieve_times.refresh,
                        project,
                        info["cache_info"]["etag"],
                    )
                )
                return self.SimpleLinks(links)
            raise self.UpstreamError("no cache links from primary for %s" %
                                     project)

        with self.keyfs.write_transaction(allow_restart=True):
            # on the master we need to write the updated links.
            existing_info = set(links or ())
            assert len(newlinks) == len(info["releaselinks"])
            linkstomap = [
                newlink
                for newinfo, link in zip(newlinks, info["releaselinks"])
                if (newlink := link) not in existing_info
            ]
            # looping over iter_maplinks creates the entries in the database
            for _entry in self.filestore.iter_maplinks(
                linkstomap, self.user.name, self.index, project
            ):
                pass
            # this stores the simple links info
            self._get_mirrorlinks(project)._save_cache_links(
                info["key_hrefs"],
                info["requires_python"],
                info["yanked"],
                info["cache_info"],
            )
            return self.SimpleLinks(newlinks)

    async def _update_simplelinks_in_future(self, newlinks_future, project, lock):
        threadlog.debug("Awaiting simple links for %r", project)
        info = await newlinks_future
        threadlog.debug("Got simple links for %r", project)

        newlinks = join_links_data(
            info["key_hrefs"], info["requires_python"], info["yanked"])
        with self.keyfs.write_transaction():
            self.keyfs.tx.on_finished(lock.release)
            # fetch current links
            mirrorlinks = self._get_mirrorlinks(project)
            links = mirrorlinks.get_fresh_links()
            if links is None or set(links) != set(newlinks):
                # we got changes, so store them
                self._update_simplelinks(project, info, links, newlinks)
                threadlog.debug(
                    "Updated simplelinks for %r in background", project)
            else:
                threadlog.debug("Unchanged simplelinks for %r", project)

    def get_simplelinks_perstage(self, project):
        """ return all releaselinks from the index, returning cached entries
        if we have a recent enough request stored locally.

        Raise UpstreamError if the pypi server cannot be reached or
        does not return a fresh enough page although we know it must
        exist.
        """
        project = normalize_name(project)
        lock = self.cache_retrieve_times.acquire(project, self.timeout)
        if lock is not None:
            self.keyfs.tx.on_finished(lock.release)
        mirrorlinks = self._get_mirrorlinks(project)
        is_expired = mirrorlinks.are_expired()
        if not is_expired and lock is not None:
            lock.release()

        links = mirrorlinks.get_links()
        if lock is None:
            if links is not None:
                threadlog.warn(
                    "serving stale links for %r, waiting for existing request timed out after %s seconds",
                    project, self.timeout)
                return self.SimpleLinks(links, stale=True)
            raise self.UpstreamError(
                f"timeout after {self.timeout} seconds while getting data for {project!r}")

        if self.offline and links is None:
            raise self.UpstreamError("offline mode")
        if self.offline or not is_expired:
            if self.offline and project not in self._offline_logging:
                threadlog.debug(
                    "using stale links for %r due to offline mode", project)
                self._offline_logging.add(project)
            return self.SimpleLinks(links, stale=True)

        if links is None:
            is_retrieval_expired = self.cache_retrieve_times.is_expired(
                project, self.cache_expiry)
            if not is_retrieval_expired:
                raise self.UpstreamNotFoundError(
                    "cached not found for project %s" % project)
            else:
                exists = self.has_project_perstage(project)
                if exists is unknown and self.no_project_list:
                    pass
                elif not exists:
                    # immediately cache the not found with no ETag
                    self.cache_retrieve_times.refresh(project, None)
                    raise self.UpstreamNotFoundError(
                        "project %s not found" % project)

        newlinks_future = self.xom.create_future()
        # we need to set this up here, as these access the database and
        # the async loop has no transaction
        # we don't resolve the key here, as _async_fetch_releaselinks runs
        # in a separate thread and only needs access to the relpath
        index_key = self.keyfs.INDEX(user=self.user.name, index=self.index)
        _key_from_link = partial(key_from_link, self.keyfs, index_key=index_key)
        try:
            self.xom.run_coroutine_threadsafe(
                self._async_fetch_releaselinks(
                    newlinks_future,
                    project,
                    mirrorlinks.get_cache_info(),
                    _key_from_link,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            if not self.xom.is_replica():
                # we process the future in the background
                # but only on primary, the replica will get the update
                # via the replication thread
                self.xom.create_task(
                    self._update_simplelinks_in_future(newlinks_future, project, lock.defer()))
            if links is not None:
                threadlog.warn(
                    "serving stale links for %r, getting data timed out after %s seconds",
                    project, self.timeout)
                return self.SimpleLinks(links, stale=True)
            raise self.UpstreamError(
                f"timeout after {self.timeout} seconds while getting data for {project!r}")
        except self.UpstreamNotModified as e:
            if links is not None:
                # immediately update the cache
                self.cache_retrieve_times.refresh(project, e.etag)
                return self.SimpleLinks(links)
            if e.etag is None:
                threadlog.error(
                    "server returned 304 Not Modified, but we have no links")
                raise
            # should not happen, but clear ETag and try again
            self.cache_retrieve_times.expire(project, etag=None)
            return self.get_simplelinks_perstage(project)
        except (self.UpstreamNotFoundError, self.UpstreamError) as e:
            # if we have an old result, return it. While this will
            # miss the rare event of actual project deletions it allows
            # to stay resilient against server misconfigurations.
            if links is not None:
                threadlog.warn(
                    "serving stale links, because of exception %s",
                    lazy_format_exception(e))
                return self.SimpleLinks(links, stale=True)
            raise

        info = newlinks_future.result()

        newlinks = join_links_data(
            info["key_hrefs"], info["requires_python"], info["yanked"])
        if links is not None and set(links) == set(newlinks):
            # no changes
            self.cache_retrieve_times.refresh(project, info["cache_info"]["etag"])
            return self.SimpleLinks(links)

        return self._update_simplelinks(project, info, links, newlinks)

    def has_project_perstage(self, project):
        project = normalize_name(project)
        if self.is_project_cached(project):
            return True
        if self.no_project_list:
            if project in self._stale_list_projects_perstage():
                return True
            return unknown
        # recheck full project list while abiding to expiration etc
        # use the internal method to avoid a copy
        return project in self._list_projects_perstage()

    def has_version_perstage(self, project: str, version: str) -> bool | Unknown:
        verdata = self.get_versiondata_perstage(project, version, with_elinks=False)
        return True if "version" in verdata else unknown

    def list_versions_perstage(self, project):
        try:
            return set(x.version for x in self.get_simplelinks_perstage(project))
        except self.UpstreamNotFoundError:
            return set()

    def get_last_project_change_serial_perstage(self, project, at_serial=None):
        project = normalize_name(project)
        tx = self.keyfs.tx
        if at_serial is None:
            at_serial = tx.at_serial
        (last_serial, projectname_ulid, projectname) = tx.get_last_serial_and_value_at(
            self.key_projectname(project), at_serial
        )
        if projectname in (absent, deleted):
            # the whole project never existed or was deleted
            return last_serial
        for mirrorfile_key in tx.conn.iter_ulidkeys_at_serial(
            (self.key_mirrorfile(project),),
            at_serial=at_serial,
            fill_cache=False,
            with_deleted=True,
        ):
            last_serial = max(last_serial, mirrorfile_key.last_serial)
            if last_serial >= at_serial:
                return last_serial
        return last_serial

    def _get_elinks(
        self, project: str, version: str, *, rel: Rel | None = None
    ) -> list:
        if rel not in (Rel.ReleaseFile, None):
            return []
        verdata = self.get_versiondata_perstage(project, version, with_elinks=True)
        return verdata["+elinks"]

    def get_versiondata_perstage(self, project, version, *, with_elinks=True):
        # we do not use normalize_name name here, so the returned data
        # contains whatever this method was called with, which is hopefully
        # the title from the project list
        verdata: dict[str, Any] = {}
        for sm in self.get_simplelinks_perstage(project):
            link_version = sm.version
            if version == link_version:
                if not verdata:
                    verdata['name'] = project
                    verdata['version'] = version
                if sm.require_python is not None:
                    verdata['requires_python'] = sm.require_python
                if sm.yanked is not None and sm.yanked is not False:
                    verdata['yanked'] = sm.yanked
                if with_elinks:
                    elinks = verdata.setdefault("+elinks", [])
                    elinks.append(
                        dict(rel=Rel.ReleaseFile, entrypath=sm.path, hashes=sm.hashes)
                    )
        return ensure_deeply_readonly(verdata)


class MirrorCustomizer(BaseStageCustomizer):
    pass


@hookimpl
def devpiserver_get_stage_customizer_classes():
    # prevent plugins from installing their own under the reserved names
    return [
        ("mirror", MirrorCustomizer)]


class ProjectNamesCache:
    """ Helper class for maintaining project names from a mirror. """
    _data: set[NormalizedName]
    _etag: str | None
    _lock: threading.RLock
    _timestamp: float

    def __init__(self):
        self._lock = threading.RLock()
        self._timestamp = -1
        self._data = set()
        self._etag = None

    def exists(self):
        with self._lock:
            return self._timestamp != -1

    def expire(self):
        with self._lock:
            self._timestamp = 0

    def is_expired(self, expiry_time):
        with self._lock:
            return (time.time() - self._timestamp) >= expiry_time

    def get(self):
        return self._data

    def get_etag(self):
        return self._etag

    def add(self, project):
        """ Add project to cache. """
        with self._lock:
            self._data.add(normalize_name(project))

    def discard(self, project):
        """ Remove project from cache. """
        with self._lock:
            self._data.discard(normalize_name(project))

    def set(self, data, etag):
        """ Set data and update timestamp. """
        with self._lock:
            if data != self._data:
                assert isinstance(data, set)
                if len(data):
                    assert isinstance(next(iter(data)), NormalizedName)
                self._data = data
            self.mark_current(etag)

    def mark_current(self, etag):
        with self._lock:
            self._timestamp = time.time()
            self._etag = etag


class ProjectUpdateInnerLock:
    # this is the object which is stored in the cache
    # it is needed to add the is_from_current_thread method
    # and to allow the WeakValueDictionary work correctly

    __slots__ = (
        '__weakref__', 'acquire', 'locked', 'release', 'thread_ident')

    def __init__(self):
        self.thread_ident = threading.get_ident()
        lock = threading.Lock()
        self.acquire = lock.acquire
        self.locked = lock.locked
        self.release = lock.release

    def is_from_current_thread(self):
        return self.thread_ident == threading.get_ident()


class ProjectUpdateLock:
    # this is a wrapper around ProjectUpdateInnerLock to allow
    # the WeakValueDictionary to work correctly

    def __init__(self, project, lock):
        self.lock = lock
        self.locked = lock.locked
        self.project = project

    def acquire(self, timeout):
        threadlog.debug("Acquiring lock (%r) for %r", self.lock, self.project)
        result = self.lock.acquire(timeout=timeout)
        return result

    def defer(self):
        lock = self.lock
        self.lock = None
        return lock

    def is_from_current_thread(self):
        if self.lock is not None:
            return self.lock.is_from_current_thread()
        return False

    def release(self):
        if self.lock is not None:
            self.lock.release()
            self.lock = None

    def __repr__(self):
        return f"<{self.__class__.__name__} project={self.project!r} lock={self.lock!r}>"


class ProjectUpdateCache:
    """ Helper class to manage when we last updated something project specific. """
    _project2lock: weakref.WeakValueDictionary
    _project2time: dict

    def __init__(self):
        self._project2time = {}
        self._project2lock = weakref.WeakValueDictionary()

    def is_expired(self, project, expiry_time):
        project = str(project)
        (t, etag) = self._project2time.get(project, (None, None))
        if t is not None:
            return (time.time() - t) >= expiry_time
        return True

    def get_etag(self, project):
        project = str(project)
        (t, etag) = self._project2time.get(project, (None, None))
        return etag

    def get_timestamp(self, project):
        project = str(project)
        (ts, etag) = self._project2time.get(project, (-1, None))
        return ts

    def refresh(self, project, etag):
        project = str(project)
        self._project2time[project] = (time.time(), etag)

    def expire(self, project, etag=None):
        project = str(project)
        if etag is None:
            self._project2time.pop(project, None)
        else:
            self._project2time[project] = (0, etag)

    def acquire(self, project, timeout=-1):
        project = str(project)
        lock = ProjectUpdateLock(
            project,
            self._project2lock.setdefault(project, ProjectUpdateInnerLock()))
        if lock.locked() and lock.is_from_current_thread():
            # remove inner lock, so this is just a dummy
            lock.lock = None
            return lock
        if lock.acquire(timeout=timeout):
            return lock
        self._project2lock.pop(project, None)
        return None

    def release(self, project):
        project = str(project)
        lock = self._project2lock.pop(project, None)
        if lock is not None and lock.locked():
            lock.release()
