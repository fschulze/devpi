from __future__ import annotations

from .config import ACLList
from .config import RemoveValue
from .config import ensure_acl_list
from .config import ensure_boolean
from .config import ensure_list
from .config import get_principals
from .exceptions import InvalidIndex
from .exceptions import InvalidIndexconfig
from .exceptions import InvalidUser
from .exceptions import MissesRegistration
from .exceptions import MissesVersion
from .exceptions import NonVolatile
from .exceptions import NotFound
from .exceptions import ReadonlyIndex
from .exceptions import UpstreamError
from .exceptions import UpstreamNotFoundError
from .exceptions import UpstreamNotModified
from .inheritance import IndexBases
from .inheritance import check_upstream_error
from .links import ELink
from .links import LinkStore
from .links import MutableLinkStore
from .links import Rel
from .links import SimpleLinks
from abc import abstractmethod
from devpi_common.metadata import get_latest_version
from devpi_common.types import cached_property
from devpi_server.filestore import AbsPath
from devpi_server.filestore import FileEntry
from devpi_server.log import threadlog
from devpi_server.markers import Absent
from devpi_server.markers import Deleted
from devpi_server.markers import unknown
from devpi_server.normalized import normalize_name
from devpi_server.readonly import ensure_deeply_readonly
from lazy import lazy
from pyramid.authorization import Allow
from typing import TYPE_CHECKING
from typing import overload
import getpass
import json
import warnings


if TYPE_CHECKING:
    from .root import RootModel
    from .schema import Schema
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from devpi_server.filestore import BaseFileEntry
    from devpi_server.filestore import Digests
    from devpi_server.interfaces import ContentOrFile
    from devpi_server.keyfs import KeyFS
    from devpi_server.keyfs_types import LocatedKey
    from devpi_server.keyfs_types import SearchKey
    from devpi_server.keyfs_types import ULIDKey
    from devpi_server.main import XOM
    from devpi_server.markers import Unknown
    from devpi_server.normalized import NormalizedName
    from devpi_server.readonly import DictViewReadonly
    from typing import Any


def apply_filter_iter(items, filter_iter):
    for item in items:
        if next(filter_iter, True):
            yield item


def run_passwd(root, username):
    user = root.get_user(username)
    log = threadlog
    if user is None:
        log.error("user %r not found", username)
        return 1
    for _i in range(3):
        pwd = getpass.getpass("enter password for %s: " % user.name)
        pwd2 = getpass.getpass("repeat password for %s: " % user.name)
        if pwd != pwd2:
            log.error("password don't match")
        else:
            break
    else:
        log.error("no password set")
        return 1
    user.modify(password=pwd)
    return None


class BaseIndex:
    keyfs: KeyFS[Schema]
    offline: bool

    def __init__(
        self,
        xom: XOM,
        username: str,
        index: str,
        ixconfig: DictViewReadonly[str, Any],
        customizer_cls: type,
    ) -> None:
        self.xom = xom
        self.username = username
        self.index = index
        self.name = username + "/" + index
        self.ixconfig = ixconfig
        self.customizer = customizer_cls(self)
        # the following attributes are per-xom singletons
        self.keyfs = xom.keyfs
        self.filestore = xom.filestore
        self.key_index = self.keyfs.schema.INDEX.locate(user=username, index=index)
        self.key_project = self.keyfs.schema.PROJECT.search(user=username, index=index)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"

    @lazy
    def index_bases(self) -> IndexBases:
        return self.get_index_bases()

    @cached_property
    def index_type(self):
        return self.ixconfig["type"]

    @overload
    def key_simpledata(
        self, project: NormalizedName | str, version_filename: tuple[str, str]
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_simpledata(
        self, project: NormalizedName | str, version_filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_simpledata(
        self,
        project: NormalizedName | str,
        version_filename: tuple[str, str] | None = None,
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.SIMPLEDATA
        (version, filename) = (
            (None, None) if version_filename is None else version_filename
        )
        (kw, meth) = (
            ({}, key.search)
            if version is None or filename is None
            else (dict(version=version, filename=filename), key.locate)
        )
        return meth(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    @overload
    def key_version(
        self, project: NormalizedName | str, version: str
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_version(
        self, project: NormalizedName | str, version: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_version(
        self, project: NormalizedName | str, version: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.VERSION
        (kw, meth) = (
            ({}, key.search) if version is None else (dict(version=version), key.locate)
        )
        return meth(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    @property
    def model(self) -> RootModel:
        return self.xom.model

    @abstractmethod
    def add_project_name(self, project: NormalizedName | str) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def no_project_list(self) -> bool:
        raise NotImplementedError

    def get_indexconfig_from_kwargs(self, **kwargs):  # noqa: PLR0912
        """Normalizes values and validates keys.

        Returns the parts touched by kwargs as dict.
        This is not the complete index configuration."""
        index_type = self.index_type
        ixconfig = {}
        # get known keys and validate them
        stage_keys = set(self.get_possible_indexconfig_keys())
        customizer_keys = set(self.customizer.get_possible_indexconfig_keys())
        conflicting = stage_keys.intersection(customizer_keys)
        if conflicting:
            raise ValueError(
                "The stage customizer for '%s' defines keys which conflict "
                "with existing index configuration keys: %s"
                % (index_type, ", ".join(sorted(conflicting)))
            )
        # prevent default values from being removed
        for key, _value in self.get_default_config_items():
            if kwargs.get(key) is RemoveValue:
                raise InvalidIndexconfig("Default values can't be removed.")
        # now process any key known by the stage class
        for key in stage_keys:
            if key not in kwargs:
                continue
            value = kwargs.pop(key)
            if value is not RemoveValue:
                value = self.normalize_indexconfig_value(key, value)
                if value is None:
                    raise ValueError("The key '%s' wasn't processed." % (key))
            ixconfig[key] = value
        # prevent removal of defaults from the customizer class
        for key, _value in self.customizer.get_default_config_items():
            if kwargs.get(key) is RemoveValue:
                raise InvalidIndexconfig("Default values can't be removed.")
        # and process any key known by the customizer class
        for key in customizer_keys:
            if key not in kwargs:
                continue
            value = kwargs.pop(key)
            if value is not RemoveValue:
                value = self.customizer.normalize_indexconfig_value(key, value)
                if value is None:
                    raise ValueError("The key '%s' wasn't processed." % (key))
            ixconfig[key] = value
        # lastly we get additional default from the hook
        hooks = self.xom.config.hook
        for defaults in hooks.devpiserver_indexconfig_defaults(index_type=index_type):
            conflicting = stage_keys.intersection(defaults)
            if conflicting:
                raise ValueError(
                    "A plugin returned the following keys which conflict with "
                    "existing index configuration keys for '%s': %s"
                    % (index_type, ", ".join(sorted(conflicting)))
                )
            for key, value in defaults.items():
                new_value = kwargs.pop(key, value)
                if new_value is not RemoveValue:
                    if isinstance(value, bool):
                        new_value = ensure_boolean(new_value)
                    elif isinstance(value, ACLList):
                        new_value = ensure_acl_list(new_value)
                    elif isinstance(value, (list, tuple, set)):
                        new_value = ensure_list(new_value)
                ixconfig.setdefault(key, new_value)
        for key, value in list(kwargs.items()):
            if value is RemoveValue:
                ixconfig[key] = kwargs.pop(key)
        ixconfig["type"] = index_type
        return (ixconfig, kwargs)

    @abstractmethod
    def get_default_config_items(self) -> Sequence[tuple[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_possible_indexconfig_keys(self) -> Sequence:
        raise NotImplementedError

    @abstractmethod
    def get_versiondata_perstage(
        self,
        project: NormalizedName | str,
        version: str,
        *,
        with_elinks: bool = True,
    ) -> DictViewReadonly[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_projects_perstage(self) -> dict[str, NormalizedName | str]:
        raise NotImplementedError

    @abstractmethod
    def has_project_perstage(self, project: NormalizedName | str) -> bool | Unknown:
        raise NotImplementedError

    @abstractmethod
    def has_version_perstage(self, project: str, version: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_elink_from_entry(self, entry: BaseFileEntry) -> ELink | None:
        raise NotImplementedError

    @abstractmethod
    def normalize_indexconfig_value(self, key: str, value: Any) -> Any:
        raise NotImplementedError

    @cached_property
    def user(self):
        # only few methods need the user object.
        return self.model.get_user(self.username)

    @property
    def ixconfig_mutable(self) -> dict:
        return self.key_index.with_resolved_parent().get_mutable()

    def delete(self) -> None:
        self.model.delete_stage(self.username, self.index)

    def get_releaselinks(self, project):
        # compatibility access method used by devpi-web and tests
        project = normalize_name(project)
        try:
            return self._make_elinks(self.get_simplelinks(project))
        except self.UpstreamNotFoundError:
            return []

    def get_releaselinks_perstage(self, project: NormalizedName | str) -> list[ELink]:
        # compatibility access method for devpi-findlinks and possibly other plugins
        project = normalize_name(project)
        return self._make_elinks(self.get_simplelinks_perstage(project))

    def _make_elinks(self, simplelinks: SimpleLinks) -> list[ELink]:
        entries = {
            e.abspath: e
            for e in self.get_entries_for_entrypaths(l.abspath for l in simplelinks)
            if e is not None
        }
        return [
            ELink(
                entries[l.abspath],
                dict(
                    relpath=l.relpath,
                    hashes=l.hashes,
                    rel=Rel.ReleaseFile,
                    require_python=l.require_python,
                    yanked=l.yanked,
                ),
            )
            for l in simplelinks
        ]

    def get_linkstore_perstage(self, name, version):
        return LinkStore(self, name, version)

    def get_mutable_linkstore_perstage(self, name, version):
        if self.customizer.readonly:
            threadlog.warn("index is marked read only")
        return MutableLinkStore(self, name, version)

    def get_keys_for_entrypaths(
        self, entrypaths: Iterable[str]
    ) -> list[LocatedKey | None]:
        return [
            key
            if (
                key := self.keyfs.match_key(
                    AbsPath(entrypath.rsplit("#", 1)[0]),
                    self.keyfs.schema.FILE_NOHASH,
                    self.keyfs.schema.FILE,
                )
            )
            is None
            else key.with_resolved_parent()
            for entrypath in entrypaths
        ]

    def get_entries_for_keys(
        self, keys: Iterable[LocatedKey | None]
    ) -> list[FileEntry | None]:
        key_to_ulidkey: dict[LocatedKey, ULIDKey | Absent | Deleted] = dict(
            self.keyfs.tx.resolve_keys(
                (k for k in keys if k is not None),
                fetch=True,
                fill_cache=True,
                new_for_missing=False,
            )
        )
        return [
            None
            if key is None
            or (ulid_key := key_to_ulidkey.get(key)) is None
            or isinstance(ulid_key, (Absent, Deleted))
            else FileEntry(ulid_key)
            for key in keys
        ]

    def get_entries_for_entrypaths(
        self, entrypaths: Iterable[str]
    ) -> list[FileEntry | None]:
        keys = self.get_keys_for_entrypaths(entrypaths)
        return self.get_entries_for_keys(keys)

    def get_link_from_entrypath(self, entrypath):
        relpath = entrypath.rsplit("#", 1)[0]
        entry = self.xom.filestore.get_file_entry(relpath)
        if entry is None or entry.project is None:
            return None
        linkstore = self.get_linkstore_perstage(entry.project, entry.version)
        links = linkstore.get_links(entrypath=relpath)
        assert len(links) < 2
        return links[0] if links else None

    @abstractmethod
    def get_simplelinks_perstage(self, project: NormalizedName | str) -> SimpleLinks:
        raise NotImplementedError

    def store_toxresult(
        self,
        link: ELink,
        content_or_file: ContentOrFile,
        *,
        filename: str | None = None,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        if self.customizer.readonly:
            raise ReadonlyIndex("index is marked read only")
        assert not isinstance(content_or_file, dict)
        linkstore = self.get_mutable_linkstore_perstage(link.project, link.version)
        return linkstore.new_reflink(
            rel=Rel.ToxResult,
            content_or_file=content_or_file,
            for_link=link,
            filename=filename,
            hashes=hashes,
            last_modified=last_modified,
        )

    def get_toxresults(self, link):
        l = []
        linkstore = self.get_linkstore_perstage(link.project, link.version)
        for reflink in linkstore.get_links(rel=Rel.ToxResult, for_entrypath=link):
            with reflink.entry.file_open_read() as f:
                l.append(json.load(f))
        return l

    def filter_versions(
        self, project: NormalizedName | str, versions: set[str]
    ) -> set[str]:
        iterator = self.customizer.get_versions_filter_iter(project, versions)
        if iterator is None:
            return versions
        return set(apply_filter_iter(versions, iterator))

    def list_versions(self, project: NormalizedName | str) -> set[str]:
        project = normalize_name(project)
        versions = set()
        for stage in self.index_bases.get_mergeable_stages(project, "list_versions"):
            with check_upstream_error(self, stage) as checker:
                res = stage.list_versions_perstage(project)
            if checker.failed:
                continue
            versions.update(res)
        return self.filter_versions(project, versions)

    @abstractmethod
    def list_versions_perstage(self, project: str) -> set:
        raise NotImplementedError

    def get_latest_version(self, name, *, stable=False):
        return get_latest_version(
            self.filter_versions(name, self.list_versions(name)), stable=stable
        )

    def get_latest_version_perstage(self, name, *, stable=False):
        return get_latest_version(
            self.filter_versions(name, self.list_versions_perstage(name)), stable=stable
        )

    def get_versiondata(
        self, project: NormalizedName | str, version: str
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if not self.filter_versions(project, {version}):
            return result
        for stage in self.index_bases.get_mergeable_stages(
            normalize_name(project), "get_versiondata"
        ):
            with check_upstream_error(self, stage) as checker:
                res = stage.get_versiondata_perstage(project, version)
            if checker.failed:
                continue
            if res:
                if not result:
                    result.update(res)
                else:
                    l = result.setdefault("+shadowing", [])
                    l.append(res)
        return result

    def get_simplelinks(
        self, project: NormalizedName | str, *, sorted_links: bool = True
    ) -> SimpleLinks:
        """Return list of (key, href) tuples where "href" is a path
        to a file entry with "#" appended hash-specs or egg-ids
        and "key" is usually the basename of the link or else
        the egg-ID if the link points to an egg.
        """
        project = normalize_name(project)
        all_links = self.SimpleLinks([])
        seen = set()

        try:
            for stage in self.index_bases.get_mergeable_stages(
                project, "get_simplelinks"
            ):
                with check_upstream_error(self, stage) as checker:
                    res = stage.get_simplelinks_perstage(project)
                if checker.failed:
                    continue
                if res is not None:
                    res = self.SimpleLinks(res)
                    all_links.stale = all_links.stale or res.stale
                iterator = self.customizer.get_simple_links_filter_iter(project, res)
                if iterator is not None:
                    res = apply_filter_iter(res, iterator)
                for link_info in res:
                    key = link_info.key
                    if key not in seen:
                        seen.add(key)
                        all_links.append(link_info)
        except self.UpstreamNotFoundError:
            return self.SimpleLinks([])

        if sorted_links:
            all_links.sort(reverse=True)
        return all_links

    def get_mirror_whitelist_info(
        self, project: NormalizedName | str
    ) -> dict[str, bool | Unknown | str | None]:
        warnings.warn(
            "The 'get_mirror_whitelist_info' method is deprecated, "
            "use 'index_bases.get_inheritance_infos()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        project = normalize_name(project)
        inheritance_infos = self.index_bases.get_inheritance_infos(project)
        return dict(
            has_mirror_base=inheritance_infos.has_project_from_mirror,
            blocked_by_mirror_whitelist=inheritance_infos.blocked_mirror_name,
        )

    def filter_projects(self, projects):
        iterator = self.customizer.get_projects_filter_iter(projects)
        if iterator is None:
            return projects
        return frozenset(apply_filter_iter(projects, iterator))

    def has_project(self, project: NormalizedName | str) -> bool | Unknown:
        if not self.filter_projects([project]):
            return False
        for stage in self.sro():
            res = stage.has_project_perstage(project)
            if res is unknown:
                return res
            if res:
                return True
        return False

    def list_projects(self) -> list[tuple[BaseIndex, dict[str, NormalizedName | str]]]:
        result = []
        for stage in self.sro():
            projects = stage.list_projects_perstage()
            result.append((stage, self.filter_projects(projects)))
        return result

    def _modify(self, **kw):
        if "type" in kw and self.index_type != kw["type"]:
            raise InvalidIndexconfig(["the 'type' of an index can't be changed"])
        kw.pop("type", None)
        kw.pop("projects", None)  # we never modify this from the outside
        keep_unknown = kw.pop("_keep_unknown", False)
        (ixconfig, unknown) = self.get_indexconfig_from_kwargs(**kw)
        if unknown:
            if keep_unknown:
                # used to import data when plugins aren't installed anymore
                ixconfig.update(unknown)
            else:
                raise InvalidIndexconfig(
                    [
                        "indexconfig got unexpected keyword arguments: %s"
                        % ", ".join("%s=%s" % x for x in unknown.items())
                    ]
                )
        # modify indexconfig
        key_index = self.key_index.with_resolved_parent()
        with key_index.update() as newconfig:
            oldconfig = dict(self.ixconfig)
            for key, value in list(ixconfig.items()):
                if value is RemoveValue:
                    newconfig.pop(key, None)
                    ixconfig.pop(key)
            newconfig.update(ixconfig)
            self.customizer.validate_config(oldconfig, newconfig)
            self.ixconfig = ensure_deeply_readonly(newconfig)
            return newconfig

    def modify(self, **kw):
        lazy.invalidate(self, "index_bases")
        newconfig = self._modify(**kw)
        threadlog.info("modified index %s: %s", self.name, newconfig)
        return newconfig

    def get_index_bases(self) -> IndexBases:
        return IndexBases(
            self,
            devpiserver_sro_skip=self.xom.config.hook.devpiserver_sro_skip,
            model=self.model,
        )

    def op_sro(self, opname, **kw):
        warnings.warn(
            "The 'op_sro' method is deprecated, "
            "use 'index_bases.iter_stages()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "project" in kw:
            project = normalize_name(kw["project"])
            if not self.filter_projects([project]):
                return
        for stage in self.sro():
            yield stage, getattr(stage, opname)(**kw)

    def op_sro_check_mirror_whitelist(self, opname, **kw):
        warnings.warn(
            "The 'op_sro_check_mirror_whitelist' method is deprecated, "
            "use 'index_bases.get_mergeable_stages' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        project = normalize_name(kw["project"])
        if not self.filter_projects([project]):
            return
        for stage in self.index_bases.get_mergeable_stages(project, opname):
            with check_upstream_error(self, stage) as checker:
                res = getattr(stage, opname)(**kw)
            if checker.failed:
                continue
            yield stage, res

    def sro(self) -> Iterator[BaseIndex]:
        """return stage resolution order."""
        return self.index_bases.iter_stages()

    def __acl__(self):
        permissions = (
            "pkg_read",
            "toxresult_upload",
            "upload",
            "index_delete",
            "index_modify",
            "del_entry",
            "del_project",
            "del_verdata",
        )
        restrict_modify = self.xom.config.restrict_modify
        acl = []
        for permission in permissions:
            method_name = "get_principals_for_%s" % permission
            method = getattr(self.customizer, method_name, None)
            if not callable(method):
                msg = f"The attribute {method_name} with value {method!r} of {self.customizer!r} is not callable."
                raise AttributeError(msg)  # noqa: TRY004
            for principal in get_principals(method(restrict_modify=restrict_modify)):
                acl.append((Allow, principal, permission))
                if permission == "upload":
                    # add pypi_submit alias for BBB
                    acl.append((Allow, principal, "pypi_submit"))
        return acl

    InvalidIndex = InvalidIndex
    InvalidIndexconfig = InvalidIndexconfig
    InvalidUser = InvalidUser
    NotFound = NotFound
    UpstreamError = UpstreamError
    UpstreamNotFoundError = UpstreamNotFoundError
    UpstreamNotModified = UpstreamNotModified
    MissesRegistration = MissesRegistration
    MissesVersion = MissesVersion
    NonVolatile = NonVolatile
    SimpleLinks = SimpleLinks
