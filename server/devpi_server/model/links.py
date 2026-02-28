from __future__ import annotations

from .exceptions import MissesRegistration
from .exceptions import NonVolatile
from devpi_common.metadata import parse_version
from devpi_common.metadata import splitbasename
from devpi_common.url import URL
from devpi_server.filestore import AbsPath
from devpi_server.filestore import Digests
from devpi_server.filestore import FileEntry
from devpi_server.filestore import MutableFileEntry
from devpi_server.keyfs_types import RelPath
from devpi_server.log import threadlog
from devpi_server.markers import NotSet
from devpi_server.markers import absent
from devpi_server.markers import notset
from devpi_server.normalized import normalize_name
from devpi_server.readonly import ensure_deeply_readonly
from devpi_server.readonly import get_mutable_deepcopy
from functools import total_ordering
from pathlib import Path
from time import gmtime
from time import strftime
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import TypedDict
from typing import overload
import enum


if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import MutableSequence
    from collections.abc import Sequence
    from devpi_common.metadata import Version
    from devpi_server.filestore import BaseFileEntry
    from devpi_server.filestore import FileStore
    from devpi_server.interfaces import ContentOrFile
    from devpi_server.keyfs_types import LocatedKey
    from devpi_server.keyfs_types import SearchKey
    from devpi_server.normalized import NormalizedName
    from devpi_server.readonly import DictViewReadonly
    from typing import Any
    from typing import Literal
    from typing import NotRequired

    RequiresPython = str | None
    Yanked = Literal[True] | str | None
    JoinedLink = tuple[str, str, RequiresPython, Yanked]


class FileLogEntry(TypedDict):
    what: str
    who: str | None
    when: str
    count: NotRequired[int]
    dst: NotRequired[str]
    src: NotRequired[str]


class Rel(enum.StrEnum):
    DocZip = "doczip"
    ReleaseFile = "releasefile"
    ToxResult = "toxresult"


def linkdictprop(name, default=notset):
    def fget(self):
        try:
            return self.linkdict[name]
        except KeyError as e:
            if isinstance(default, NotSet):
                raise AttributeError(name) from e  # noqa: TRY004 - API
            return default

    return property(fget)


F = TypeVar("F", FileEntry, MutableFileEntry)


class ELink(Generic[F]):
    """model Link using entrypathes for referencing."""

    __slots__ = (
        "_basename",
        "_entry",
        "index",
        "linkdict",
        "project",
        "user",
        "version",
    )

    _entry: F
    _log: MutableSequence[FileLogEntry] = linkdictprop("log")
    index_relpath = linkdictprop("relpath")
    for_relpath = linkdictprop("for_relpath", default=None)
    rel = linkdictprop("rel", default=None)
    require_python = linkdictprop("require_python")
    yanked = linkdictprop("yanked")

    def __init__(self, entry: F, linkdict: dict) -> None:
        assert "hash_spec" not in linkdict
        self._entry = entry
        self.linkdict = linkdict
        if self.for_relpath is not None:
            assert "#" not in self.for_relpath
        self.user = entry.user
        self.index = entry.index
        self.project = entry.project
        self.version = entry.version

    @property
    def best_available_hash_type(self):
        return self.hashes.best_available_type

    @property
    def best_available_hash_spec(self):
        return self.hashes.best_available_spec

    @property
    def best_available_hash_value(self):
        return self.hashes.best_available_value

    @property
    def hashes(self):
        return self.entry.hashes

    @property
    def basename(self):
        _basename = getattr(self, "_basename", None)
        if _basename is None:
            _basename = self._basename = Path(self.relpath).name
        return _basename

    @property
    def entrypath(self):
        entrypath = self.relpath
        hash_spec = self.best_available_hash_spec
        if hash_spec:
            return f"{entrypath}#{hash_spec}"
        return entrypath

    def matches_hashes(self, hashes):
        return self.hashes == hashes

    @property
    def for_entrypath(self) -> RelPath:
        return RelPath(f"{self.user}/{self.index}/{self.for_relpath}")

    @property
    def relpath(self) -> RelPath:
        return RelPath(f"{self.user}/{self.index}/{self.index_relpath}")

    def __repr__(self) -> str:
        return "<ELink rel=%r entrypath=%r>" % (self.rel, self.entrypath)

    @property
    def entry(self) -> F:
        if isinstance(self._entry, NotSet):
            raise RuntimeError  # noqa: TRY004
        return self._entry

    def add_log(
        self,
        what: str,
        who: str | None,
        *,
        count: int | None = None,
        dst: str | None = None,
        src: str | None = None,
        when: str | None = None,
    ) -> None:
        d = FileLogEntry(
            what=what, who=who, when=strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        )
        if count is not None:
            d["count"] = count
        if dst is not None:
            d["dst"] = dst
        if src is not None:
            d["src"] = src
        if when is not None:
            assert isinstance(when, str)
            d["when"] = when
        self._log.append(d)

    def add_logs(self, logs: Iterable[FileLogEntry | dict]) -> None:
        for log in logs:
            unknown_keys = set(log).difference(
                FileLogEntry.__required_keys__ | FileLogEntry.__optional_keys__
            )
            if unknown_keys:
                msg = f"Unknown keys {', '.join(sorted(unknown_keys))} for FileLogEntry"
                raise ValueError(msg)
            self.add_log(
                log["what"],
                log["who"],
                count=log.get("count"),
                dst=log.get("dst"),
                src=log.get("src"),
                when=log["when"],
            )

    def get_logs(self) -> list[FileLogEntry]:
        return list(getattr(self, "_log", []))


class LinkStore:
    filestore: FileStore

    def __init__(self, stage, project, version):
        self.stage = stage
        self.filestore = stage.filestore
        self.project = normalize_name(project)
        self.version = version
        if not self.stage.has_version_perstage(project, version):
            raise MissesRegistration(
                "%s-%s on stage %s at %s",
                project,
                version,
                stage.name,
                stage.keyfs.tx.at_serial,
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.project} {self.stage.name} {self.version}>"

    def _get_elinks(self, rel):
        return self.stage._get_elinks(self.project, self.version, rel=rel)

    def _get_entries_for_entrypaths(self, elinks):
        return self.stage.get_entries_for_entrypaths(elinks)

    def get_links(
        self,
        rel: Rel | None = None,
        basename: str | None = None,
        entrypath: str | None = None,
        for_entrypath: ELink | str | None = None,
    ) -> list[ELink]:
        if entrypath is not None:
            assert "#" not in entrypath

        if isinstance(for_entrypath, ELink):
            for_entrypath = for_entrypath.relpath
        elif for_entrypath is not None:
            assert "#" not in for_entrypath

        def fil(elink):
            return (
                (not basename or basename == Path(elink["relpath"]).name)
                and (
                    not entrypath or entrypath in (elink["entrypath"], elink["relpath"])
                )
                and (not for_entrypath or for_entrypath == elink.get("for_entrypath"))
            )

        elinks = [elink for elink in self._get_elinks(rel) if fil(elink)]
        entries = {
            e.abspath: e
            for e in self._get_entries_for_entrypaths(l["entrypath"] for l in elinks)
            if e is not None
        }
        return [ELink(entries[l["entrypath"]], l) for l in elinks]

    @property
    def metadata(self):
        verdata = self.stage.get_versiondata_perstage(
            self.project, self.version, with_elinks=False
        )
        return ensure_deeply_readonly(
            {k: v for k, v in verdata.items() if not k.startswith("+")}
        )


class MutableLinkStore(LinkStore):
    def create_linked_entry(
        self,
        rel: Rel,
        basename: str,
        content_or_file: ContentOrFile,
        *,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        overwrite = None
        for link in self.get_links(rel=rel, basename=basename):
            if not self.stage.ixconfig.get("volatile"):
                exc = NonVolatile(
                    "rel=%s basename=%s on stage %s" % (rel, basename, self.stage.name)
                )
                exc.link = link
                raise exc
            assert overwrite is None
            overwrite = sum(
                x.get("count", 0)
                for x in link.get_logs()
                if x.get("what") == "overwrite"
            )
        self.remove_links(rel=rel, basename=basename)
        file_entry = self._create_file_entry(
            basename, content_or_file, hashes=hashes, last_modified=last_modified
        )
        link = self._add_link_to_file_entry(rel, file_entry)
        if overwrite is not None:
            link.add_log("overwrite", None, count=overwrite + 1)
        return link

    def _get_entries_for_entrypaths(self, elinks):
        return self.stage.get_mutable_entries_for_entrypaths(elinks)

    def key_doczip(self) -> LocatedKey[dict, DictViewReadonly]:
        return self.stage.key_doczip(self.project, self.version)

    @overload
    def key_simpledata(self, filename: str) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_simpledata(
        self, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_simpledata(
        self, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        if filename is None:
            return self.stage.key_simpledata(self.project)
        return self.stage.key_simpledata(self.project, (self.version, filename))

    @overload
    def key_toxresult(self, filename: str) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_toxresult(
        self, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_toxresult(
        self, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        return self.stage.key_toxresult(self.project, self.version, filename)

    @overload
    def key_versionfile(self, filename: str) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_versionfile(
        self, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_versionfile(
        self, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        return self.stage.key_versionfile(self.project, self.version, filename)

    def _get_elinks(self, rel):
        return get_mutable_deepcopy(super()._get_elinks(rel))

    @property
    def metadata(self):
        return get_mutable_deepcopy(super().metadata)

    def new_reflink(
        self,
        rel: Rel,
        content_or_file: ContentOrFile,
        for_link: ELink,
        *,
        filename: str | None = None,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        links = self.get_links(entrypath=for_link.relpath)
        assert len(links) == 1, f"need exactly one reference, got {links}"
        assert for_link.relpath == links[0].relpath
        base_entry = for_link.entry
        if filename is None:
            other_reflinks = self.get_links(rel=rel, for_entrypath=for_link.relpath)
            timestamp = strftime("%Y%m%d%H%M%S", gmtime())
            filename = "%s.%s-%s-%d" % (
                base_entry.basename,
                rel,
                timestamp,
                len(other_reflinks),
            )
        entry = self._create_file_entry(
            filename,
            content_or_file,
            hashes=hashes,
            last_modified=last_modified,
            ref_hash_spec=base_entry.ref_hash_spec,
        )
        return self._add_link_to_file_entry(rel, entry, for_link=for_link)

    def remove_links(self, rel=None, basename=None, for_entrypath=None):
        del_links = self.get_links(
            rel=rel, basename=basename, for_entrypath=for_entrypath
        )
        was_deleted = []
        key_doczip = self.key_doczip().with_resolved_parent()
        key_toxresult = self.key_toxresult().with_resolved_parent()
        key_versionfile = self.key_versionfile().with_resolved_parent()
        key_simpledata = self.key_simpledata().with_resolved_parent()
        version = self.version
        if del_links:
            for link in del_links:
                filename = link.entry.basename
                link.entry.delete()
                match link.rel:
                    case Rel.DocZip:
                        key_doczip.delete()
                    case Rel.ToxResult:
                        key_toxresult(filename).delete()
                    case Rel.ReleaseFile:
                        key_versionfile(filename).delete()
                        key_simpledata(version=version, filename=filename).delete()
                    case _:
                        raise RuntimeError(link.rel)
                was_deleted.append(link.relpath)
                threadlog.info("deleted %r link %s", link.rel, link.relpath)
        has_versionfiles = next(key_versionfile.iter_ulidkeys(), absent) is not absent
        if was_deleted and has_versionfiles:
            for relpath in was_deleted:
                self.remove_links(for_entrypath=relpath)

    def _create_file_entry(
        self,
        basename: str,
        content_or_file: ContentOrFile,
        *,
        hashes: Digests,
        last_modified: str | None = None,
        ref_hash_spec: str | None = None,
    ) -> MutableFileEntry:
        entry = self.filestore.store(
            user=self.stage.username,
            index=self.stage.index,
            basename=basename,
            content_or_file=content_or_file,
            last_modified=last_modified,
            ref_hash_spec=ref_hash_spec,
            hashes=hashes,
        )
        entry.project = self.project
        entry.version = self.version
        return entry

    def _add_link_to_file_entry(
        self, rel: Rel, file_entry: BaseFileEntry, for_link: ELink | str | None = None
    ) -> ELink:
        new_linkdict = {
            "relpath": file_entry.index_relpath,
            "log": [],
        }
        if for_link:
            assert isinstance(for_link, ELink)
            new_linkdict["for_relpath"] = for_link.index_relpath
        match rel:
            case Rel.DocZip:
                key = self.key_doczip().with_resolved_parent()
            case Rel.ToxResult:
                key = self.key_toxresult(file_entry.basename).with_resolved_parent()
            case Rel.ReleaseFile:
                key = self.key_versionfile(file_entry.basename).with_resolved_parent()
            case _:
                raise RuntimeError(rel)
        if key.exists():
            raise RuntimeError
        key.set(new_linkdict)
        threadlog.info("added %r link %s", rel, file_entry.relpath)
        return ELink(file_entry, new_linkdict)


@total_ordering
class SimplelinkMeta:
    """helper class to provide information for items from get_simplelinks()"""

    __slots__ = (
        "__basename",
        "__cmpval",
        "__ext",
        "__hashes",
        "__index",
        "__name",
        "__relpath",
        "__url",
        "__user",
        "__version",
        "core_metadata",
        "href",
        "key",
        "require_python",
        "yanked",
    )
    __cmpval: tuple | NotSet
    __hashes: Digests | NotSet
    __relpath: str | NotSet

    def __init__(
        self,
        link_info: tuple[str, str, RequiresPython, Yanked],
        *,
        core_metadata: bool = False,
    ) -> None:
        self.__basename = notset
        self.__cmpval = notset
        self.__ext = notset
        self.__hashes = notset
        self.__index = notset
        self.__name = notset
        self.__relpath = notset
        self.__url = notset
        self.__user = notset
        self.__version = notset
        self.core_metadata = False
        (self.key, self.href, self.require_python, self.yanked) = link_info
        if core_metadata and self.basename.endswith(".whl"):
            self.core_metadata = True

    def __hash__(self) -> int:
        return hash(
            (
                self.__basename,
                self.__cmpval,
                self.__ext,
                self.__hashes,
                self.__index,
                self.__name,
                self.__relpath,
                self.__url,
                self.__user,
                self.__version,
                self.core_metadata,
                self.href,
                self.key,
                self.require_python,
                self.yanked,
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            other = other.cmpval
        return self.cmpval == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            other = other.cmpval
        return self.cmpval < other

    def __splitbasename(self) -> None:
        (self.__name, self.__version, self.__ext) = splitbasename(
            self.basename, checkarch=False
        )

    def __parse_url(self) -> None:
        url = URL(self.href)
        self.__basename = url.basename
        self.__hashes = Digests()
        if hash_type := url.hash_type:
            self.__hashes[hash_type] = url.hash_value
        parts = url.path.split("/")
        self.__user = parts[0]
        self.__index = parts[1]
        self.__relpath = "/".join(parts[2:])

    @property
    def abspath(self) -> AbsPath:
        return AbsPath(f"{self.user}/{self.index}/{self.relpath}")

    @property
    def basename(self) -> str:
        if self.__basename is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__basename, str)
        return self.__basename

    @property
    def hashes(self) -> Digests:
        if self.__hashes is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__hashes, Digests)
        return self.__hashes

    @property
    def index(self) -> str:
        if self.__index is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__index, str)
        return self.__index

    @property
    def path(self) -> str:
        if self.__relpath is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__index, str)
            assert isinstance(self.__relpath, str)
            assert isinstance(self.__user, str)
        return f"{self.__user}/{self.__index}/{self.__relpath}"

    @property
    def relpath(self) -> str:
        if self.__relpath is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__relpath, str)
        return self.__relpath

    @property
    def user(self) -> str:
        if self.__user is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__user, str)
        return self.__user

    @property
    def name(self) -> str:
        if self.__name is notset:
            self.__splitbasename()
        if TYPE_CHECKING:
            assert isinstance(self.__name, str)
        return self.__name

    @property
    def version(self) -> str:
        if self.__version is notset:
            self.__splitbasename()
        if TYPE_CHECKING:
            assert isinstance(self.__version, str)
        return self.__version

    @property
    def ext(self) -> str:
        if self.__ext is notset:
            self.__splitbasename()
        if TYPE_CHECKING:
            assert isinstance(self.__ext, str)
        return self.__ext

    @property
    def cmpval(self) -> tuple[Version, NormalizedName, str]:
        if self.__cmpval is notset:
            self.__cmpval = (
                parse_version(self.version),
                normalize_name(self.name),
                self.ext,
            )
        if TYPE_CHECKING:
            assert isinstance(self.__cmpval, tuple)
            assert isinstance(self.__cmpval[0], Version)
            assert isinstance(self.__cmpval[1], NormalizedName)
            assert isinstance(self.__cmpval[2], str)
        return self.__cmpval

    def __repr__(self) -> str:
        clsname = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return (
            f"<{clsname} "
            f"key={self.key!r} "
            f"href={self.href!r} "
            f"core_metadata={self.core_metadata!r} "
            f"require_python={self.require_python!r} "
            f"yanked={self.yanked!r}>"
        )


@total_ordering
class SimpleLinks:
    __slots__ = ("_links", "stale")
    _links: list[SimplelinkMeta]
    stale: bool

    def __init__(
        self,
        links: Sequence[JoinedLink] | SimpleLinks,
        *,
        core_metadata: bool = False,
        stale: bool = False,
    ) -> None:
        assert links is not None
        if isinstance(links, SimpleLinks):
            self._links = links._links
            self.stale = links.stale or stale
        else:
            self._links = [
                SimplelinkMeta(x, core_metadata=core_metadata) for x in links
            ]
            self.stale = stale

    def __hash__(self):
        return hash((self._links, self.stale))

    def __iter__(self):
        return self._links.__iter__()

    def __len__(self) -> int:
        return len(self._links)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            other = other._links
        return self._links == other

    def __lt__(self, other):
        if isinstance(other, type(self)):
            other = other._links
        return self._links < other

    def append(self, item):
        self._links.append(item)

    def sort(self, *args, **kw):
        self._links.sort(*args, **kw)

    def __repr__(self) -> str:
        clsname = f"{self.__class__.__module__}.{self.__class__.__name__}"
        content = ", ".join(repr(x) for x in self._links)
        return f"<{clsname} stale={self.stale!r} [{content}]>"
