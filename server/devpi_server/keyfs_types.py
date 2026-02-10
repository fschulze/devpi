from __future__ import annotations

from .markers import Absent
from .markers import Deleted
from .markers import absent
from .markers import deleted
from .readonly import ensure_deeply_readonly
from .readonly import get_mutable_deepcopy
from attrs import define
from attrs import field
from attrs import frozen
from collections.abc import Hashable
from string import Formatter
from typing import Generic
from typing import NewType
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import overload
import contextlib
import random
import re
import time
import weakref


if TYPE_CHECKING:
    from .interfaces import IStorage
    from .interfaces import IStorageConnection
    from .interfaces import IWriter
    from .keyfs import KeyFS
    from .normalized import NormalizedName
    from .readonly import DictViewReadonly
    from .readonly import ListViewReadonly
    from .readonly import SetViewReadonly
    from .readonly import TupleViewReadonly
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path
    from re import Match
    from re import Pattern
    from typing import Any
    from typing import Self

    KeyFSTypesRO = (
        bool
        | bytes
        | DictViewReadonly
        | float
        | frozenset
        | int
        | ListViewReadonly
        | SetViewReadonly
        | str
        | TupleViewReadonly
    )


ULID_NS_DIVISOR = 1_000_000_000
ULID_RAND_BITS = 28


class ULID(int):
    def __new__(cls, ulid: int) -> Self:
        assert ulid >= 0
        return super().__new__(cls, ulid)

    @classmethod
    def new(
        cls,
        *,
        _randbits: Callable = random.getrandbits,
        _time_ns: Callable = time.time_ns,
    ) -> Self:
        ns = _time_ns()
        ts_part = (ns // ULID_NS_DIVISOR) << ULID_RAND_BITS
        rand_part = _randbits(ULID_RAND_BITS)
        return super().__new__(cls, ts_part | rand_part)

    @property
    def ts_part(self) -> int:
        return self >> ULID_RAND_BITS

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


KeyFSTypes = bool | bytes | dict | float | frozenset | int | list | set | str | tuple
KeyType = TypeVar("KeyType")


@frozen
class Record(Generic[KeyType]):
    key: ULIDKey[KeyType]
    value: KeyFSTypes | Deleted
    back_serial: int
    old_key: ULIDKey[KeyType] | Absent
    old_value: KeyFSTypesRO | Absent | Deleted

    def __attrs_post_init__(self) -> None:
        if not isinstance(value := self.value, Deleted):
            if not isinstance(value, self.key.key_type):
                msg = f"Mismatching value type {type(value)} for record with {self.key}"
                raise TypeError(msg)
            if not isinstance(self.old_value, (Absent, Deleted)) and not isinstance(
                ensure_deeply_readonly(value), type(self.old_value)
            ):
                msg = f"Mismatching types for value {value!r} and old_value {self.old_value!r}"
                raise TypeError(msg)


RelPath = NewType("RelPath", str)


@frozen
class KeyData(Generic[KeyType]):
    key: ULIDKey[KeyType]
    serial: int
    back_serial: int
    value: KeyFSTypesRO | Deleted

    @property
    def last_serial(self) -> int:
        return self.serial

    @property
    def mutable_value(self) -> KeyType | Deleted:
        return deleted if (val := self.value) is deleted else get_mutable_deepcopy(val)


@frozen
class StorageInfo:
    name: str = field(kw_only=True)
    description: str = field(kw_only=True, default="")
    exists: Callable[[Path, dict], bool] = field(kw_only=True)
    hidden: bool = field(default=False, kw_only=True)
    storage_cls: type[IStorage] = field(kw_only=True)
    connection_cls: type[IStorageConnection] = field(kw_only=True)
    writer_cls: type[IWriter] = field(kw_only=True)
    storage_factory: Callable = field(kw_only=True)
    process_settings: Callable | None = field(kw_only=True)
    settings: dict = field(kw_only=True, default={})

    @property
    def storage_with_filesystem(self) -> bool:
        from .interfaces import IDBIOFileConnection

        return not IDBIOFileConnection.implementedBy(self.connection_cls)


@define
class FilePathInfo:
    relpath: RelPath
    hash_digest: str | None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FilePathInfo):
            return self.relpath == other.relpath
        return False

    def __hash__(self) -> int:
        return hash(self.relpath)


_formatter = Formatter()


def get_pattern_keys(pattern: str) -> frozenset[str]:
    return frozenset(x[1] for x in _formatter.parse(pattern) if x[1] is not None)


def get_params(pattern: str, kw: dict[str, str]) -> dict[str, str]:
    keys = get_pattern_keys(pattern)
    params = {k: kw[k] for k in keys}
    if frozenset(kw).difference(keys):
        msg = f"Not all parameters contained in pattern {pattern!r}: {kw!r}"
        raise ValueError(msg)
    return params


def get_rex_reverse(pattern: str) -> Pattern:
    def repl(match: Match) -> str:
        name = match.group(1)
        return rf"(?P<{name}>[^\/]+)"

    rex_pattern = re.sub(r"\{(.+?)\}", repl, pattern.replace("+", r"\+"))
    return re.compile(f"^{rex_pattern}$")


def iter_lineage(key: PatternedKey) -> Iterator[PatternedKey]:
    ancestor_key = key.parent_key
    while ancestor_key:
        yield ancestor_key
        ancestor_key = ancestor_key.parent_key


H = TypeVar("H", bound=Hashable)


class LocatedKey(Generic[KeyType]):
    __slots__ = ("_hash", "_keyfs", "key_name", "key_type", "location", "params")
    _hash: int
    _keyfs: weakref.ReferenceType[KeyFS]
    key_name: str
    key_type: type[KeyType]
    location: str
    params: dict[str, str]

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        location: str,
        key_type: type[KeyType],
        params: dict | None = None,
    ) -> None:
        self._keyfs = weakref.ref(keyfs)
        self.key_name = key_name
        assert "{" not in location
        self.location = location
        self.key_type = key_type
        self.params = {} if params is None else params

    def __hash__(self) -> int:
        _hash = getattr(self, "_hash", None)
        if _hash is None:
            _hash = self._hash = hash((self.key_name, self.relpath))
        return _hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocatedKey):
            return NotImplemented
        return self.key_name == other.key_name and self.relpath == other.relpath

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.location!r}>"

    def delete(self) -> None:
        return self.keyfs.tx.delete(self)

    def deleted(self) -> bool:
        return self.keyfs.tx.deleted(self)

    def exists(self) -> bool:
        return self.keyfs.tx.exists(self)

    @overload
    def get(self: LocatedKey[bool], default: Absent = absent) -> bool: ...

    @overload
    def get(self: LocatedKey[bytes], default: Absent = absent) -> bytes: ...

    @overload
    def get(self: LocatedKey[dict], default: Absent = absent) -> DictViewReadonly: ...

    @overload
    def get(self: LocatedKey[float], default: Absent = absent) -> float: ...

    @overload
    def get(self: LocatedKey[frozenset], default: Absent = absent) -> frozenset: ...

    @overload
    def get(self: LocatedKey[int], default: Absent = absent) -> int: ...

    @overload
    def get(self: LocatedKey[list], default: Absent = absent) -> ListViewReadonly: ...

    @overload
    def get(
        self: LocatedKey[set[H]], default: Absent = absent
    ) -> SetViewReadonly[H]: ...

    @overload
    def get(self: LocatedKey[str], default: Absent = absent) -> str: ...

    @overload
    def get(self: LocatedKey[tuple], default: Absent = absent) -> TupleViewReadonly: ...

    @overload
    def get(self, default: Absent = absent) -> KeyFSTypesRO: ...

    @overload
    def get(self, default: Any = absent) -> Any: ...

    def get(self, default: Any = absent) -> Any:
        return self.keyfs.tx.get(self, default=default)

    def get_mutable(self) -> KeyType:
        return self.keyfs.tx.get_mutable(self)

    def is_dirty(self) -> bool:
        return self.keyfs.tx.is_dirty(self)

    @property
    def back_serial(self) -> int:
        return self.keyfs.tx.back_serial(self)

    @property
    def last_serial(self) -> int:
        return self.keyfs.tx.last_serial(self)

    def make_ulid_key(self, ulid: ULID) -> ULIDKey[KeyType]:
        return ULIDKey(
            self.keyfs,
            self.key_name,
            self.location,
            ulid,
            self.key_type,
            params=self.params,
        )

    def new_ulidkey(self) -> ULIDKey[KeyType]:
        ulid = self.keyfs._new_ulid()
        return self.make_ulid_key(ulid)

    @property
    def ulid(self) -> ULID | Absent:
        try:
            return self.resolve(fetch=False).ulid
        except KeyError:
            return absent

    @property
    def relpath(self) -> RelPath:
        return RelPath(self.location)

    @property
    def keyfs(self) -> KeyFS:
        keyfs = self._keyfs()
        assert keyfs is not None
        return keyfs

    def resolve(self, *, fetch: bool) -> ULIDKey:
        return self.keyfs.tx.resolve(self, fetch=fetch)

    def set(self, val: KeyType) -> None:
        try:
            ulid_key = self.resolve(fetch=True)
        except KeyError:
            self.keyfs.tx.set(self, val)
        else:
            ulid_key.set(val)

    @contextlib.contextmanager
    def update(self) -> Iterator[KeyType]:
        val = self.get_mutable()
        assert isinstance(val, self.key_type)
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)


class PatternedKey(Generic[KeyType]):
    __slots__ = (
        "_keyfs",
        "_rex_reverse",
        "key_name",
        "key_type",
        "parent_key",
        "pattern",
    )
    _keyfs: weakref.ReferenceType[KeyFS]
    _rex_reverse: Pattern
    key_name: str
    key_type: type[KeyType]
    parent_key: PatternedKey | None
    pattern: str

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        pattern: str,
        parent_key: PatternedKey | None,
        key_type: type[KeyType],
    ) -> None:
        self._keyfs = weakref.ref(keyfs)
        self.key_name = key_name
        self.pattern = pattern
        self.parent_key = parent_key
        self.key_type = key_type

    def __call__(self, **kw: NormalizedName | str) -> LocatedKey[KeyType]:
        for val in kw.values():
            if "/" in val:
                raise ValueError(val)
        location = self.full_pattern.format_map(kw)
        keyfs = self._keyfs()
        assert keyfs is not None
        return LocatedKey(keyfs, self.key_name, location, self.key_type, params=kw)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.pattern!r} {self.parent_key!r}>"

    def extract_params(self, relpath: str) -> dict:
        m = self.rex_reverse.match(relpath)
        return m.groupdict() if m is not None else {}

    def iter_patterns(self) -> Iterator[str]:
        lineage_patterns: Iterator[str] = (x.pattern for x in iter_lineage(self))
        parts = (
            (self.pattern, *lineage_patterns)
            if self.pattern
            else tuple(lineage_patterns)
        )
        return reversed(parts)

    @property
    def full_pattern(self) -> str:
        return "/".join(self.iter_patterns())

    @property
    def rex_reverse(self) -> Pattern:
        _rex_reverse = getattr(self, "_rex_reverse", None)
        if _rex_reverse is None:
            _rex_reverse = self._rex_reverse = get_rex_reverse(self.full_pattern)
        return _rex_reverse


class ULIDKey(Generic[KeyType]):
    __slots__ = ("key_name", "key_type", "keyfs", "location", "params", "ulid")
    keyfs: KeyFS
    key_name: str
    key_type: type[KeyType]
    location: str
    params: dict[str, str]
    ulid: ULID

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        location: str,
        ulid: ULID,
        key_type: type[KeyType],
        params: dict | None = None,
    ) -> None:
        self.keyfs = keyfs
        self.key_name = key_name
        assert "{" not in location
        self.location = location
        self.key_type = key_type
        self.params = {} if params is None else params
        self.ulid = ulid

    def __hash__(self) -> int:
        return hash(self.ulid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ULIDKey):
            return NotImplemented
        return self.ulid == other.ulid

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.location!r} {self.ulid!r}>"

    def delete(self) -> None:
        return self.keyfs.tx.delete(self)

    def exists(self) -> bool:
        return self.keyfs.tx.exists(self)

    @overload
    def get(self: ULIDKey[bool], default: Absent = absent) -> bool: ...

    @overload
    def get(self: ULIDKey[bytes], default: Absent = absent) -> bytes: ...

    @overload
    def get(self: ULIDKey[dict], default: Absent = absent) -> DictViewReadonly: ...

    @overload
    def get(self: ULIDKey[float], default: Absent = absent) -> float: ...

    @overload
    def get(self: ULIDKey[frozenset], default: Absent = absent) -> frozenset: ...

    @overload
    def get(self: ULIDKey[int], default: Absent = absent) -> int: ...

    @overload
    def get(self: ULIDKey[list], default: Absent = absent) -> ListViewReadonly: ...

    @overload
    def get(self: ULIDKey[set], default: Absent = absent) -> SetViewReadonly: ...

    @overload
    def get(self: ULIDKey[str], default: Absent = absent) -> str: ...

    @overload
    def get(self: ULIDKey[tuple], default: Absent = absent) -> TupleViewReadonly: ...

    @overload
    def get(self, default: Absent = absent) -> KeyFSTypesRO: ...

    @overload
    def get(self, default: Any = absent) -> Any: ...

    def get(self, default: Any = absent) -> Any:
        return self.keyfs.tx.get(self, default=default)

    def get_mutable(self) -> KeyType:
        return self.keyfs.tx.get_mutable(self)

    def is_dirty(self) -> bool:
        return self.keyfs.tx.is_dirty(self)

    @property
    def back_serial(self) -> int:
        return self.keyfs.tx.back_serial(self)

    @property
    def last_serial(self) -> int:
        return self.keyfs.tx.last_serial(self)

    def new_ulidkey(self) -> ULIDKey[KeyType]:
        ulid = self.keyfs._new_ulid()
        return ULIDKey(
            self.keyfs,
            self.key_name,
            self.location,
            ulid,
            self.key_type,
            params=self.params,
        )

    @property
    def relpath(self) -> RelPath:
        return RelPath(self.location)

    def set(self, val: KeyType) -> None:
        self.keyfs.tx.set(self, val)

    @contextlib.contextmanager
    def update(self) -> Iterator[KeyType]:
        val = self.get_mutable()
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)
