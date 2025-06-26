from __future__ import annotations

from .markers import Absent
from .markers import Deleted
from .readonly import ensure_deeply_readonly
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
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface import implementer
import contextlib
import random
import re
import time


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
    key: LocatedKey[KeyType]
    value: KeyFSTypes | Deleted
    back_serial: int
    old_value: KeyFSTypesRO | Absent | Deleted

    def __attrs_post_init__(self) -> None:
        if not isinstance(value := self.value, Deleted):
            if not isinstance(value, self.key.type):
                msg = f"Mismatching value type {type(value)} for record with {self.key}"
                raise TypeError(msg)
            if not isinstance(self.old_value, (Absent, Deleted)) and not isinstance(
                ensure_deeply_readonly(value), type(self.old_value)
            ):
                msg = f"Mismatching types for value {value!r} and old_value {self.old_value!r}"
                raise TypeError(msg)


RelPath = NewType("RelPath", str)


@frozen
class KeyData:
    relpath: RelPath
    keyname: str
    serial: int
    back_serial: int
    value: KeyFSTypesRO | Deleted

    @property
    def last_serial(self):
        return self.serial


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
    settings: dict = field(kw_only=True, default={})

    @property
    def storage_with_filesystem(self):
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


def extract_params(key, relpath):
    m = key.rex_reverse.match(relpath)
    return m.groupdict() if m is not None else {}


def get_params(pattern, kw):
    keys = frozenset(x[1] for x in _formatter.parse(pattern) if x[1] is not None)
    params = {k: kw[k] for k in keys}
    if frozenset(kw).difference(keys):
        msg = f"Not all parameters contained in pattern {pattern!r}: {kw!r}"
        raise ValueError(msg)
    return params


def get_rex_reverse(pattern):
    def repl(match):
        name = match.group(1)
        return rf"(?P<{name}>[^\/]+)"

    rex_pattern = re.sub(r"\{(.+?)\}", repl, pattern.replace("+", r"\+"))
    return re.compile(f"^{rex_pattern}$")


def iter_lineage(key):
    ancestor_key = key.parent_key
    while ancestor_key:
        yield ancestor_key
        ancestor_key = ancestor_key.parent_key


class IKeyFSKey(Interface):
    key_name = Attribute("""
        The name of the key type. """)


H = TypeVar("H", bound=Hashable)


@implementer(IKeyFSKey)
class LocatedKey(Generic[KeyType]):
    __slots__ = ("key_name", "keyfs", "location", "name", "params", "type")
    keyfs: KeyFS
    key_name: str
    location: str
    name: str
    params: dict[str, str]

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        location: str,
        name: str,
        key_type: type[KeyType],
        params: dict | None = None,
    ) -> None:
        self.keyfs = keyfs
        self.key_name = key_name
        assert "{" not in location
        assert "{" not in name
        self.location = location
        self.name = name
        self.type: type[KeyType] = key_type
        self.params = {} if params is None else params

    def __hash__(self):
        return hash((self.key_name, self.relpath))

    def __eq__(self, other):
        return self.key_name == other.key_name and self.relpath == other.relpath

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.key_name} {self.type.__name__} {self.location!r} {self.name!r}>"

    def delete(self):
        return self.keyfs.tx.delete(self)

    def deleted(self):
        return self.keyfs.tx.deleted(self)

    def exists(self):
        return self.keyfs.tx.exists(self)

    @overload
    def get(self: LocatedKey[bool]) -> bool: ...

    @overload
    def get(self: LocatedKey[bytes]) -> bytes: ...

    @overload
    def get(self: LocatedKey[dict]) -> DictViewReadonly: ...

    @overload
    def get(self: LocatedKey[float]) -> float: ...

    @overload
    def get(self: LocatedKey[frozenset]) -> frozenset: ...

    @overload
    def get(self: LocatedKey[int]) -> int: ...

    @overload
    def get(self: LocatedKey[list]) -> ListViewReadonly: ...

    @overload
    def get(self: LocatedKey[set[H]]) -> SetViewReadonly[H]: ...

    @overload
    def get(self: LocatedKey[str]) -> str: ...

    @overload
    def get(self: LocatedKey[tuple]) -> TupleViewReadonly: ...

    def get(self) -> KeyFSTypesRO:
        return self.keyfs.tx.get(self)

    def get_mutable(self) -> KeyType:
        return self.keyfs.tx.get_mutable(self)

    def is_dirty(self) -> bool:
        return self.keyfs.tx.is_dirty(self)

    @property
    def last_serial(self) -> int | None:
        try:
            return self.keyfs.tx.last_serial(self)
        except KeyError:
            return None

    @property
    def relpath(self) -> RelPath:
        location = self.location
        name = self.name
        if location and name:
            return RelPath(f"{location}/{name}")
        if location:
            return RelPath(location)
        return RelPath(name)

    def set(self, val: KeyType) -> None:
        if not isinstance(val, self.type) and not issubclass(self.type, type(val)):
            raise TypeError(
                "%r requires value of type %r, got %r"
                % (self.relpath, self.type.__name__, type(val).__name__)
            )
        self.keyfs.tx.set(self, val)

    @contextlib.contextmanager
    def update(self) -> Iterator[KeyType]:
        val = self.keyfs.tx.get_mutable(self)
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)


@implementer(IKeyFSKey)
class NamedKey(Generic[KeyType]):
    __slots__ = (
        "_rex_reverse",
        "key_name",
        "keyfs",
        "name",
        "params",
        "parent_key",
        "type",
    )

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        name: str,
        parent_key: NamedKey | NamedKeyFactory,
        key_type: type[KeyType],
        params: dict | None = None,
    ) -> None:
        if parent_key is None:
            raise ValueError
        self.keyfs = keyfs
        self.key_name = key_name
        self.name = name
        self.parent_key = parent_key
        self.type: type[KeyType] = key_type
        self.params = {} if params is None else params
        self._rex_reverse = None

    def __call__(self, **kw: NormalizedName | str) -> LocatedKey[KeyType]:
        for val in kw.values():
            if "/" in val:
                raise ValueError(val)
        pattern = "/".join(reversed(tuple(x.pattern for x in iter_lineage(self))))
        params = get_params(pattern, kw)
        location = pattern.format_map(params)
        return LocatedKey(
            self.keyfs, self.key_name, location, self.name, self.type, params=params
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.key_name} {self.type.__name__} {self.name!r} {self.parent_key!r}>"

    def extract_params(self, relpath):
        return extract_params(self, relpath)

    @property
    def rex_reverse(self):
        if self._rex_reverse is None:
            parts = [self.name] if self.name else []
            parts.extend(x.pattern for x in iter_lineage(self))
            self._rex_reverse = get_rex_reverse("/".join(reversed(parts)))
        return self._rex_reverse


@implementer(IKeyFSKey)
class NamedKeyFactory(Generic[KeyType]):
    __slots__ = (
        "_rex_reverse",
        "key_name",
        "keyfs",
        "name",
        "parent_key",
        "pattern",
        "type",
    )

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        pattern: str,
        parent_key: NamedKey | NamedKeyFactory | None,
        key_type: type[KeyType],
    ) -> None:
        self.keyfs = keyfs
        self.key_name = key_name
        name_parts = []
        pattern_parts = []
        parts_iter = reversed(pattern.split("/"))
        for part in parts_iter:
            if "{" in part:
                pattern_parts.append(part)
                break
            name_parts.append(part)
        pattern_parts.extend(parts_iter)
        self.name = "/".join(reversed(name_parts))
        assert "{" not in self.name
        self.pattern = "/".join(reversed(pattern_parts))
        assert "{" in self.pattern
        self.parent_key = parent_key
        self.type: type[KeyType] = key_type
        self._rex_reverse = None

    def __call__(self, **kw: NormalizedName | str) -> LocatedKey[KeyType]:
        for val in kw.values():
            if "/" in val:
                raise ValueError(val)
        location = self.pattern.format_map(kw)
        if self.parent_key is None:
            return LocatedKey(
                self.keyfs, self.key_name, location, self.name, self.type, params=kw
            )
        parent_key = self.parent_key(**kw)
        location = f"{parent_key.location}/{location}"
        return LocatedKey(
            self.keyfs, self.key_name, location, self.name, self.type, params=kw
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.key_name} {self.type.__name__} {self.pattern!r} {self.name!r} {self.parent_key!r}>"

    def extract_params(self, relpath):
        return extract_params(self, relpath)

    @property
    def rex_reverse(self):
        if self._rex_reverse is None:
            parts = [self.name, self.pattern] if self.name else [self.pattern]
            parts.extend(x.pattern for x in iter_lineage(self))
            self._rex_reverse = get_rex_reverse("/".join(reversed(parts)))
        return self._rex_reverse
