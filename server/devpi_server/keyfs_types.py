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
from string import Formatter
from typing import TYPE_CHECKING
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
    from .readonly import DictViewReadonly
    from .readonly import ListViewReadonly
    from .readonly import Readonly
    from .readonly import SetViewReadonly
    from .readonly import TupleViewReadonly
    from collections.abc import Generator
    from collections.abc import Iterator
    from pathlib import Path
    from re import Match
    from re import Pattern
    from typing import Callable
    from typing import Union
    from typing_extensions import Self

    KeyFSTypesRO = Union[
        bool,
        bytes,
        DictViewReadonly[str, Readonly],
        float,
        frozenset,
        int,
        ListViewReadonly,
        SetViewReadonly,
        str,
        TupleViewReadonly,
    ]
    KeyFSTypes = Union[
        bool, bytes, dict[str, object], float, frozenset, int, list, set, str, tuple
    ]


ULID_NS_DIVISOR = 1_000_000_000
ULID_SHIFT = 28
# steal a few bits from the seconds part for more randomness to avoid collisions
ULID_RAND_BITS = 30


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
        high_part = (ns // ULID_NS_DIVISOR) << ULID_SHIFT
        low_part = _randbits(ULID_RAND_BITS)
        return super().__new__(cls, high_part ^ low_part)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


@frozen
class Record:
    key: LocatedKey
    ulid: ULID
    value: KeyFSTypes | Deleted
    back_serial: int
    old_ulid: ULID | Absent
    old_value: KeyFSTypesRO | Absent | Deleted

    def __attrs_post_init__(self):
        if (
            not isinstance(self.value, Deleted)
            and not isinstance(self.old_value, (Absent, Deleted))
            and not isinstance(ensure_deeply_readonly(self.value), type(self.old_value))
        ):
            msg = f"Mismatching types for value {self.value!r} and old_value {self.old_value!r}"
            raise TypeError(msg)


@frozen
class KeyData:
    relpath: str
    keyname: str
    ulid: ULID
    serial: int
    back_serial: int
    value: KeyFSTypesRO | Deleted

    @property
    def last_serial(self) -> int:
        return self.serial

    @property
    def mutable_value(self) -> KeyFSTypes | Deleted:
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
    relpath: str
    hash_digest: str | None


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


class IKeyFSKey(Interface):
    key_name = Attribute("""
        The name of the key type. """)


@implementer(IKeyFSKey)
class LocatedKey:
    __slots__ = ("key_name", "key_type", "keyfs", "location", "params")
    key_name: str
    keyfs: KeyFS
    key_type: type[KeyFSTypes]
    location: str
    params: dict[str, str]

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        location: str,
        key_type: type[KeyFSTypes],
        params: dict | None = None,
    ) -> None:
        self.keyfs = keyfs
        self.key_name = key_name
        assert "{" not in location
        self.location = location
        self.key_type = key_type
        self.params = {} if params is None else params

    def __hash__(self) -> int:
        return hash((self.key_name, self.relpath))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocatedKey):
            return NotImplemented
        return self.key_name == other.key_name and self.relpath == other.relpath

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.location!r}>"

    def delete(self) -> None:
        return self.keyfs.tx.delete(self)

    def exists(self) -> bool:
        return self.keyfs.tx.exists(self)

    def get(self) -> KeyFSTypesRO:
        return self.keyfs.tx.get(self)

    def get_mutable(self) -> KeyFSTypes:
        return self.keyfs.tx.get_mutable(self)

    def is_dirty(self) -> bool:
        return self.keyfs.tx.is_dirty(self)

    @property
    def back_serial(self) -> int | None:
        try:
            return self.keyfs.tx.back_serial(self)
        except KeyError:
            return None

    @property
    def last_serial(self) -> int | None:
        try:
            return self.keyfs.tx.last_serial(self)
        except KeyError:
            return None

    @property
    def ulid(self) -> ULID | Absent:
        try:
            return self.keyfs.tx.ulid(self)
        except KeyError:
            return absent

    @property
    def relpath(self) -> str:
        return self.location

    def set(self, val: KeyFSTypes) -> None:
        if not isinstance(val, self.key_type) and not issubclass(
            self.key_type, type(val)
        ):
            raise TypeError(
                "%r requires value of type %r, got %r"
                % (self.relpath, self.key_type.__name__, type(val).__name__)
            )
        self.keyfs.tx.set(self, val)

    @contextlib.contextmanager
    def update(self) -> Generator[KeyFSTypes, None, None]:
        val = self.keyfs.tx.get_mutable(self)
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)


@implementer(IKeyFSKey)
class PatternedKey:
    __slots__ = (
        "_rex_reverse",
        "key_name",
        "key_type",
        "keyfs",
        "parent_key",
        "pattern",
    )
    _rex_reverse: Pattern
    key_type: type[KeyFSTypes]

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        pattern: str,
        parent_key: PatternedKey | None,
        key_type: type[KeyFSTypes],
    ) -> None:
        self.keyfs = keyfs
        self.key_name = key_name
        self.pattern = pattern
        self.parent_key = parent_key
        self.key_type = key_type

    def __call__(self, **kw: str) -> LocatedKey:
        for val in kw.values():
            if "/" in val:
                raise ValueError(val)
        location = self.full_pattern.format_map(kw)
        return LocatedKey(self.keyfs, self.key_name, location, self.key_type, params=kw)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.pattern!r} {self.parent_key!r}>"

    def extract_params(self, relpath: str) -> dict:
        m = self.rex_reverse.match(relpath)
        return m.groupdict() if m is not None else {}

    def iter_patterns(self) -> Iterator[str]:
        lineage_patterns = (x.pattern for x in iter_lineage(self))
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
