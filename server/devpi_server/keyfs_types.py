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
    from typing import Any
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
    key: ULIDKey
    value: KeyFSTypes | Deleted
    back_serial: int
    old_key: ULIDKey | Absent
    old_value: KeyFSTypesRO | Absent | Deleted

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.key, ULIDKey)
        assert isinstance(self.old_key, (ULIDKey, Absent))
        if (
            not isinstance(self.value, Deleted)
            and not isinstance(self.old_value, (Absent, Deleted))
            and not isinstance(ensure_deeply_readonly(self.value), type(self.old_value))
        ):
            msg = f"Mismatching types for value {self.value!r} and old_value {self.old_value!r}"
            raise TypeError(msg)


@frozen
class KeyData:
    key: ULIDKey
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


def get_simple_pattern_key(pattern: str) -> str | None:
    result = list(_formatter.parse(pattern))
    if len(result) != 1:
        return None
    ((literal_text, field_name, format_spec, conversion),) = result
    if literal_text == "" and format_spec == "" and conversion is None:
        return field_name
    return None


@overload
def iter_lineage(key: LocatedKey) -> Iterator[LocatedKey]:
    pass


@overload
def iter_lineage(key: PatternedKey) -> Iterator[PatternedKey]:
    pass


@overload
def iter_lineage(key: ULIDKey) -> Iterator[ULIDKey]:
    pass


def iter_lineage(
    key: LocatedKey | PatternedKey | ULIDKey,
) -> Iterator[LocatedKey | PatternedKey | ULIDKey]:
    ancestor_key = key.parent_key
    while ancestor_key:
        yield ancestor_key
        ancestor_key = ancestor_key.parent_key


class LocatedKey:
    __slots__ = (
        "_hash",
        "_keyfs",
        "key_name",
        "key_type",
        "name",
        "params",
        "parent_key",
    )
    _hash: int
    _keyfs: weakref.ReferenceType[KeyFS]
    key_name: str
    key_type: type[KeyFSTypes]
    name: str
    parent_key: LocatedKey | ULIDKey | None
    params: dict[str, str]

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        name: str,
        parent_key: LocatedKey | ULIDKey | None,
        key_type: type[KeyFSTypes],
        params: dict | None = None,
    ) -> None:
        self._keyfs = weakref.ref(keyfs)
        self.key_name = key_name
        assert "{" not in name
        self.name = name
        self.parent_key = parent_key
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
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.name!r} {self.parent_key!r}>"

    def delete(self) -> None:
        return self.keyfs.tx.delete(self)

    def exists(self) -> bool:
        return self.keyfs.tx.exists(self)

    @overload
    def get(self, default: Absent) -> KeyFSTypesRO: ...

    @overload
    def get(self, default: Any) -> Any: ...

    def get(self, default: Any = absent) -> Any:
        return self.keyfs.tx.get(self, default=default)

    def get_mutable(self) -> KeyFSTypes:
        return self.keyfs.tx.get_mutable(self)

    def is_dirty(self) -> bool:
        return self.keyfs.tx.is_dirty(self)

    @property
    def back_serial(self) -> int:
        return self.keyfs.tx.back_serial(self)

    @property
    def last_serial(self) -> int:
        return self.keyfs.tx.last_serial(self)

    def make_ulid_key(
        self,
        ulid: ULID,
        *,
        parent_key: ULIDKey | Absent | None = absent,
        parent_ulid: ULID | Absent | None = absent,
    ) -> ULIDKey:
        if not isinstance(parent_key, Absent):
            assert isinstance(parent_ulid, Absent)
            parent_ulid = None if parent_key is None else parent_key.ulid
        if isinstance(parent_ulid, Absent):
            assert isinstance(parent_key, Absent)
            parent_ulid = self.parent_ulid
        ulid_key = ULIDKey(
            self.keyfs,
            self.key_name,
            self.location,
            self.name,
            ulid,
            parent_ulid,
            self.key_type,
            params=self.params,
        )
        if isinstance(parent_key, ULIDKey):
            ulid_key._parent_key = parent_key
        return ulid_key

    def new_ulid(self) -> ULIDKey:
        return self.make_ulid_key(ULID.new())

    @property
    def ulid(self) -> ULID | Absent:
        try:
            return self.resolve(fetch=False).ulid
        except KeyError:
            return absent

    @property
    def parent_ulid(self) -> ULID | None:
        parent_key = self.parent_key
        if parent_key is None:
            return None
        if isinstance(parent_key, ULIDKey):
            return parent_key.ulid
        return parent_key.resolve(fetch=False).ulid

    @property
    def query_parent_ulid(self) -> int:
        parent_ulid = self.parent_ulid
        return -1 if parent_ulid is None else int(parent_ulid)

    @property
    def location(self) -> str:
        if self.parent_key is None:
            return ""
        return self.parent_key.relpath

    @property
    def relpath(self) -> str:
        location = self.location
        name = self.name
        if location and name:
            return f"{location}/{name}"
        if location:
            return location
        return name

    @property
    def keyfs(self) -> KeyFS:
        keyfs = self._keyfs()
        assert keyfs is not None
        return keyfs

    def resolve(self, *, fetch: bool) -> ULIDKey:
        return self.keyfs.tx.resolve(self, fetch=fetch)

    def set(self, val: KeyFSTypes) -> None:
        try:
            ulid_key = self.resolve(fetch=True)
        except KeyError:
            self.keyfs.tx.set(self, val)
        else:
            ulid_key.set(val)

    @contextlib.contextmanager
    def update(self) -> Generator[KeyFSTypes, None, None]:
        val = self.get_mutable()
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)


class PatternedKey:
    __slots__ = (
        "_full_rex_reverse",
        "_hash",
        "_keyfs",
        "_keys",
        "_rex_reverse",
        "_simple_pattern_key",
        "key_name",
        "key_type",
        "parent_key",
        "pattern",
    )
    _full_rex_reverse: Pattern
    _hash: int
    _keyfs: weakref.ReferenceType[KeyFS]
    _keys: frozenset[str]
    _rex_reverse: Pattern
    _simple_pattern_key: str | None
    key_name: str
    key_type: type[KeyFSTypes]
    parent_key: PatternedKey | None
    pattern: str

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        pattern: str,
        parent_key: PatternedKey | None,
        key_type: type[KeyFSTypes],
    ) -> None:
        self._keyfs = weakref.ref(keyfs)
        self.key_name = key_name
        self.pattern = pattern
        self.parent_key = parent_key
        self.key_type = key_type

    def __call__(
        self, *args: str, **kw: LocatedKey | ULIDKey | str
    ) -> LocatedKey | SearchKey:
        given_parent_key = kw.pop("parent_key", absent)
        assert isinstance(given_parent_key, (LocatedKey, ULIDKey, Absent))
        for val in kw.values():
            assert isinstance(val, str)
            if "/" in val:
                raise ValueError(val)
        if not isinstance(given_parent_key, Absent):
            kw.update(given_parent_key.params)
        parent_key: LocatedKey | SearchKey | ULIDKey | None = (
            None if self.parent_key is None else self.parent_key(**kw)
        )
        assert parent_key is None or isinstance(parent_key, LocatedKey)
        if not isinstance(given_parent_key, Absent):
            assert parent_key is not None
            assert given_parent_key.key_name == parent_key.key_name
            assert given_parent_key.relpath == parent_key.relpath
            parent_key = given_parent_key
        keyfs = self._keyfs()
        assert keyfs is not None
        if len(missing := self.keys.difference(kw)) == 1 and len(args) == 1:
            (k,) = missing
            (val,) = args
            kw[k] = val
        elif args:
            msg = f"{self.__class__.__name__}.__call__() takes at most 1 positional argument {len(args)} were given"
            raise TypeError(msg)
        if self.keys.difference(kw):
            return SearchKey(
                keyfs, self.key_name, self.pattern, parent_key, self.key_type, params=kw
            )
        if self.simple_pattern_key is not None:
            name = kw[self.simple_pattern_key]
            assert isinstance(name, str)
        else:
            name = self.pattern.format_map(kw)
        return LocatedKey(
            keyfs, self.key_name, name, parent_key, self.key_type, params=kw
        )

    def __hash__(self) -> int:
        _hash = getattr(self, "_hash", None)
        if _hash is None:
            _hash = self._hash = hash((self.key_name, self.full_pattern))
        return _hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PatternedKey):
            return NotImplemented
        return (
            self.key_name == other.key_name and self.full_pattern == other.full_pattern
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.pattern!r} {self.parent_key!r}>"

    def extract_params(self, relpath: str) -> dict:
        m = self.full_rex_reverse.match(relpath)
        return m.groupdict() if m is not None else {}

    @property
    def keys(self) -> frozenset[str]:
        _keys = getattr(self, "_keys", None)
        if _keys is None:
            _keys = self._keys = get_pattern_keys(self.pattern)
        return _keys

    def make_ulid_key(
        self,
        location: str,
        name: str,
        ulid: ULID,
        parent_key: ULIDKey | Absent | None = absent,
        parent_ulid: ULID | Absent | None = absent,
        parent_params: dict | None = None,
    ) -> ULIDKey:
        if not isinstance(parent_key, Absent):
            assert isinstance(parent_ulid, Absent)
            parent_ulid = None if parent_key is None else parent_key.ulid
        else:
            assert not isinstance(parent_ulid, Absent)
        if parent_ulid is None and self.simple_pattern_key is not None:
            params = {self.simple_pattern_key: name}
        elif parent_params is not None and self.simple_pattern_key is not None:
            params = parent_params | {self.simple_pattern_key: name}
        else:
            if location and name:
                relpath = f"{location}/{name}"
            elif location:
                relpath = location
            else:
                relpath = name
            m = self.full_rex_reverse.match(relpath)
            params = m.groupdict() if m is not None else {}
        keyfs = self._keyfs()
        assert keyfs is not None
        ulid_key = ULIDKey(
            keyfs,
            self.key_name,
            location,
            name,
            ulid,
            parent_ulid,
            self.key_type,
            params=params,
        )
        if isinstance(parent_key, ULIDKey):
            ulid_key._parent_key = parent_key
        return ulid_key

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
    def full_rex_reverse(self) -> Pattern:
        _full_rex_reverse = getattr(self, "_full_rex_reverse", None)
        if _full_rex_reverse is None:
            _full_rex_reverse = self._full_rex_reverse = get_rex_reverse(
                self.full_pattern
            )
        return _full_rex_reverse

    @property
    def rex_reverse(self) -> Pattern:
        _rex_reverse = getattr(self, "_rex_reverse", None)
        if _rex_reverse is None:
            _rex_reverse = self._rex_reverse = get_rex_reverse(self.pattern)
        return _rex_reverse

    @property
    def simple_pattern_key(self) -> str | None:
        _simple_pattern_key = getattr(self, "_simple_pattern_key", absent)
        if isinstance(_simple_pattern_key, Absent):
            _simple_pattern_key = self._simple_pattern_key = get_simple_pattern_key(
                self.pattern
            )
        return _simple_pattern_key


class SearchKey:
    __slots__ = (
        "_hash",
        "_keyfs",
        "_keys",
        "_parent_ulidkey",
        "_simple_pattern_key",
        "key_name",
        "key_type",
        "params",
        "parent_key",
        "pattern",
    )
    _hash: int
    _keyfs: weakref.ReferenceType[KeyFS]
    _keys: frozenset[str]
    _parent_ulidkey: ULIDKey | None
    _simple_pattern_key: str | None
    key_name: str
    key_type: type[KeyFSTypes]
    parent_key: LocatedKey | ULIDKey | None
    pattern: str
    params: dict[str, str]

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        pattern: str,
        parent_key: LocatedKey | ULIDKey | None,
        key_type: type[KeyFSTypes],
        params: dict,
    ) -> None:
        self._keyfs = weakref.ref(keyfs)
        self.key_name = key_name
        self.pattern = pattern
        self.parent_key = parent_key
        self.key_type = key_type
        self.params = params

    def __call__(self, *args: str, **kw: LocatedKey | ULIDKey | str) -> LocatedKey:
        given_parent_key = kw.pop("parent_key", absent)
        assert isinstance(given_parent_key, (LocatedKey, ULIDKey, Absent))
        for val in kw.values():
            assert isinstance(val, str)
            if "/" in val:
                raise ValueError(val)
        if not isinstance(given_parent_key, Absent):
            kw.update(given_parent_key.params)
        if self.simple_pattern_key and len(args) == 1:
            name = args[0]
        else:
            missing = self.keys.difference(kw)
            if missing:
                (k,) = missing
                (val,) = args
                kw[k] = val
            elif args:
                msg = f"{self.__class__.__name__}.__call__() takes at most 1 positional argument {len(args)} were given"
                raise TypeError(msg)
            if set(kw).intersection(self.params):
                msg = f"{kw!r} overlaps {self.params!r}"
                raise TypeError(msg)
            if self.simple_pattern_key:
                _name = kw[self.simple_pattern_key]
                assert isinstance(_name, str)
                name = _name
            else:
                name = self.pattern.format_map(kw)
        keyfs = self._keyfs()
        assert keyfs is not None
        return LocatedKey(
            keyfs,
            self.key_name,
            name,
            self.parent_ulidkey,
            self.key_type,
            params=self.params | kw,
        )

    def __hash__(self) -> int:
        _hash = getattr(self, "_hash", None)
        if _hash is None:
            _hash = self._hash = hash((self.key_name, self.location))
        return _hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocatedKey):
            return NotImplemented
        return self.key_name == other.key_name and self.location == other.location

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.parent_key!r}>"

    def iter_ulidkey_values(
        self, *, fill_cache: bool = True
    ) -> Iterator[tuple[ULIDKey, KeyFSTypesRO]]:
        keyfs = self._keyfs()
        assert keyfs is not None
        yield from keyfs.tx.iter_ulidkey_values_for((self,), fill_cache=fill_cache)

    def iter_ulidkeys(self, *, fill_cache: bool = True) -> Iterator[ULIDKey]:
        keyfs = self._keyfs()
        assert keyfs is not None
        yield from keyfs.tx.iter_ulidkeys_for((self,), fill_cache=fill_cache)

    @property
    def keys(self) -> frozenset[str]:
        _keys = getattr(self, "_keys", None)
        if _keys is None:
            _keys = self._keys = get_pattern_keys(self.pattern)
        return _keys

    @property
    def location(self) -> str:
        if self.parent_key is None:
            return ""
        return self.parent_key.relpath

    @property
    def parent_ulid(self) -> ULID | None:
        parent_ulidkey = self.parent_ulidkey
        if parent_ulidkey is None:
            return None
        return parent_ulidkey.ulid

    @property
    def parent_ulidkey(self) -> ULIDKey | None:
        _parent_ulidkey = getattr(self, "_parent_ulidkey", absent)
        if isinstance(_parent_ulidkey, Absent):
            parent_key = self.parent_key
            if isinstance(parent_key, ULIDKey):
                _parent_ulidkey = self._parent_ulidkey = parent_key
            else:
                _parent_ulidkey = self._parent_ulidkey = (
                    None if parent_key is None else parent_key.resolve(fetch=False)
                )
        return _parent_ulidkey

    @property
    def query_parent_ulid(self) -> int:
        parent_ulidkey = self.parent_ulidkey
        return -1 if parent_ulidkey is None else int(parent_ulidkey.ulid)

    @property
    def simple_pattern_key(self) -> str | None:
        _simple_pattern_key = getattr(self, "_simple_pattern_key", absent)
        if isinstance(_simple_pattern_key, Absent):
            _simple_pattern_key = self._simple_pattern_key = get_simple_pattern_key(
                self.pattern
            )
        return _simple_pattern_key


class ULIDKey:
    __slots__ = (
        "_parent_key",
        "key_name",
        "key_type",
        "keyfs",
        "location",
        "name",
        "params",
        "parent_ulid",
        "ulid",
    )
    _parent_key: ULIDKey
    keyfs: KeyFS
    key_name: str
    key_type: type[KeyFSTypes]
    location: str
    name: str
    params: dict[str, str]
    parent_ulid: ULID | None
    ulid: ULID

    def __init__(
        self,
        keyfs: KeyFS,
        key_name: str,
        location: str,
        name: str,
        ulid: ULID,
        parent_ulid: ULID | None,
        key_type: type[KeyFSTypes],
        params: dict | None = None,
    ) -> None:
        self.keyfs = keyfs
        self.key_name = key_name
        assert "{" not in location
        assert "{" not in name
        self.location = location
        self.name = name
        self.key_type = key_type
        self.params = {} if params is None else params
        self.ulid = ulid
        self.parent_ulid = parent_ulid

    def __hash__(self) -> int:
        return hash(self.ulid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ULIDKey):
            return NotImplemented
        return self.ulid == other.ulid

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.key_name} {self.key_type.__name__} {self.location!r} {self.parent_ulid!r} {self.name!r} {self.ulid!r}>"

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
    def back_serial(self) -> int:
        return self.keyfs.tx.back_serial(self)

    @property
    def last_serial(self) -> int:
        return self.keyfs.tx.last_serial(self)

    def new_ulid(self) -> ULIDKey:
        return ULIDKey(
            self.keyfs,
            self.key_name,
            self.location,
            self.name,
            ULID.new(),
            self.parent_ulid,
            self.key_type,
            params=self.params,
        )

    @property
    def parent_key(self) -> ULIDKey | None:
        if self.parent_ulid is None:
            return None
        _parent_key = getattr(self, "_parent_key", None)
        if _parent_key is None:
            _parent_key = self._parent_key = self.keyfs.tx.key_for_ulid(
                self.parent_ulid
            )
        return _parent_key

    @property
    def query_parent_ulid(self) -> int:
        parent_ulid = self.parent_ulid
        return -1 if parent_ulid is None else int(parent_ulid)

    @property
    def relpath(self) -> str:
        location = self.location
        name = self.name
        if location and name:
            return f"{location}/{name}"
        if location:
            return location
        return name

    def set(self, val: KeyFSTypes) -> None:
        self.keyfs.tx.set(self, val)

    @contextlib.contextmanager
    def update(self) -> Generator[KeyFSTypes, None, None]:
        val = self.get_mutable()
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)
