from __future__ import annotations

from .markers import Absent
from .markers import Deleted
from .readonly import ensure_deeply_readonly
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
    from .readonly import SetViewReadonly
    from .readonly import TupleViewReadonly
    from pathlib import Path
    from typing import Callable
    from typing import Union
    from typing_extensions import Self

    KeyFSTypesRO = Union[
        bool,
        bytes,
        DictViewReadonly,
        float,
        frozenset,
        int,
        ListViewReadonly,
        SetViewReadonly,
        str,
        TupleViewReadonly,
    ]
    KeyFSTypes = Union[bool, bytes, dict, float, frozenset, int, list, set, str, tuple]


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
    value: KeyFSTypes | None
    back_serial: int
    old_value: KeyFSTypesRO | Absent

    def __attrs_post_init__(self):
        if (
            self.value is not None
            and not isinstance(self.old_value, (Absent, Deleted, type(None)))
            and not isinstance(ensure_deeply_readonly(self.value), type(self.old_value))
        ):
            msg = f"Mismatching types for value {self.value!r} and old_value {self.old_value!r}"
            raise TypeError(msg)


@frozen
class KeyData:
    relpath: str
    keyname: str
    serial: int
    back_serial: int
    value: KeyFSTypesRO | None

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
    relpath: str
    hash_digest: str | None


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


@implementer(IKeyFSKey)
class LocatedKey:
    __slots__ = ("key_name", "keyfs", "location", "name", "params", "type")

    def __init__(self, keyfs, key_name, location, name, key_type, params=None):
        self.keyfs = keyfs
        self.key_name = key_name
        assert "{" not in location
        assert "{" not in name
        self.location = location
        self.name = name
        self.type = key_type
        self.params = {} if params is None else params

    def __hash__(self):
        return hash((self.key_name, self.relpath))

    def __eq__(self, other):
        return self.key_name == other.key_name and self.relpath == other.relpath

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.key_name} {self.type.__name__} {self.location!r} {self.name!r}>"

    def delete(self):
        return self.keyfs.tx.delete(self)

    def exists(self):
        return self.keyfs.tx.exists(self)

    def get(self):
        return self.keyfs.tx.get(self)

    def get_mutable(self):
        return self.keyfs.tx.get_mutable(self)

    def is_dirty(self):
        return self.keyfs.tx.is_dirty(self)

    @property
    def last_serial(self):
        try:
            return self.keyfs.tx.last_serial(self)
        except KeyError:
            return None

    @property
    def relpath(self):
        location = self.location
        name = self.name
        if location and name:
            return f"{location}/{name}"
        if location:
            return location
        return name

    def set(self, val):
        if not isinstance(val, self.type) and not issubclass(self.type, type(val)):
            raise TypeError(
                "%r requires value of type %r, got %r"
                % (self.relpath, self.type.__name__, type(val).__name__)
            )
        self.keyfs.tx.set(self, val)

    @contextlib.contextmanager
    def update(self):
        val = self.keyfs.tx.get_mutable(self)
        yield val
        # no exception, so we can set and thus mark dirty the object
        self.set(val)


@implementer(IKeyFSKey)
class NamedKey:
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
        key_type: type[KeyFSTypes],
        params: dict | None = None,
    ):
        if parent_key is None:
            raise ValueError
        self.keyfs = keyfs
        self.key_name = key_name
        self.name = name
        self.parent_key = parent_key
        self.type = key_type
        self.params = {} if params is None else params
        self._rex_reverse = None

    def __call__(self, **kw):
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
class NamedKeyFactory:
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
        key_type: type[KeyFSTypes],
    ):
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
        self.type = key_type
        self._rex_reverse = None

    def __call__(self, **kw):
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
