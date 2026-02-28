from __future__ import annotations

from .keyfs_types import KeyType
from .keyfs_types import KeyTypeRO
from .keyfs_types import LocatedKey
from .keyfs_types import PatternedKey
from inspect import getmembers
from typing import Generic
from typing import TYPE_CHECKING
from typing import overload


if TYPE_CHECKING:
    from .keyfs import KeyFS
    from collections.abc import Iterator


class KeyFSSchemaMeta(type):
    @overload
    @classmethod
    def decl_anonymous_key(
        cls,
        key_name: str,
        parent_key: None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> LocatedKeyDescriptor[KeyType, KeyTypeRO]: ...

    @overload
    @classmethod
    def decl_anonymous_key(
        cls,
        key_name: str,
        parent_key: PatternedKeyDescriptor,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> PatternedKeyDescriptor[KeyType, KeyTypeRO]: ...

    @classmethod
    def decl_anonymous_key(
        cls,
        key_name: str,
        parent_key: PatternedKeyDescriptor | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> (
        LocatedKeyDescriptor[KeyType, KeyTypeRO]
        | PatternedKeyDescriptor[KeyType, KeyTypeRO]
    ):
        if parent_key is None:
            return LocatedKeyDescriptor(key_name, "", "", key_type, key_rotype)
        return PatternedKeyDescriptor(key_name, "", parent_key, key_type, key_rotype)

    @classmethod
    def decl_located_key(
        cls,
        key_name: str,
        location: str,
        name: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> LocatedKeyDescriptor[KeyType, KeyTypeRO]:
        return LocatedKeyDescriptor(key_name, location, name, key_type, key_rotype)

    @classmethod
    def decl_patterned_key(
        cls,
        key_name: str,
        pattern_or_name: str,
        parent_key: PatternedKeyDescriptor | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> PatternedKeyDescriptor[KeyType, KeyTypeRO]:
        return PatternedKeyDescriptor(
            key_name, pattern_or_name, parent_key, key_type, key_rotype
        )


def validated_key_name(key_name: str) -> str:
    allowed_chars = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    assert not set(key_name).difference(allowed_chars), f"Invalid key name: {key_name}"
    return key_name


class LocatedKeyDescriptor(Generic[KeyType, KeyTypeRO]):
    __slots__ = ("key_name", "key_rotype", "key_type", "location", "name")

    def __init__(
        self,
        key_name: str,
        location: str,
        name: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> None:
        assert "{" not in name, name
        assert "{" not in location, location
        self.key_name = validated_key_name(key_name)
        self.location = location
        self.name = name
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema] | None = None
    ) -> LocatedKey[KeyType, KeyTypeRO]:
        key: LocatedKey[KeyType, KeyTypeRO] = LocatedKey(
            instance.keyfs, self.key_name, f"{self.location}/{self.name}", self.key_type
        )
        # cache attribute
        instance.__dict__[self.key_name] = key
        return key


class PatternedKeyDescriptor(Generic[KeyType, KeyTypeRO]):
    __slots__ = ("key_name", "key_rotype", "key_type", "parent_key", "pattern")

    def __init__(
        self,
        key_name: str,
        pattern: str,
        parent_key: PatternedKey | PatternedKeyDescriptor | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> None:
        assert "{" in pattern or not pattern, pattern
        assert not isinstance(parent_key, LocatedKey)
        self.key_name = validated_key_name(key_name)
        self.pattern = pattern
        self.parent_key = parent_key
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema]
    ) -> PatternedKey[KeyType, KeyTypeRO]:
        parent_key = self.parent_key
        if isinstance(parent_key, PatternedKeyDescriptor):
            parent_key = parent_key.__get__(instance, owner)
        key: PatternedKey[KeyType, KeyTypeRO] = PatternedKey(
            instance.keyfs,
            self.key_name,
            self.pattern,
            parent_key,
            self.key_type,
        )
        # cache attribute
        instance.__dict__[self.key_name] = key
        return key


class KeyFSSchema(metaclass=KeyFSSchemaMeta):
    def __init__(self, keyfs: KeyFS):
        self.keyfs = keyfs

    def __iter__(self) -> Iterator[LocatedKey | PatternedKey]:
        for _name, key in getmembers(
            self, lambda x: isinstance(x, (LocatedKey, PatternedKey))
        ):
            yield key

    @overload
    def anonymous_key(
        self,
        key_name: str,
        parent_key: None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> LocatedKey[KeyType, KeyTypeRO]: ...

    @overload
    def anonymous_key(
        self,
        key_name: str,
        parent_key: PatternedKey,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> PatternedKey[KeyType, KeyTypeRO]: ...

    def anonymous_key(
        self,
        key_name: str,
        parent_key: PatternedKey | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> LocatedKey[KeyType, KeyTypeRO] | PatternedKey[KeyType, KeyTypeRO]:
        assert not hasattr(self, key_name)
        if parent_key is None:
            setattr(
                self,
                key_name,
                LocatedKeyDescriptor(key_name, "", "", key_type, key_rotype).__get__(
                    self, self.__class__
                ),
            )
        else:
            setattr(
                self,
                key_name,
                PatternedKeyDescriptor(
                    key_name, "", parent_key, key_type, key_rotype
                ).__get__(self, self.__class__),
            )
        return getattr(self, key_name)

    def located_key(
        self,
        key_name: str,
        location: str,
        name: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> LocatedKey[KeyType, KeyTypeRO]:
        assert not hasattr(self, key_name)
        setattr(
            self,
            key_name,
            LocatedKeyDescriptor(
                key_name, location, name, key_type, key_rotype
            ).__get__(self, self.__class__),
        )
        key = getattr(self, key_name)
        self.keyfs._storage.register_key(key)
        return key

    def patterned_key(
        self,
        key_name: str,
        pattern: str,
        parent_key: PatternedKey | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> PatternedKey[KeyType, KeyTypeRO]:
        assert not hasattr(self, key_name)
        setattr(
            self,
            key_name,
            PatternedKeyDescriptor(
                key_name, pattern, parent_key, key_type, key_rotype
            ).__get__(self, self.__class__),
        )
        key = getattr(self, key_name)
        self.keyfs._storage.register_key(key)
        return key
