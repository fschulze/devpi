from __future__ import annotations

from .keyfs_types import KeyType
from .keyfs_types import KeyTypeRO
from .keyfs_types import LocatedKey
from .keyfs_types import NamedKey
from .keyfs_types import NamedKeyFactory
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
        parent_key: NamedKeyDescriptor | NamedKeyFactoryDescriptor,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> NamedKeyDescriptor[KeyType, KeyTypeRO]: ...

    @classmethod
    def decl_anonymous_key(
        cls,
        key_name: str,
        parent_key: NamedKeyDescriptor | NamedKeyFactoryDescriptor | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> (
        LocatedKeyDescriptor[KeyType, KeyTypeRO]
        | NamedKeyDescriptor[KeyType, KeyTypeRO]
    ):
        if parent_key is None:
            return LocatedKeyDescriptor(key_name, "", "", key_type, key_rotype)
        return NamedKeyDescriptor(key_name, "", parent_key, key_type, key_rotype)

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
    def decl_named_key(
        cls,
        key_name: str,
        pattern_or_name: str,
        parent_key: NamedKeyDescriptor | NamedKeyFactoryDescriptor,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> NamedKeyDescriptor[KeyType, KeyTypeRO]:
        return NamedKeyDescriptor(
            key_name, pattern_or_name, parent_key, key_type, key_rotype
        )

    @classmethod
    def decl_named_key_factory(
        cls,
        key_name: str,
        pattern_or_name: str,
        parent_key: NamedKeyDescriptor | NamedKeyFactoryDescriptor | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> NamedKeyFactoryDescriptor[KeyType, KeyTypeRO]:
        return NamedKeyFactoryDescriptor(
            key_name, pattern_or_name, parent_key, key_type, key_rotype
        )


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
        self.key_name = key_name
        self.location = location
        self.name = name
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema] | None = None
    ) -> LocatedKey[KeyType, KeyTypeRO]:
        key: LocatedKey[KeyType, KeyTypeRO] = LocatedKey(
            instance.keyfs, self.key_name, self.location, self.name, self.key_type
        )
        # cache attribute
        instance.__dict__[self.key_name] = key
        return key


class NamedKeyDescriptor(Generic[KeyType, KeyTypeRO]):
    __slots__ = ("key_name", "key_rotype", "key_type", "parent_key", "pattern_or_name")

    def __init__(
        self,
        key_name: str,
        pattern_or_name: str,
        parent_key: NamedKey
        | NamedKeyDescriptor
        | NamedKeyFactory
        | NamedKeyFactoryDescriptor,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> None:
        assert "{" not in pattern_or_name
        self.key_name = key_name
        self.pattern_or_name = pattern_or_name
        self.parent_key = parent_key
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema]
    ) -> NamedKey[KeyType, KeyTypeRO]:
        parent_key = self.parent_key
        if isinstance(parent_key, NamedKeyDescriptor):
            parent_key = parent_key.__get__(instance, owner)
        if isinstance(parent_key, NamedKeyFactoryDescriptor):
            parent_key = parent_key.__get__(instance, owner)
        key: NamedKey[KeyType, KeyTypeRO] = NamedKey(
            instance.keyfs,
            self.key_name,
            self.pattern_or_name,
            parent_key,
            self.key_type,
        )
        # cache attribute
        instance.__dict__[self.key_name] = key
        return key


class NamedKeyFactoryDescriptor(Generic[KeyType, KeyTypeRO]):
    __slots__ = ("key_name", "key_rotype", "key_type", "parent_key", "pattern_or_name")

    def __init__(
        self,
        key_name: str,
        pattern_or_name: str,
        parent_key: NamedKey
        | NamedKeyDescriptor
        | NamedKeyFactory
        | NamedKeyFactoryDescriptor
        | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> None:
        assert "{" in pattern_or_name
        self.key_name = key_name
        self.pattern_or_name = pattern_or_name
        self.parent_key = parent_key
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema]
    ) -> NamedKeyFactory[KeyType, KeyTypeRO]:
        parent_key = self.parent_key
        if isinstance(parent_key, NamedKeyDescriptor):
            parent_key = parent_key.__get__(instance, owner)
        if isinstance(parent_key, NamedKeyFactoryDescriptor):
            parent_key = parent_key.__get__(instance, owner)
        key: NamedKeyFactory[KeyType, KeyTypeRO] = NamedKeyFactory(
            instance.keyfs,
            self.key_name,
            self.pattern_or_name,
            parent_key,
            self.key_type,
        )
        # cache attribute
        instance.__dict__[self.key_name] = key
        return key


class KeyFSSchema(metaclass=KeyFSSchemaMeta):
    def __init__(self, keyfs: KeyFS):
        self.keyfs = keyfs

    def __iter__(self) -> Iterator[LocatedKey | NamedKey]:
        for _name, key in getmembers(
            self, lambda x: isinstance(x, (LocatedKey, NamedKey))
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
        parent_key: NamedKey | NamedKeyFactory,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> NamedKey[KeyType, KeyTypeRO]: ...

    def anonymous_key(
        self,
        key_name: str,
        parent_key: NamedKey | NamedKeyFactory | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> LocatedKey[KeyType, KeyTypeRO] | NamedKey[KeyType, KeyTypeRO]:
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
                NamedKeyDescriptor(
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
        return getattr(self, key_name)

    def named_key(
        self,
        key_name: str,
        pattern_or_name: str,
        parent_key: NamedKey | NamedKeyFactory,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> NamedKey[KeyType, KeyTypeRO]:
        assert not hasattr(self, key_name)
        setattr(
            self,
            key_name,
            NamedKeyDescriptor(
                key_name, pattern_or_name, parent_key, key_type, key_rotype
            ).__get__(self, self.__class__),
        )
        return getattr(self, key_name)

    def named_key_factory(
        self,
        key_name: str,
        pattern_or_name: str,
        parent_key: NamedKey | NamedKeyFactory | None,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> NamedKeyFactory[KeyType, KeyTypeRO]:
        assert not hasattr(self, key_name)
        setattr(
            self,
            key_name,
            NamedKeyFactoryDescriptor(
                key_name, pattern_or_name, parent_key, key_type, key_rotype
            ).__get__(self, self.__class__),
        )
        return getattr(self, key_name)
