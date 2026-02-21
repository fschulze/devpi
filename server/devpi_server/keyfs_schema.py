from __future__ import annotations

from .keyfs_types import KeyType
from .keyfs_types import KeyTypeRO
from .keyfs_types import PTypedKey
from .keyfs_types import RelPath
from .keyfs_types import TypedKey
from inspect import getmembers
from typing import Generic
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .keyfs import KeyFS
    from collections.abc import Iterator


class KeyFSSchemaMeta(type):
    @classmethod
    def decl_ptypedkey(
        cls,
        name: str,
        path: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> PTypedKeyDescriptor[KeyType, KeyTypeRO]:
        return PTypedKeyDescriptor(name, path, key_type, key_rotype)

    @classmethod
    def decl_typedkey(
        cls,
        name: str,
        path: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> TypedKeyDescriptor[KeyType, KeyTypeRO]:
        return TypedKeyDescriptor(name, path, key_type, key_rotype)


class PTypedKeyDescriptor(Generic[KeyType, KeyTypeRO]):
    __slots__ = ("key_rotype", "key_type", "name", "path")

    def __init__(
        self,
        name: str,
        path: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> None:
        assert "{" in path
        self.name = name
        self.path = path
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema] | None = None
    ) -> PTypedKey[KeyType, KeyTypeRO]:
        key: PTypedKey[KeyType, KeyTypeRO] = PTypedKey(
            instance.keyfs, self.path, self.key_type, self.name
        )
        # cache attribute
        instance.__dict__[self.name] = key
        return key


class TypedKeyDescriptor(Generic[KeyType, KeyTypeRO]):
    __slots__ = ("key_rotype", "key_type", "name", "path")

    def __init__(
        self,
        name: str,
        path: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],
    ) -> None:
        assert "{" not in path
        self.name = name
        self.path = path
        self.key_type = key_type
        self.key_rotype = key_rotype

    def __get__(
        self, instance: KeyFSSchema, owner: type[KeyFSSchema]
    ) -> TypedKey[KeyType, KeyTypeRO]:
        key: TypedKey[KeyType, KeyTypeRO] = TypedKey(
            instance.keyfs, RelPath(self.path), self.key_type, self.name
        )
        # cache attribute
        instance.__dict__[self.name] = key
        return key


class KeyFSSchema(metaclass=KeyFSSchemaMeta):
    def __init__(self, keyfs: KeyFS):
        self.keyfs = keyfs

    def __iter__(self) -> Iterator[PTypedKey | TypedKey]:
        for _name, key in getmembers(
            self, lambda x: isinstance(x, (PTypedKey, TypedKey))
        ):
            yield key

    def ptypedkey(
        self,
        name: str,
        path: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],  # noqa: ARG002 - typing only
    ) -> PTypedKey[KeyType, KeyTypeRO]:
        assert not hasattr(self, name)
        setattr(
            self,
            name,
            PTypedKeyDescriptor(name, path, key_type, key_rotype).__get__(
                self, self.__class__
            ),
        )
        return getattr(self, name)

    def typedkey(
        self,
        name: str,
        path: str,
        key_type: type[KeyType],
        key_rotype: type[KeyTypeRO],  # noqa: ARG002 - typing only
    ) -> TypedKey[KeyType, KeyTypeRO]:
        assert not hasattr(self, name)
        setattr(
            self,
            name,
            TypedKeyDescriptor(name, path, key_type, key_rotype).__get__(
                self, self.__class__
            ),
        )
        return getattr(self, name)
