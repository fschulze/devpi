from __future__ import annotations

from devpi_common.validation import safe_name_rex
from typing import cast
from typing_extensions import Self


class NormalizedName(str):
    __slots__ = ('original',)

    original: str

    def __new__(cls, name: Self | str) -> Self:
        if isinstance(name, NormalizedName):
            return cast(Self, name)
        result = super().__new__(
            cls, safe_name_rex.sub('-', name).lower())
        result.original = name
        return result

    @classmethod
    def from_strings(cls, original: str, normalized: str) -> Self:
        result = super().__new__(
            cls, normalized)
        result.original = original
        return result

    def __repr__(self) -> str:
        orig = super().__repr__()
        return f"<{self.__class__.__name__} {self.original!r} {orig}>"


normalize_name = NormalizedName
