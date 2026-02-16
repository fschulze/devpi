from __future__ import annotations


class Absent:
    __slots__ = ()

    def __repr__(self) -> str:
        return '<absent>'


absent = Absent()


class Deleted:
    __slots__ = ()

    def __repr__(self) -> str:
        return '<deleted>'


deleted = Deleted()


class NoDefault:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<nodefault>"


nodefault = NoDefault()


class NotSet:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<notset>"


notset = NotSet()


class Unknown:
    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<unknown>"


unknown = Unknown()
