try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum  # noqa: F401


try:
    from inspect import get_annotations
except ImportError:

    def get_annotations(obj):
        if hasattr(obj, "__annotations__"):
            return obj.__annotations__
        if hasattr(obj, "__call__"):  # noqa: B004 - we actually need to check for the attribute here
            return get_annotations(obj.__call__)
        if hasattr(obj, "__func__"):
            return get_annotations(obj.__func__)
        return {}
