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


from tempfile import SpooledTemporaryFile


# before Python 3.11 some methods were missing
if not hasattr(SpooledTemporaryFile, "readable"):

    def readable(self):
        return self._file.readable()

    SpooledTemporaryFile.readable = readable  # type: ignore[method-assign]

if not hasattr(SpooledTemporaryFile, "readinto"):

    def readinto(self, buffer):
        return self._file.readinto(buffer)

    SpooledTemporaryFile.readinto = readinto  # type: ignore[attr-defined]

if not hasattr(SpooledTemporaryFile, "seekable"):

    def seekable(self):
        return self._file.seekable()

    SpooledTemporaryFile.seekable = seekable  # type: ignore[method-assign]

if not hasattr(SpooledTemporaryFile, "writable"):

    def writable(self):
        return self._file.writable()

    SpooledTemporaryFile.writable = writable  # type: ignore[method-assign]
