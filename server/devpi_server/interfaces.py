from __future__ import annotations

from contextlib import closing
from inspect import getfullargspec
from typing import TYPE_CHECKING
from zope.interface import Attribute
from zope.interface import Interface
from zope.interface.interface import adapter_hooks
from zope.interface.verify import verifyObject


if TYPE_CHECKING:
    from .keyfs_types import FilePathInfo
    from .keyfs_types import PTypedKey
    from .keyfs_types import Record
    from .keyfs_types import RelpathInfo
    from .keyfs_types import TypedKey
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path
    from types import TracebackType
    from typing import Any
    from typing import Callable
    from typing import IO
    from typing import Optional


class IDBIOFileConnection(Interface):
    def commit_files_without_increasing_serial() -> None:
        """Writes any files which have been changed without
        increasing the serial."""

    def io_file_delete(path: FilePathInfo, *, is_last_of_hash: bool) -> None:
        """Deletes the file at path."""

    def io_file_exists(path: FilePathInfo) -> bool:
        """Returns True if file at path exists."""

    def io_file_get(path: FilePathInfo) -> bytes:
        """Returns binary content of the file at path."""

    def io_file_new_open(path: FilePathInfo) -> IO[bytes]:
        """Returns a new open file like object for binary writing."""

    def io_file_open(path: FilePathInfo) -> IO[bytes]:
        """Returns an open file like object for binary reading."""

    def io_file_os_path(path: FilePathInfo) -> str | None:
        """Returns the real path to the file if the storage is filesystem
        based, otherwise None."""

    def io_file_set(path: FilePathInfo, content_or_file: bytes | IO[bytes]) -> None:
        """Set the binary content of the file at path."""

    def io_file_size(path: FilePathInfo) -> int | None:
        """Returns the size of the file at path."""


class IIOFile(Interface):
    def __enter__() -> AbstractContextManager:
        pass

    def __exit__(  # noqa: PLE0302, PYI036
        cls: type[BaseException] | None,
        val: BaseException | None,  # noqa: PYI036
        tb: TracebackType | None,  # noqa: PYI036
    ) -> bool | None:
        pass

    def commit() -> None:
        """Commit changed files to storage."""

    def delete(path: FilePathInfo, *, is_last_of_hash: bool) -> None:
        """Deletes the file at path."""

    def exists(path: FilePathInfo) -> bool:
        """Returns True if file at path exists."""

    def get_content(path: FilePathInfo) -> bytes:
        """Returns binary content of the file at path."""

    def is_dirty() -> bool:
        """Indicate whether there are any file changes pending."""

    def is_path_dirty(path: str) -> bool:
        """Indicate whether the real path to the file, if the storage is
        filesystem based, is dirty."""

    def new_open(path: FilePathInfo) -> IO[bytes]:
        """Returns a new open file like object for binary writing."""

    def open_read(path: FilePathInfo) -> IO[bytes]:
        """Returns an open file like object for binary reading."""

    def os_path(path: FilePathInfo) -> str | None:
        """Returns the real path to the file if the storage is filesystem
        based, otherwise None."""

    def perform_crash_recovery() -> None:
        """Perform recovery from crash during two phase commit."""

    def rollback() -> None:
        """Rollback changes to files."""

    def set_content(path: FilePathInfo, content_or_file: bytes | IO[bytes]) -> None:
        """Set the binary content of the file at path."""

    def size(path: FilePathInfo) -> int | None:
        """Returns the size of the file at path."""


class IStorage(Interface):
    def __init__(
        basedir: Path, *, notify_on_commit: Callable, cache_size: int, settings: dict
    ) -> None:
        """Create the storage object and initialize it."""

    def add_key(key: PTypedKey | TypedKey) -> None:
        """Register key information."""

    def get_connection(
        *, closing: bool, write: bool, timeout: int
    ) -> IStorageConnection:
        """Returns a connection to the storage."""

    def perform_crash_recovery() -> None:
        """Perform recovery from crash during two phase commit."""


class IStorageConnection(Interface):
    def db_read_last_changelog_serial() -> int:
        """ Return last stored serial.
            Returns -1 if nothing is stored yet. """

    def db_read_typedkey(relpath: str) -> tuple[str, int]:
        """ Return key name and serial for given relpath.
            Raises KeyError if not found. """

    def get_changes(serial: int) -> dict:
        """ Returns deserialized readonly changes for given serial. """

    def get_raw_changelog_entry(serial: int) -> Optional[bytes]:
        """Returns serialized changes for given serial."""

    def get_rel_renames(serial: int) -> Optional[Iterable]:
        """Returns deserialized rel_renames for given serial."""

    def get_relpath_at(relpath: str, serial: int) -> Any:
        """Get tuple of (last_serial, back_serial, value) for given relpath
        at given serial.
        Raises KeyError if not found."""

    def iter_relpaths_at(
        typedkeys: Iterable[PTypedKey | TypedKey], at_serial: int
    ) -> Iterator[RelpathInfo]:
        """Iterate over all relpaths of the given typed keys starting
        from at_serial until the first serial in the database."""

    def write_transaction(io_file: IIOFile | None) -> IWriter:
        """Returns a context providing class with a IWriter interface."""


class IWriter(Interface):
    commit_serial = Attribute("""
        The current to be commited serial set when entering the context manager. """)

    def __enter__() -> AbstractContextManager:
        pass

    def __exit__(  # noqa: PLE0302, PYI036
        cls: type[BaseException] | None,
        val: BaseException | None,  # noqa: PYI036
        tb: TracebackType | None,  # noqa: PYI036
    ) -> bool | None:
        pass

    def records_set(records: Sequence[Record]) -> None:
        pass

    def set_rel_renames(rel_renames: Sequence[str]) -> None:
        pass


# some adapters for legacy plugins


def unwrap_connection_obj(obj: Any) -> Any:
    if isinstance(obj, closing):
        obj = obj.thing  # type: ignore[attr-defined]
    return obj


def get_connection_class(obj: Any) -> type:
    return unwrap_connection_obj(obj).__class__


def verify_connection_interface(obj: Any) -> None:
    verifyObject(IStorageConnection, unwrap_connection_obj(obj))


_adapters = {}


def _register_adapter(func: Callable) -> None:
    spec = getfullargspec(func)
    iface = spec.annotations[spec.args[0]]
    if isinstance(iface, str):
        iface = globals()[iface]
    if iface in _adapters:
        msg = f"Adapter for {iface.getName()!r} already registered."
        raise RuntimeError(msg)
    _adapters[iface] = func


@adapter_hooks.append
def adapt(iface: Interface, obj: Any) -> Any:
    if iface in _adapters:
        return _adapters[iface](iface, obj)
    msg = f"don't know how to adapt {obj!r} to {iface.getName()!r}."  # type: ignore[attr-defined]
    raise ValueError(msg)
