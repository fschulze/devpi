from __future__ import annotations

from .interfaces import IIOFile
from typing import TYPE_CHECKING
from zope.interface import implementer


if TYPE_CHECKING:
    from .keyfs_types import FilePathInfo
    from types import TracebackType
    from typing import Any
    from typing import IO
    from typing_extensions import Self


@implementer(IIOFile)
class DBIOFile:
    def __init__(
        self,
        conn: Any,
        settings: dict,  # noqa: ARG002
    ) -> None:
        self.conn = conn

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        cls: type[BaseException] | None,
        val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if cls is not None:
            self.rollback()
            return False
        return True

    def commit(self) -> None:
        return self.conn.commit_files_without_increasing_serial()

    def delete(self, path: FilePathInfo, *, is_last_of_hash: bool) -> None:
        return self.conn.io_file_delete(path, is_last_of_hash=is_last_of_hash)

    def exists(self, path: FilePathInfo) -> bool:
        return self.conn.io_file_exists(path)

    def get_content(self, path: FilePathInfo) -> bytes:
        return self.conn.io_file_get(path)

    def get_rel_renames(self) -> list:
        return []

    def is_dirty(self) -> bool:
        return bool(self.conn.dirty_files)

    def is_path_dirty(self, path: str) -> bool:
        return path in self.conn.dirty_files

    def new_open(self, path: FilePathInfo) -> IO[bytes]:
        return self.conn.io_file_new_open(path)

    def open_read(self, path: FilePathInfo) -> IO[bytes]:
        return self.conn.io_file_open(path)

    def os_path(self, path: FilePathInfo) -> str | None:
        return self.conn.io_file_os_path(path)

    def perform_crash_recovery(self) -> None:
        return self.conn.storage.perform_crash_recovery()

    def rollback(self) -> None:
        return self.conn.rollback()

    def set_content(
        self, path: FilePathInfo, content_or_file: bytes | IO[bytes]
    ) -> None:
        return self.conn.io_file_set(path, content_or_file)

    def size(self, path: FilePathInfo) -> int | None:
        return self.conn.io_file_size(path)
