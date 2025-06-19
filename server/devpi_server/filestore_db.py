from __future__ import annotations

from .interfaces import IDBIOFileConnection
from .interfaces import IIOFile
from typing import TYPE_CHECKING
from zope.interface import implementer
from zope.interface.verify import verifyObject


if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any
    from typing_extensions import Self


@implementer(IIOFile)
class DBIOFile:
    def __init__(self, conn: Any) -> None:
        conn = IDBIOFileConnection(conn)
        verifyObject(IDBIOFileConnection, conn)
        self._dirty_files = conn.dirty_files
        self.commit = conn.commit_files_without_increasing_serial
        self.delete = conn.io_file_delete
        self.exists = conn.io_file_exists
        self.get_content = conn.io_file_get
        self.new_open = conn.io_file_new_open
        self.open_read = conn.io_file_open
        self.os_path = conn.io_file_os_path
        self.perform_crash_recovery = getattr(
            conn.storage, "perform_crash_recovery", self._perform_crash_recovery
        )
        self.rollback = getattr(conn, "_drop_dirty_files", self._rollback)
        self.set_content = conn.io_file_set
        self.size = conn.io_file_size

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

    def get_rel_renames(self) -> list:
        return []

    def is_dirty(self) -> bool:
        return bool(self._dirty_files)

    def _perform_crash_recovery(self) -> None:
        pass

    def _rollback(self) -> None:
        pass
