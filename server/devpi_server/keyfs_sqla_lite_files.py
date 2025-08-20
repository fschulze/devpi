from __future__ import annotations

from .config import hookimpl
from .filestore_fs import LazyChangesFormatter
from .fileutil import SpooledTemporaryFile
from .interfaces import IDBIOFileConnection
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .keyfs import KeyfsTimeoutError
from .keyfs_sqla_base import BaseConnection
from .keyfs_sqla_base import BaseStorage
from .keyfs_sqla_base import Writer
from .keyfs_sqla_base import cache_metrics
from .keyfs_types import StorageInfo
from .log import threadlog
from .markers import Absent
from .markers import absent
from .mythread import current_thread
from io import BytesIO
from typing import TYPE_CHECKING
from typing import overload
from zope.interface import implementer
import contextlib
import os
import shutil
import sqlalchemy as sa
import time


if TYPE_CHECKING:
    from .keyfs_types import FilePathInfo
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path
    from pyramid.request import Request
    from sqlalchemy.types import _Binary
    from typing import Callable
    from typing import IO
    from typing import Literal


@implementer(IDBIOFileConnection)
@implementer(IStorageConnection)
class Connection(BaseConnection):
    files_table: sa.Table

    def commit_files_without_increasing_serial(self) -> None:
        try:
            (files_commit, files_del) = self._write_dirty_files()
            if files_commit or files_del:
                threadlog.debug(
                    "wrote files without increasing serial: %s",
                    LazyChangesFormatter((), files_commit, files_del),
                )
        except BaseException:
            self.rollback()
            raise
        else:
            self.commit()

    def get_next_serial(self) -> int:
        return self.last_changelog_serial + 1

    def io_file_delete(self, path: FilePathInfo, *, is_last_of_hash: bool) -> None:  # noqa: ARG002
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.pop(path.relpath, None)
        if f is not None:
            f.close()
        self.dirty_files[path.relpath] = None

    def io_file_exists(self, path: FilePathInfo) -> bool:
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is not absent:
            return f is not None
        result = self._sqlaconn.execute(
            sa.select(self.files_table.c.path).where(
                self.files_table.c.path == path.relpath
            )
        ).scalar()
        return result is not None

    def io_file_get(self, path: FilePathInfo) -> bytes:
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise OSError
        if not isinstance(f, Absent):
            pos = f.tell()
            f.seek(0)
            content = f.read()
            f.seek(pos)
            return content
        content = self._sqlaconn.execute(
            sa.select(self.files_table.c.data).where(
                self.files_table.c.path == path.relpath
            )
        ).scalar()
        if content is None:
            raise OSError
        return content

    def io_file_new_open(self, path: FilePathInfo) -> IO[bytes]:  # noqa: ARG002
        return SpooledTemporaryFile(max_size=1048576)

    def io_file_open(self, path: FilePathInfo) -> IO[bytes]:
        dirty_file = self.dirty_files.get(path.relpath, absent)
        if dirty_file is None:
            raise OSError
        if isinstance(dirty_file, Absent):
            return BytesIO(self.io_file_get(path))
        f = SpooledTemporaryFile()
        # we need a new file to prevent the dirty_file from being closed
        dirty_file.seek(0)
        shutil.copyfileobj(dirty_file, f)
        dirty_file.seek(0)
        f.seek(0)
        return f

    def io_file_os_path(self, path: FilePathInfo) -> str | None:  # noqa: ARG002
        return None

    def io_file_set(
        self, path: FilePathInfo, content_or_file: bytes | IO[bytes]
    ) -> None:
        assert not os.path.isabs(path.relpath)
        assert not path.relpath.endswith("-tmp")
        f = self.dirty_files.get(path.relpath, None)
        if f is None:
            f = SpooledTemporaryFile(max_size=1048576)
        if isinstance(content_or_file, bytes):
            f.write(content_or_file)
            f.seek(0)
        else:
            assert content_or_file.seekable()
            content_or_file.seek(0)
            shutil.copyfileobj(content_or_file, f)
        self.dirty_files[path.relpath] = f

    def io_file_size(self, path: FilePathInfo) -> int | None:
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise OSError
        if not isinstance(f, Absent):
            pos = f.tell()
            size = f.seek(0, 2)
            f.seek(pos)
            return size
        result = self._sqlaconn.execute(
            sa.select(self.files_table.c.size).where(
                self.files_table.c.path == path.relpath
            )
        ).scalar()
        return None if result is None else result

    def _file_write(self, path: str, f: IO[bytes]) -> None:
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        f.seek(0)
        content = f.read()
        f.close()
        self._sqlaconn.execute(
            sa.insert(self.files_table).values(
                path=path, size=len(content), data=content
            )
        )

    def _file_delete(self, path: str) -> None:
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        self._sqlaconn.execute(
            sa.delete(self.files_table).where(self.files_table.c.path == path)
        )

    def _write_dirty_files(self) -> tuple[Sequence, Sequence]:
        files_del = []
        files_commit = []
        for path, f in self.dirty_files.items():
            if f is None:
                self._file_delete(path)
                files_del.append(path)
            else:
                self._file_write(path, f)
                files_commit.append(path)
        self.dirty_files.clear()
        return (files_commit, files_del)


@implementer(IStorage)
class Storage(BaseStorage):
    db_filename = ".sqlite_alchemy_files"

    def __init__(
        self,
        basedir: Path,
        *,
        notify_on_commit: Callable,
        cache_size: int,
        settings: dict,
    ) -> None:
        super().__init__(
            basedir,
            notify_on_commit=notify_on_commit,
            cache_size=cache_size,
            settings=settings,
        )
        self.sqlpath = self.basedir / self.db_filename
        self.ro_engine = sa.create_engine(self._url(mode="ro"), echo=False)
        self.rw_engine = sa.create_engine(
            self._url(mode="rw"), echo=False, poolclass=sa.NullPool
        )
        self.ensure_tables_exist()

    def close(self) -> None:
        self.ro_engine.dispose()
        self.rw_engine.dispose()

    def define_tables(
        self, metadata_obj: sa.MetaData, binary_type: type[_Binary]
    ) -> dict:
        tables = super().define_tables(metadata_obj, binary_type)
        assert "files_table" not in tables
        tables["files_table"] = sa.Table(
            "files",
            metadata_obj,
            sa.Column("path", sa.String, primary_key=True),
            sa.Column("size", sa.Integer, nullable=False),
            sa.Column("data", binary_type, nullable=False),
        )
        return tables

    @classmethod
    def exists(cls, basedir: Path, settings: dict) -> bool:  # noqa: ARG003
        sqlpath = basedir / cls.db_filename
        return sqlpath.exists()

    def _url(self, *, mode: str) -> str:
        return f"sqlite+pysqlite:///file:{self.sqlpath}?mode={mode}&timeout=30&uri=true"

    def ensure_tables_exist(self) -> None:
        metadata_obj = sa.MetaData()
        tables = self.define_tables(metadata_obj, sa.BINARY)
        for name, table in tables.items():
            setattr(self, name, table)
        if not self.sqlpath.exists():
            engine = sa.create_engine(
                self._url(mode="rwc"), echo=False, poolclass=sa.NullPool
            )
            metadata_obj.create_all(engine)
            engine.dispose()

    def _execute_conn_pragmas(self, conn: sa.Connection) -> None:
        c = conn.connection.cursor()
        c.execute("PRAGMA cache_size = 200000")
        c.close()

    @overload
    def get_connection(
        self, *, closing: Literal[True], write: bool = False, timeout: int = 30
    ) -> AbstractContextManager[Connection]:
        pass

    @overload
    def get_connection(
        self, *, closing: Literal[False], write: bool = False, timeout: int = 30
    ) -> Connection:
        pass

    def get_connection(
        self, *, closing: bool = True, write: bool = False, timeout: int = 30
    ) -> Connection | AbstractContextManager[Connection]:
        engine = self.rw_engine if write else self.ro_engine
        sqlaconn = engine.connect()
        self._execute_conn_pragmas(sqlaconn)
        if write:
            start_time = time.monotonic()
            thread = current_thread()
            while 1:
                try:
                    sqlaconn.execute(sa.text("begin immediate"))
                    break
                except sa.exc.OperationalError as e:
                    # another thread may be writing, give it a chance to finish
                    time.sleep(0.1)
                    if hasattr(thread, "exit_if_shutdown"):
                        thread.exit_if_shutdown()
                    elapsed = time.monotonic() - start_time
                    if elapsed > timeout:
                        # if it takes this long, something is wrong
                        msg = f"Timeout after {int(elapsed)} seconds."
                        raise KeyfsTimeoutError(msg) from e
        conn = Connection(sqlaconn, self)
        if closing:
            return contextlib.closing(conn)
        return conn

    def perform_crash_recovery(self) -> None:
        pass


@hookimpl
def devpiserver_describe_storage_backend(settings: dict) -> StorageInfo:
    return StorageInfo(
        name="sqla_lite_files",
        description="SQLite backend using SQLAlchemy with files in DB for testing only",
        exists=Storage.exists,
        hidden=True,
        storage_cls=Storage,
        connection_cls=Connection,
        writer_cls=Writer,
        storage_factory=Storage,
        settings=settings,
    )


@hookimpl
def devpiserver_metrics(request: Request) -> list[tuple[str, str, object]]:
    result: list[tuple[str, str, object]] = []
    xom = request.registry["xom"]
    storage = xom.keyfs._storage
    if isinstance(storage, Storage):
        result.extend(cache_metrics(storage))
    return result
