from __future__ import annotations

from .config import hookimpl
from .filestore_fs_base import LazyChangesFormatter
from .interfaces import IDBIOFileConnection
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .keyfs_sqla_base import BaseConnection
from .keyfs_sqla_base import Writer
from .keyfs_sqla_base import cache_metrics
from .keyfs_sqla_lite_base import LiteBaseStorage
from .keyfs_types import StorageInfo
from .log import threadlog
from .markers import Absent
from .markers import absent
from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING
from zope.interface import implementer
import os
import shutil
import sqlalchemy as sa


if TYPE_CHECKING:
    from .keyfs_types import FilePathInfo
    from collections.abc import Sequence
    from pyramid.request import Request
    from sqlalchemy.types import _Binary
    from typing import IO


@implementer(IDBIOFileConnection)
@implementer(IStorageConnection)
class Connection(BaseConnection):
    files_table: sa.Table
    storage: Storage

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
        f = SpooledTemporaryFile()  # noqa: SIM115 - the file obj is returned
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
            f = SpooledTemporaryFile(max_size=1048576)  # noqa: SIM115 - the file obj is stored
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

    def analyze(self) -> None:
        super().analyze()
        self.storage.ro_engine.dispose()


@implementer(IStorage)
class Storage(LiteBaseStorage):
    Connection = Connection
    db_filename = ".sqlite_alchemy_files"

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
        process_settings=Storage.process_settings,
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
