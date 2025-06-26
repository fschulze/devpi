from __future__ import annotations

from devpi_server.filestore_fs import LazyChangesFormatter
from devpi_server.fileutil import SpooledTemporaryFile
from devpi_server.interfaces import IDBIOFileConnection
from devpi_server.interfaces import IStorage
from devpi_server.interfaces import IStorageConnection
from devpi_server.keyfs_sqla_base import BaseConnection
from devpi_server.keyfs_sqla_base import BaseStorage
from devpi_server.keyfs_sqla_base import Writer
from devpi_server.keyfs_sqla_base import cache_metrics
from devpi_server.keyfs_types import StorageInfo
from devpi_server.log import threadlog
from devpi_server.markers import Absent
from devpi_server.markers import absent
from devpi_server.model import ensure_boolean
from io import BytesIO
from io import RawIOBase
from pg8000.native import literal
from pluggy import HookimplMarker
from sqlalchemy.dialects.postgresql import BYTEA
from typing import TYPE_CHECKING
from typing import overload
from zope.interface import implementer
import contextlib
import os
import shutil
import sqlalchemy as sa
import ssl


if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from devpi_server.keyfs_types import FilePathInfo
    from pathlib import Path
    from pyramid.request import Request
    from sqlalchemy.types import _Binary
    from typing import Callable
    from typing import IO
    from typing import Literal
    from typing_extensions import Buffer


devpiserver_hookimpl = HookimplMarker("devpiserver")


SIGNATURE = b"PGCOPY\n\xff\r\n\x00"


class FileIn(RawIOBase):
    _data_size: int
    _state: Callable[[], None] | str
    _to_read: int

    def __init__(self, target_f: IO[bytes]) -> None:
        self.target_f = target_f
        self._read_buffer = bytearray(65536)
        self._set_state("signature", len(SIGNATURE))
        self._data_size = -1

    @property
    def got_data(self) -> bool:
        return self._data_size != -1

    def _set_state(self, state: str, to_read: int) -> None:
        state_method = getattr(self, f"_handle_{state}", None)
        if callable(state_method):
            self._state = state_method
        else:
            self._state = state
        self._state_bytes_read = 0
        self._to_read = to_read

    def _handle_signature(self) -> None:
        signature = bytes(self._read_buffer[: self._to_read])
        if signature != SIGNATURE:
            raise RuntimeError(f"Invalid PGCOPY signature {signature!r}")
        self._set_state("flags", 4)

    def _handle_flags(self) -> None:
        flags = int.from_bytes(self._read_buffer[: self._to_read], "big")
        # ignore lower 16 bits
        if (flags & ~0xFFFF) != 0:
            raise RuntimeError(f"Invalid PGCOPY flags {flags!r}")
        self._set_state("headerextsize", 4)

    def _handle_headerextsize(self) -> None:
        headerextsize = int.from_bytes(self._read_buffer[: self._to_read], "big")
        if headerextsize == 0:
            self._set_state("tuplecount", 2)
        else:
            self._set_state("headerext", headerextsize)

    def _handle_headerext(self) -> None:
        self._set_state("tuplecount", 2)

    def _handle_tuplecount(self) -> None:
        tuplecount = int.from_bytes(self._read_buffer[: self._to_read], "big")
        if tuplecount == 1:
            self._set_state("datasize", 4)
        elif tuplecount == 65535:
            self._set_state("finished", 0)
        elif self._data_size != -1:
            raise RuntimeError("More than one tuple")
        else:
            raise RuntimeError(f"Invalid tuple count {tuplecount!r}")

    def _handle_datasize(self) -> None:
        self._data_size = int.from_bytes(self._read_buffer[: self._to_read], "big")
        self._set_state("data", 0)

    def write(self, b: Buffer) -> int:
        data = memoryview(b)
        data_bytes_read = 0
        while True:
            if self._state == "data":
                to_read = min(
                    self._data_size - self.target_f.tell(),
                    len(data) - data_bytes_read)
                if to_read > 0:
                    chunk = data[data_bytes_read:]
                    self.target_f.write(chunk)
                    return len(chunk)
                self._set_state("tuplecount", 2)
                continue
            to_read = min(
                self._to_read,
                len(self._read_buffer) - self._state_bytes_read)
            if to_read < 0:
                # buffer size exceeded or something else wrong
                raise RuntimeError("Can't read more data")
            chunk = data[data_bytes_read:data_bytes_read + to_read]
            self._read_buffer[
                self._state_bytes_read:
                self._state_bytes_read + to_read] = chunk
            data_bytes_read += len(chunk)
            self._state_bytes_read += len(chunk)
            if self._state_bytes_read < self._to_read:
                break
            if callable(self._state):
                self._state()
            elif self._state == "finished":
                break
            else:
                raise RuntimeError("Invalid state {state!r}")
        return data_bytes_read


class FileOut(RawIOBase):
    def __init__(self, path: str, source_f: IO[bytes]) -> None:
        self.source_f = source_f
        size = source_f.seek(0, 2)
        source_f.seek(0)
        encoded_path = path.encode("utf-8")
        self.header_f = BytesIO(
            b"".join(
                (
                    SIGNATURE,
                    b"\x00\x00\x00\x00",  # flags
                    b"\x00\x00\x00\x00",  # header extension
                    b"\x00\x03",  # num fields in tuple
                    len(encoded_path).to_bytes(4, "big"),
                    encoded_path,
                    b"\x00\x00\x00\x04",  # size of INTEGER for "size" field
                    size.to_bytes(4, "big"),
                    size.to_bytes(4, "big"),  # size of binary file field
                )
            )
        )
        self.footer_f = BytesIO(b"\xff\xff")

    def readinto(self, buffer: Buffer) -> int:
        assert hasattr(self.source_f, "readinto")
        count: int | None = self.header_f.readinto(buffer)
        if count:
            return count
        count = self.source_f.readinto(buffer)
        if count:
            return count
        count = self.footer_f.readinto(buffer)
        if count:
            return count
        return 0


@implementer(IDBIOFileConnection)
@implementer(IStorageConnection)
class Connection(BaseConnection):
    files_table: sa.Table
    storage: Storage

    def __init__(self, sqlaconn: sa.engine.Connection, storage: Storage) -> None:
        super().__init__(sqlaconn, storage)
        threadlog.debug("Using use_copy=%r", storage.use_copy)
        if storage.use_copy:
            self.io_file_open = self._copy_io_file_open
            self.io_file_get = self._copy_io_file_get
            self._file_write = self._copy_file_write
        else:
            self.io_file_open = self._select_io_file_open
            self.io_file_get = self._select_io_file_get
            self._file_write = self._insert_file_write

    def _lock(self) -> None:
        self._sqlaconn.execute(sa.select(sa.func.pg_advisory_xact_lock(1))).scalar()

    def get_next_serial(self) -> int:
        result = self._sqlaconn.execute(
            sa.select(sa.func.nextval("changelog_serial_seq"))
        ).scalar()
        assert result is not None
        return result

    def commit_files_without_increasing_serial(self) -> None:
        try:
            self._lock()
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

    def _copy_io_file_get(self, path: FilePathInfo) -> bytes:
        with self._copy_io_file_open(path) as f:
            res = f.read()
            if len(res) > 1048576:
                threadlog.warn(
                    "Read %.1f megabytes into memory in postgresql io_file_get for %s",
                    len(res) / 1048576,
                    path.relpath,
                )
            return res

    def _select_io_file_get(self, path: FilePathInfo) -> bytes:
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

    def _copy_io_file_open(self, path: FilePathInfo) -> IO[bytes]:
        dirty_file = self.dirty_files.get(path.relpath, absent)
        if dirty_file is None:
            raise OSError
        f = SpooledTemporaryFile()
        if not isinstance(dirty_file, Absent):
            # we need a new file to prevent the dirty_file from being closed
            dirty_file.seek(0)
            shutil.copyfileobj(dirty_file, f)
            dirty_file.seek(0)
            f.seek(0)
            return f
        q = f"""
            COPY (
                SELECT data FROM files WHERE path = {literal(path.relpath)})
            TO STDOUT WITH (FORMAT binary);"""  # noqa: S608 - we are escaping with literal
        stream = FileIn(f)
        self._sqlaconn.connection.run(  # type: ignore[attr-defined]
            q, path=path.relpath, stream=stream
        )
        if stream.got_data:
            f.seek(0)
            return f
        f.close()
        raise OSError(f"File not found at '{path.relpath}'")

    def _select_io_file_open(self, path: FilePathInfo) -> IO[bytes]:
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

    def _copy_file_write(self, path: str, f: IO[bytes]) -> None:
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        q = """
            COPY files (path, size, data) FROM STDIN WITH (FORMAT binary);"""
        f.seek(0)
        with self._sqlaconn.begin_nested():
            self._sqlaconn.connection.run(  # type: ignore[attr-defined]
                q, stream=FileOut(path, f)
            )
        f.close()

    def _insert_file_write(self, path: str, f: IO[bytes]) -> None:
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
        self._lock()
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
    SSL_OPT_KEYS = ("ssl_check_hostname", "ssl_ca_certs", "ssl_certfile", "ssl_keyfile")
    database = "devpi"
    host = "localhost"
    port = "5432"
    unix_sock = None
    user = "devpi"
    password = None
    use_copy = True
    ssl_context = None
    poolclass: type[sa.Pool] = sa.QueuePool

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

        for key in ("database", "host", "port", "unix_sock", "user", "password"):
            if key in settings:
                setattr(self, key, settings[key])

        if any(key in settings for key in self.SSL_OPT_KEYS):
            self.ssl_context = ssl_context = ssl.create_default_context(
                cafile=settings.get('ssl_ca_certs'))

            if 'ssl_certfile' in settings:
                ssl_context.load_cert_chain(settings['ssl_certfile'],
                                            keyfile=settings.get('ssl_keyfile'))

            check_hostname = settings.get('ssl_check_hostname')
            if check_hostname is not None and not ensure_boolean(check_hostname):
                ssl_context.check_hostname = False

        self.use_copy = ensure_boolean(
            settings.get("use_copy", os.environ.get("DEVPI_PG_USE_COPY", self.use_copy))
        )

        user = self.user if self.user else ""
        password = f":{self.password}" if self.password else ""
        url = f"postgresql+pg8000://{user}{password}@{self.host}:{self.port}/{self.database}"
        self.engine = sa.create_engine(
            url,
            connect_args=dict(unix_sock=self.unix_sock, ssl_context=self.ssl_context),
            echo=False,
            poolclass=self.poolclass,
        )
        self.ensure_tables_exist()

    def close(self) -> None:
        self.engine.dispose()

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
            sa.Column("data", BYTEA, nullable=False),
        )
        sa.Sequence("changelog_serial_seq", minvalue=0, start=0, metadata=metadata_obj)
        return tables

    @classmethod
    def exists(cls, basedir: Path, settings: dict) -> bool:  # noqa: ARG003
        return True

    def ensure_tables_exist(self) -> None:
        metadata_obj = sa.MetaData()
        tables = self.define_tables(metadata_obj, BYTEA)
        for name, table in tables.items():
            setattr(self, name, table)
        metadata_obj.create_all(self.engine)

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
        self,
        *,
        closing: bool = True,
        write: bool = False,  # noqa: ARG002
        timeout: int = 30,  # noqa: ARG002
    ) -> Connection | AbstractContextManager[Connection]:
        sqlaconn = self.engine.connect()
        conn = Connection(sqlaconn, self)
        if closing:
            return contextlib.closing(conn)
        return conn

    def perform_crash_recovery(self) -> None:
        pass


@devpiserver_hookimpl
def devpiserver_describe_storage_backend(settings: dict) -> StorageInfo:
    return StorageInfo(
        name="sqla_pg8000",
        description="Postgresql backend using SQLAlchemy with files in DB",
        exists=Storage.exists,
        storage_cls=Storage,
        connection_cls=Connection,
        writer_cls=Writer,
        storage_factory=Storage,
        settings=settings,
    )


@devpiserver_hookimpl
def devpiserver_metrics(request: Request) -> list[tuple[str, str, object]]:
    result: list[tuple[str, str, object]] = []
    xom = request.registry["xom"]
    storage = xom.keyfs._storage
    if isinstance(storage, Storage):
        result.extend(cache_metrics(storage))
    return result
