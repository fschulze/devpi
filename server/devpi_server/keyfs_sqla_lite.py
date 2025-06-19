from __future__ import annotations

from .config import hookimpl
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .keyfs import KeyfsTimeoutError
from .keyfs_sqla_base import BaseConnection
from .keyfs_sqla_base import BaseStorage
from .keyfs_sqla_base import Writer
from .keyfs_sqla_base import cache_metrics
from .keyfs_types import StorageInfo
from .mythread import current_thread
from typing import TYPE_CHECKING
from typing import overload
from zope.interface import implementer
import contextlib
import sqlalchemy as sa
import time


if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path
    from pyramid.request import Request
    from typing import Callable
    from typing import Literal


@implementer(IStorageConnection)
class Connection(BaseConnection):
    storage: Storage

    def get_next_serial(self) -> int:
        return self.last_changelog_serial + 1

    def _write_dirty_files(self) -> tuple[Sequence, Sequence]:
        return ([], [])

    def analyze(self) -> None:
        super().analyze()
        self.storage.ro_engine.dispose()


@implementer(IStorage)
class Storage(BaseStorage):
    db_filename = ".sqlite_alchemy"

    def __init__(
        self, basedir: Path, *, notify_on_commit: Callable, settings: dict
    ) -> None:
        super().__init__(basedir, notify_on_commit=notify_on_commit, settings=settings)
        self.sqlpath = self.basedir / self.db_filename
        self.ro_engine = sa.create_engine(self._url(mode="ro"), echo=False)
        self.rw_engine = sa.create_engine(
            self._url(mode="rw"), echo=False, poolclass=sa.NullPool
        )
        self.ensure_tables_exist()

    def close(self) -> None:
        self.ro_engine.dispose()
        self.rw_engine.dispose()

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
        name="sqla_lite",
        description="SQLite backend using SQLAlchemy with files on the filesystem",
        exists=Storage.exists,
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
