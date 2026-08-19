from __future__ import annotations

from .config import hookimpl
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .keyfs_sqla_base import BaseConnection
from .keyfs_sqla_base import Writer
from .keyfs_sqla_base import cache_metrics
from .keyfs_sqla_lite_base import LiteBaseStorage
from .keyfs_types import StorageInfo
from typing import TYPE_CHECKING
from zope.interface import implementer
import sqlalchemy as sa


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pyramid.request import Request
    from typing import Any


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
class Storage(LiteBaseStorage):
    Connection = Connection
    db_filename = ".sqlite_alchemy"

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
        c.execute("PRAGMA busy_timeout=1000")
        c.execute("PRAGMA cache_size = 200000")
        c.close()

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
    result: list[tuple[str, str, Any]] = []
    xom = request.registry["xom"]
    storage = xom.keyfs._storage
    if isinstance(storage, Storage):
        result.extend(cache_metrics(storage))
    return result
