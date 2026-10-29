from __future__ import annotations

from .config import hookimpl
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .keyfs_sqla_base import BaseConnection
from .keyfs_sqla_base import BaseStorage
from .keyfs_sqla_base import Writer
from .keyfs_sqla_base import cache_metrics
from .keyfs_types import StorageInfo
from pkgutil import resolve_name
from typing import TYPE_CHECKING
from zope.interface import implementer
import sqlalchemy as sa
import weakref


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from pathlib import Path
    from pyramid.request import Request


@implementer(IStorageConnection)
class Connection(BaseConnection):
    def get_next_serial(self) -> int:
        return self.last_changelog_serial + 1

    def _write_dirty_files(self) -> tuple[Sequence, Sequence]:
        return ([], [])


@implementer(IStorage)
class Storage(BaseStorage):
    Connection = Connection
    poolclass: type[sa.Pool] = sa.QueuePool

    def __init__(
        self, basedir: Path, *, notify_on_commit: Callable, settings: dict
    ) -> None:
        super().__init__(basedir, notify_on_commit=notify_on_commit, settings=settings)
        self.binary_type = resolve_name(
            settings.get("binary_type", "sqlalchemy.BINARY")
        )
        self.engine = sa.create_engine(
            settings["url"],
            echo=False,
            poolclass=self.poolclass,
        )
        weakref.finalize(self, self.engine.dispose)
        self.ensure_tables_exist()

    def ensure_tables_exist(self) -> None:
        metadata_obj = sa.MetaData()
        tables = self.define_tables(metadata_obj, self.binary_type)
        for name, table in tables.items():
            setattr(self, name, table)
        metadata_obj.create_all(self.engine)

    @classmethod
    def exists(cls, basedir: Path, settings: dict) -> bool:  # noqa: ARG003
        return True

    def perform_crash_recovery(self) -> None:
        pass


@hookimpl
def devpiserver_describe_storage_backend(settings: dict) -> StorageInfo:
    return StorageInfo(
        name="sqla",
        description="Generic backend using SQLAlchemy with files on the filesystem",
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
