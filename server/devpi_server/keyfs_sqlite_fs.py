from .config import hookimpl
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .keyfs_sqlite import BaseConnection
from .keyfs_sqlite import BaseStorage
from .keyfs_sqlite import Writer
from .keyfs_types import StorageInfo
from zope.interface import implementer


@implementer(IStorageConnection)
class Connection(BaseConnection):
    def _write_dirty_files(self):
        return ([], [])


@implementer(IStorage)
class Storage(BaseStorage):
    Connection = Connection
    db_filename = ".sqlite"
    expected_schema = dict(
        index=dict(
            kv_serial_idx="""
                CREATE INDEX kv_serial_idx ON kv (serial);
            """,
            kv_key_keyname_idx="""
                CREATE UNIQUE INDEX kv_key_keyname_idx ON kv (key, keyname);
            """,
        ),
        table=dict(
            changelog="""
                CREATE TABLE changelog (
                    serial INTEGER PRIMARY KEY,
                    data BLOB NOT NULL
                )
            """,
            kv="""
                CREATE TABLE kv (
                    key TEXT NOT NULL,
                    keyname TEXT NOT NULL,
                    serial INTEGER
                )
            """,
        ),
    )

    def close(self):
        pass

    def perform_crash_recovery(self):
        pass


@hookimpl
def devpiserver_describe_storage_backend(settings):
    return StorageInfo(
        name="sqlite",
        description="SQLite backend with files on the filesystem",
        exists=Storage.exists,
        storage_cls=Storage,
        connection_cls=Connection,
        writer_cls=Writer,
        storage_factory=Storage,
        settings=settings,
    )
