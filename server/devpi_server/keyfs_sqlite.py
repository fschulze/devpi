from __future__ import annotations

from .config import hookimpl
from .filestore_fs import LazyChangesFormatter
from .fileutil import SpooledTemporaryFile
from .fileutil import dumps
from .fileutil import loads
from .interfaces import IDBIOFileConnection
from .interfaces import IStorage
from .interfaces import IStorageConnection
from .interfaces import IWriter
from .keyfs import KeyfsTimeoutError
from .keyfs_types import KeyData
from .keyfs_types import StorageInfo
from .log import thread_pop_log
from .log import thread_push_log
from .log import threadlog
from .markers import absent
from .markers import deleted
from .mythread import current_thread
from .readonly import ReadonlyView
from .readonly import ensure_deeply_readonly
from .readonly import get_mutable_deepcopy
from .sizeof import gettotalsizeof
from devpi_common.types import cached_property
from io import BytesIO
from repoze.lru import LRUCache
from typing import TYPE_CHECKING
from zope.interface import implementer
import contextlib
import os
import shutil
import sqlite3
import time


if TYPE_CHECKING:
    from .keyfs_types import IKeyFSKey
    from .keyfs_types import LocatedKey
    from .keyfs_types import Record
    from collections.abc import Iterable
    from collections.abc import Iterator


class BaseConnection:
    def __init__(self, sqlconn, basedir, storage):
        self._sqlconn = sqlconn
        self._basedir = basedir
        self.dirty_files = {}
        self.storage = storage
        self._changelog_cache = storage._changelog_cache
        self._relpath_cache = storage._relpath_cache

    def _explain(self, query, *args):
        # for debugging
        c = self._sqlconn.cursor()
        r = c.execute("EXPLAIN " + query, *args)
        result = r.fetchall()
        c.close()
        return result

    def _explain_query_plan(self, query, *args):
        # for debugging
        c = self._sqlconn.cursor()
        r = c.execute("EXPLAIN QUERY PLAN " + query, *args)
        result = r.fetchall()
        c.close()
        return result

    def _print_rows(self, rows):
        # for debugging
        for row in rows:
            print(row)  # noqa: T201

    def execute(self, query, *args):
        c = self._sqlconn.cursor()
        # print(query)
        # self._print_rows(self._explain(query, *args))
        # self._print_rows(self._explain_query_plan(query, *args))
        r = c.execute(query, *args)
        result = r.fetchall()
        c.close()
        return result

    def executemany(self, query, *args):
        c = self._sqlconn.cursor()
        # print(query)
        # self._print_rows(self._explain(query, *args))
        # self._print_rows(self._explain_query_plan(query, *args))
        r = c.executemany(query, *args)
        result = r.fetchall()
        c.close()
        return result

    def fetchall(self, query, *args):
        c = self._sqlconn.cursor()
        # print(query)
        # self._print_rows(self._explain(query, *args))
        # self._print_rows(self._explain_query_plan(query, *args))
        r = c.execute(query, *args)
        result = r.fetchall()
        c.close()
        return result

    def fetchone(self, query, *args):
        c = self._sqlconn.cursor()
        # print(query)
        # self._print_rows(self._explain(query, *args))
        # self._print_rows(self._explain_query_plan(query, *args))
        r = c.execute(query, *args)
        result = r.fetchone()
        c.close()
        return result

    def iterall(self, query, *args):
        c = self._sqlconn.cursor()
        # print(query)
        # self._print_rows(self._explain(query, *args))
        # self._print_rows(self._explain_query_plan(query, *args))
        yield from c.execute(query, *args)
        c.close()

    def lastrowid(self, query, *args):
        c = self._sqlconn.cursor()
        # print(query)
        # self._print_rows(self._explain(query, *args))
        # self._print_rows(self._explain_query_plan(query, *args))
        c.execute(query, *args)
        result = c.lastrowid
        c.close()
        return result

    def close(self):
        self._sqlconn.close()

    def commit(self):
        self._sqlconn.commit()

    def rollback(self):
        try:
            self._sqlconn.rollback()
        except sqlite3.ProgrammingError as e:
            if not e.args or 'closed database' not in e.args[0]:
                raise

    @cached_property
    def last_changelog_serial(self):
        return self.db_read_last_changelog_serial()

    def db_read_last_changelog_serial(self):
        q = 'SELECT MAX(_ROWID_) FROM "changelog" LIMIT 1'
        res = self.fetchone(q)[0]
        return -1 if res is None else res

    def last_key_serial(self, key: LocatedKey) -> int:
        q = "SELECT serial FROM kv WHERE keyname = ? AND key = ?"
        row = self.fetchone(q, (key.key_name, key.relpath))
        if row is None:
            raise KeyError(key)
        (serial,) = row
        return serial

    def db_write_typedkeys(self, data):
        new_typedkeys = []
        updated_typedkeys = []
        for key, keyname, serial, back_serial in data:
            if back_serial == -1:
                new_typedkeys.append(dict(key=key, keyname=keyname, serial=serial))
            else:
                updated_typedkeys.append(
                    dict(
                        key=key, keyname=keyname, serial=serial, back_serial=back_serial
                    )
                )
        if new_typedkeys:
            q = "INSERT INTO kv (key, keyname, serial) VALUES (:key, :keyname, :serial)"
            self.executemany(q, new_typedkeys)
        if updated_typedkeys:
            q = "UPDATE kv SET serial = :serial WHERE key = :key AND keyname = :keyname AND serial = :back_serial"
            self.executemany(q, updated_typedkeys)

    def write_changelog_entry(self, serial, entry):
        threadlog.debug("writing changelog for serial %s", serial)
        data = dumps(entry)
        self.execute(
            "INSERT INTO changelog (serial, data) VALUES (?, ?)",
            (serial, sqlite3.Binary(data)))

    def get_raw_changelog_entry(self, serial):
        q = "SELECT data FROM changelog WHERE serial = ?"
        row = self.fetchone(q, (serial,))
        if row is not None:
            return bytes(row[0])
        return None

    def iter_changes_at(self, serial: int) -> Iterator[KeyData]:
        changes = self._changelog_cache.get(serial, absent)
        if changes is absent:
            data = self.get_raw_changelog_entry(serial)
            if data is None:
                return
            changes, rel_renames = loads(data)
            # make values in changes read only so no calling site accidentally
            # modifies data
            changes = ensure_deeply_readonly(changes)
            assert isinstance(changes, ReadonlyView)
            self._changelog_cache.put(serial, changes)
        for keyname, relpath, back_serial, val in changes:
            yield KeyData(
                relpath=relpath,
                keyname=keyname,
                serial=serial,
                back_serial=back_serial,
                value=deleted if val is None else val,
            )

    def iter_rel_renames(self, serial: int) -> Iterator[str]:
        if serial == -1:
            return
        data = self.get_raw_changelog_entry(serial)
        (changes, rel_renames) = loads(data)
        yield from rel_renames

    def _get_key_at_serial(self, key: LocatedKey, serial: int) -> KeyData:
        last_serial = self.last_key_serial(key)
        serials_and_values = self._iter_serial_and_value_backwards(key, last_serial)
        try:
            keydata = next(serials_and_values)
            last_serial = keydata.last_serial
            while last_serial >= 0:
                if last_serial > serial:
                    keydata = next(serials_and_values)
                    last_serial = keydata.last_serial
                    continue
                return keydata
        except StopIteration:
            pass
        raise KeyError(key)

    def _iter_serial_and_value_backwards(
        self, key: LocatedKey, last_serial: int
    ) -> Iterator[KeyData]:
        while last_serial >= 0:
            changes = {
                (c.keyname, c.relpath): c for c in self.iter_changes_at(last_serial)
            }
            change = changes.get((key.key_name, key.relpath))
            if change is None:
                raise RuntimeError("no transaction entry at %s" % (last_serial))
            yield change
            last_serial = change.back_serial

        # we could not find any change below at_serial which means
        # the key didn't exist at that point in time

    def get_key_at_serial(self, key: LocatedKey, serial: int) -> KeyData:
        cache_key = (key.key_name, key.relpath)
        result = self._relpath_cache.get((serial, cache_key), absent)
        if result is absent:
            result = self._changelog_cache.get((serial, cache_key), absent)
        if result is absent:
            result = self._get_key_at_serial(key, serial)
        if (
            result.value is not deleted
            and gettotalsizeof(result.value, maxlen=100000) is None
        ):
            # result is big, put it in the changelog cache,
            # which has fewer entries to preserve memory
            self._changelog_cache.put((serial, cache_key), result)
        else:
            # result is small
            self._relpath_cache.put((serial, cache_key), result)
        return result

    def iter_keys_at_serial(
        self, keys: Iterable[IKeyFSKey], at_serial: int
    ) -> Iterator[KeyData]:
        keynames = frozenset(k.key_name for k in keys)
        keyname_id_values = {"keynameid%i" % i: k for i, k in enumerate(keynames)}
        q = """
            SELECT key, keyname
            FROM kv
            WHERE serial=:serial AND keyname IN (:keynames)
        """
        q = q.replace(':keynames', ", ".join(':' + x for x in keyname_id_values))
        for serial in range(at_serial, -1, -1):
            rows = self.fetchall(q, dict(
                serial=serial,
                **keyname_id_values))
            if not rows:
                continue
            changes = {c.relpath: c for c in self.iter_changes_at(serial)}
            for relpath, keyname in rows:
                change = changes[relpath]
                assert change.keyname == keyname
                yield change

    def write_transaction(self, io_file):
        return Writer(self.storage, self, io_file)


@implementer(IDBIOFileConnection)
@implementer(IStorageConnection)
class Connection(BaseConnection):
    def io_file_os_path(self, path):
        return None

    def io_file_exists(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is not absent:
            return f is not None
        q = "SELECT path FROM files WHERE path = ?"
        result = self.fetchone(q, (path.relpath,))
        return result is not None

    def io_file_set(self, path, content_or_file):
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

    def io_file_new_open(self, path):  # noqa: ARG002
        return SpooledTemporaryFile(max_size=1048576)

    def io_file_open(self, path):
        dirty_file = self.dirty_files.get(path.relpath, absent)
        if dirty_file is None:
            raise IOError()
        if dirty_file is absent:
            return BytesIO(self.io_file_get(path))
        f = SpooledTemporaryFile()
        # we need a new file to prevent the dirty_file from being closed
        dirty_file.seek(0)
        shutil.copyfileobj(dirty_file, f)
        dirty_file.seek(0)
        f.seek(0)
        return f

    def io_file_get(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise IOError()
        elif f is not absent:
            pos = f.tell()
            f.seek(0)
            content = f.read()
            f.seek(pos)
            return content
        q = "SELECT data FROM files WHERE path = ?"
        content = self.fetchone(q, (path.relpath,))
        if content is None:
            raise IOError()
        return bytes(content[0])

    def io_file_size(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise IOError()
        elif f is not absent:
            pos = f.tell()
            size = f.seek(0, 2)
            f.seek(pos)
            return size
        q = "SELECT size FROM files WHERE path = ?"
        result = self.fetchone(q, (path.relpath,))
        if result is not None:
            return result[0]

    def io_file_delete(self, path, *, is_last_of_hash):  # noqa: ARG002
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.pop(path.relpath, None)
        if f is not None:
            f.close()
        self.dirty_files[path.relpath] = None

    def _file_write(self, path, f):
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        q = "INSERT INTO files (path, size, data) VALUES (?, ?, ?)"
        f.seek(0)
        content = f.read()
        f.close()
        self.fetchone(q, (path, len(content), sqlite3.Binary(content)))

    def _file_delete(self, path):
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        q = "DELETE FROM files WHERE path = ?"
        self.fetchone(q, (path,))

    def _write_dirty_files(self):
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

    def commit_files_without_increasing_serial(self):
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


class BaseStorage:
    def __init__(self, basedir, *, notify_on_commit, cache_size, settings=None):  # noqa: ARG002
        self.basedir = basedir
        self.sqlpath = self.basedir / self.db_filename
        self._notify_on_commit = notify_on_commit
        changelog_cache_size = max(1, cache_size // 20)
        relpath_cache_size = max(1, cache_size - changelog_cache_size)
        self._changelog_cache = LRUCache(changelog_cache_size)  # is thread safe
        self._relpath_cache = LRUCache(relpath_cache_size)  # is thread safe
        self.last_commit_timestamp = time.time()
        self.ensure_tables_exist()

    @classmethod
    def exists(cls, basedir, settings):  # noqa: ARG003
        sqlpath = basedir / cls.db_filename
        return sqlpath.exists()

    def _get_sqlconn_uri_kw(self, uri):
        return sqlite3.connect(
            uri, timeout=60, isolation_level=None, uri=True)

    def _get_sqlconn_uri(self, uri):
        return sqlite3.connect(
            uri, timeout=60, isolation_level=None)

    def _get_sqlconn_path(self, uri):
        return sqlite3.connect(
            self.sqlpath.strpath, timeout=60, isolation_level=None)

    def _execute_conn_pragmas(self, sqlconn, *, write):  # noqa: ARG002
        c = sqlconn.cursor()
        c.execute("PRAGMA cache_size = 200000")
        c.close()

    def _get_sqlconn(self, uri):
        # we will try different connection methods and overwrite _get_sqlconn
        # with the first successful one
        try:
            # the uri keyword is only supported from Python 3.4 onwards and
            # possibly other Python implementations
            conn = self._get_sqlconn_uri_kw(uri)
            # remember for next time
            self._get_sqlconn = self._get_sqlconn_uri_kw
        except TypeError as e:
            if e.args and 'uri' in e.args[0] and 'keyword argument' in e.args[0]:
                threadlog.warn(
                    "The uri keyword for 'sqlite3.connect' isn't supported by "
                    "this Python version.")
            else:
                raise
        except sqlite3.OperationalError as e:
            threadlog.warn("%s" % e)
            threadlog.warn(
                "The installed version of sqlite3 doesn't seem to support "
                "the uri keyword for 'sqlite3.connect'.")
        except sqlite3.NotSupportedError:
            threadlog.warn(
                "The installed version of sqlite3 doesn't support the uri "
                "keyword for 'sqlite3.connect'.")
        else:
            return conn
        try:
            # sqlite3 might be compiled with default URI support
            conn = self._get_sqlconn_uri(uri)
            # remember for next time
            self._get_sqlconn = self._get_sqlconn_uri
        except sqlite3.OperationalError as e:
            # log the error and switch to using the path
            threadlog.warn("%s" % e)
            threadlog.warn(
                "Opening the sqlite3 db without options in URI. There is a "
                "higher possibility of read/write conflicts between "
                "threads, causing slowdowns due to retries.")
            conn = self._get_sqlconn_path(uri)
            # remember for next time
            self._get_sqlconn = self._get_sqlconn_path
            return conn
        else:
            return conn

    def register_key(self, key):
        pass

    def get_connection(self, *, closing=True, write=False, timeout=30):
        # we let the database serialize all writers at connection time
        # to play it very safe (we don't have massive amounts of writes).
        mode = "ro"
        if write:
            mode = "rw"
        if not self.sqlpath.exists():
            mode = "rwc"
        uri = "file:%s?mode=%s" % (self.sqlpath, mode)
        sqlconn = self._get_sqlconn(uri)
        self._execute_conn_pragmas(sqlconn, write=write)
        if write:
            start_time = time.monotonic()
            thread = current_thread()
            while 1:
                try:
                    sqlconn.execute("begin immediate")
                    break
                except sqlite3.OperationalError as e:
                    # another thread may be writing, give it a chance to finish
                    time.sleep(0.1)
                    if hasattr(thread, "exit_if_shutdown"):
                        thread.exit_if_shutdown()
                    elapsed = time.monotonic() - start_time
                    if elapsed > timeout:
                        # if it takes this long, something is wrong
                        raise KeyfsTimeoutError(
                            f"Timeout after {int(elapsed)} seconds.") from e
        conn = self.Connection(sqlconn, self.basedir, self)
        if closing:
            return contextlib.closing(conn)
        return conn

    def _reflect_schema(self):
        result = {}
        with self.get_connection(write=False) as conn:
            c = conn._sqlconn.cursor()
            rows = c.execute("""
                SELECT type, name, sql FROM sqlite_master""")
            for row in rows:
                result.setdefault(row[0], {})[row[1]] = row[2]
        return result

    def ensure_tables_exist(self):
        schema = self._reflect_schema()
        missing = dict()
        for kind, objs in self.expected_schema.items():
            for name, q in objs.items():
                if name not in schema.get(kind, set()):
                    missing.setdefault(kind, dict())[name] = q
        if not missing:
            return
        with self.get_connection(write=True) as conn:
            if not schema:
                threadlog.info("DB: Creating schema")
            else:
                threadlog.info("DB: Updating schema")
            c = conn._sqlconn.cursor()
            for kind in ('table', 'index'):
                objs = missing.pop(kind, {})
                for name in list(objs):
                    q = objs.pop(name)
                    c.execute(q)
                assert not objs
            c.close()
            conn.commit()
        assert not missing


@implementer(IStorage)
class Storage(BaseStorage):
    Connection = Connection
    db_filename = ".sqlite_db"
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
            files="""
                CREATE TABLE files (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    data BLOB NOT NULL
                )
            """,
        ),
    )

    def perform_crash_recovery(self):
        pass


@hookimpl
def devpiserver_describe_storage_backend(settings):
    return StorageInfo(
        name="sqlite_db_files",
        description="SQLite backend with files in DB for testing only",
        exists=Storage.exists,
        hidden=True,
        storage_cls=Storage,
        connection_cls=Connection,
        writer_cls=Writer,
        storage_factory=Storage,
        settings=settings,
    )


@hookimpl
def devpiserver_metrics(request):
    result = []
    xom = request.registry["xom"]
    storage = xom.keyfs._storage
    if not isinstance(storage, BaseStorage):
        return result
    changelog_cache = getattr(storage, '_changelog_cache', None)
    relpath_cache = getattr(storage, '_relpath_cache', None)
    if changelog_cache is None and relpath_cache is None:
        return result
    # get sizes for changelog_cache
    evictions = changelog_cache.evictions if changelog_cache else 0
    hits = changelog_cache.hits if changelog_cache else 0
    lookups = changelog_cache.lookups if changelog_cache else 0
    misses = changelog_cache.misses if changelog_cache else 0
    size = changelog_cache.size if changelog_cache else 0
    # add sizes for relpath_cache
    evictions += relpath_cache.evictions if relpath_cache else 0
    hits += relpath_cache.hits if relpath_cache else 0
    lookups += relpath_cache.lookups if relpath_cache else 0
    misses += relpath_cache.misses if relpath_cache else 0
    size += relpath_cache.size if relpath_cache else 0
    result.extend([
        ('devpi_server_storage_cache_evictions', 'counter', evictions),
        ('devpi_server_storage_cache_hits', 'counter', hits),
        ('devpi_server_storage_cache_lookups', 'counter', lookups),
        ('devpi_server_storage_cache_misses', 'counter', misses),
        ('devpi_server_storage_cache_size', 'gauge', size)])
    if changelog_cache:
        result.extend([
            ('devpi_server_changelog_cache_evictions', 'counter', changelog_cache.evictions),
            ('devpi_server_changelog_cache_hits', 'counter', changelog_cache.hits),
            ('devpi_server_changelog_cache_lookups', 'counter', changelog_cache.lookups),
            ('devpi_server_changelog_cache_misses', 'counter', changelog_cache.misses),
            ('devpi_server_changelog_cache_size', 'gauge', changelog_cache.size),
            ('devpi_server_changelog_cache_items', 'gauge', len(changelog_cache.data) if changelog_cache.data else 0)])
    if relpath_cache:
        result.extend([
            ('devpi_server_relpath_cache_evictions', 'counter', relpath_cache.evictions),
            ('devpi_server_relpath_cache_hits', 'counter', relpath_cache.hits),
            ('devpi_server_relpath_cache_lookups', 'counter', relpath_cache.lookups),
            ('devpi_server_relpath_cache_misses', 'counter', relpath_cache.misses),
            ('devpi_server_relpath_cache_size', 'gauge', relpath_cache.size),
            ('devpi_server_relpath_cache_items', 'gauge', len(relpath_cache.data) if relpath_cache.data else 0)])
    return result


@implementer(IWriter)
class Writer:
    def __init__(self, storage, conn, io_file):
        self.conn = conn
        self.io_file = io_file
        self.storage = storage
        self.rel_renames = []

    def records_set(self, records: Iterable[Record]) -> None:
        self.changes = []
        for record in records:
            assert not isinstance(record.value, ReadonlyView), record.value
            value = (
                None if record.value is deleted else get_mutable_deepcopy(record.value)
            )
            self.changes.append(
                (record.key.key_name, record.key.relpath, record.back_serial, value)
            )

    def set_rel_renames(self, rel_renames):
        self.rel_renames = rel_renames

    def __enter__(self):
        self.commit_serial = self.conn.last_changelog_serial + 1
        self.log = thread_push_log("fswriter%s:" % self.commit_serial)
        return self

    def __exit__(self, cls, val, tb):
        commit_serial = self.commit_serial
        try:
            del self.commit_serial
            if cls is None:
                changes_formatter = self.commit(commit_serial)
                self.log.info("committed at %s", commit_serial)
                self.log.debug("committed: %s", changes_formatter)

                self.storage._notify_on_commit(commit_serial)
            else:
                self.rollback()
                self.log.info("roll back at %s", commit_serial)
        except BaseException:
            self.rollback()
            raise
        finally:
            thread_pop_log("fswriter%s:" % commit_serial)

    def commit(self, commit_serial):
        data = []
        for keyname, relpath, back_serial, _value in self.changes:
            if back_serial is None:
                raise RuntimeError
            data.append((relpath, keyname, commit_serial, back_serial))
        self.conn.db_write_typedkeys(data)
        del data
        entry = (self.changes, self.rel_renames)
        self.conn.write_changelog_entry(commit_serial, entry)
        (files_commit, files_del) = self.conn._write_dirty_files()
        self.conn.commit()
        self.storage.last_commit_timestamp = time.time()
        return LazyChangesFormatter(
            {c[:2] for c in self.changes}, files_commit, files_del
        )

    def rollback(self):
        self.conn.rollback()
