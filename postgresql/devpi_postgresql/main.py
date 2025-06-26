from __future__ import annotations

from devpi_common.types import cached_property
from devpi_server.filestore_fs import LazyChangesFormatter
from devpi_server.fileutil import SpooledTemporaryFile
from devpi_server.fileutil import dumps
from devpi_server.fileutil import loads
from devpi_server.interfaces import IDBIOFileConnection
from devpi_server.interfaces import IStorage
from devpi_server.interfaces import IStorageConnection
from devpi_server.interfaces import IWriter
from devpi_server.keyfs_types import KeyData
from devpi_server.keyfs_types import StorageInfo
from devpi_server.log import thread_pop_log
from devpi_server.log import thread_push_log
from devpi_server.log import threadlog
from devpi_server.markers import absent
from devpi_server.model import ensure_boolean
from devpi_server.readonly import ReadonlyView
from devpi_server.readonly import ensure_deeply_readonly
from devpi_server.readonly import get_mutable_deepcopy
from devpi_server.sizeof import gettotalsizeof
from io import BytesIO
from io import RawIOBase
from pluggy import HookimplMarker
from repoze.lru import LRUCache
from typing import TYPE_CHECKING
from zope.interface import implementer
import contextlib
import os
import pg8000.native
import shutil
import ssl
import time


if TYPE_CHECKING:
    from collections.abc import Iterator


devpiserver_hookimpl = HookimplMarker("devpiserver")


SIGNATURE = b"PGCOPY\n\xff\r\n\x00"


class FileIn(RawIOBase):
    def __init__(self, target_f):
        self.target_f = target_f
        self._read_buffer = bytearray(65536)
        self._set_state("signature", len(SIGNATURE))
        self._data_size = -1

    @property
    def got_data(self):
        return self._data_size != -1

    def _set_state(self, state, to_read):
        self._state = state
        self._state_bytes_read = 0
        self._to_read = to_read

    def write(self, data):
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
            if to_read <= 0:
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
            if self._state == "signature":
                signature = bytes(self._read_buffer[:self._to_read])
                if signature != SIGNATURE:
                    raise RuntimeError(f"Invalid PGCOPY signature {signature!r}")
                self._set_state("flags", 4)
            elif self._state == "flags":
                flags = int.from_bytes(self._read_buffer[:self._to_read], "big")
                # ignore lower 16 bits
                if (flags & ~0xffff) != 0:
                    raise RuntimeError(f"Invalid PGCOPY flags {flags!r}")
                self._set_state("headerextsize", 4)
            elif self._state == "headerextsize":
                headerextsize = int.from_bytes(self._read_buffer[:self._to_read], "big")
                if headerextsize == 0:
                    self._set_state("tuplecount", 2)
                else:
                    self._set_state("headerext", headerextsize)
            elif self._state == "headerextsize":
                self._set_state("tuplecount", 2)
            elif self._state == "tuplecount":
                tuplecount = int.from_bytes(self._read_buffer[:self._to_read], "big")
                if tuplecount == 1:
                    self._set_state("datasize", 4)
                elif tuplecount == 65535:
                    self._set_state("finished", 0)
                    break
                elif self._data_size != -1:
                    raise RuntimeError("More than one tuple")
                else:
                    raise RuntimeError(f"Invalid tuple count {tuplecount!r}")
            elif self._state == "datasize":
                self._data_size = int.from_bytes(self._read_buffer[:self._to_read], "big")
                self._set_state("data", 0)
            else:
                raise RuntimeError("Invalid state {state!r}")
        return data_bytes_read


class FileOut(RawIOBase):
    def __init__(self, path, source_f):
        self.source_f = source_f
        size = source_f.seek(0, 2)
        source_f.seek(0)
        encoded_path = path.encode("utf-8")
        self.header_f = BytesIO(
            SIGNATURE
            + b"\x00\x00\x00\x00"  # flags
            b"\x00\x00\x00\x00"  # header extension
            b"\x00\x03"  # num fields in tuple
            + len(encoded_path).to_bytes(4, "big")
            + encoded_path
            + b"\x00\x00\x00\x04"  # size of INTEGER for "size" field
            + size.to_bytes(4, "big")
            + size.to_bytes(4, "big")  # size of binary file field
        )
        self.footer_f = BytesIO(b"\xff\xff")

    def readinto(self, buffer):
        count = self.header_f.readinto(buffer)
        if count > 0:
            return count
        count = self.source_f.readinto(buffer)
        if count > 0:
            return count
        count = self.footer_f.readinto(buffer)
        if count > 0:
            return count
        return 0


@implementer(IDBIOFileConnection)
@implementer(IStorageConnection)
class Connection:
    def __init__(self, sqlconn, storage):
        self._sqlconn = sqlconn
        self.dirty_files = {}
        self.changes = {}
        self.storage = storage
        self._changelog_cache = storage._changelog_cache
        self._relpath_cache = storage._relpath_cache
        threadlog.debug("Using use_copy=%r", storage.use_copy)
        if storage.use_copy:
            self.io_file_open = self._copy_io_file_open
            self.io_file_get = self._copy_io_file_get
            self._file_write = self._copy_file_write
        else:
            self.io_file_open = self._select_io_file_open
            self.io_file_get = self._select_io_file_get
            self._file_write = self._insert_file_write

    def close(self):
        if hasattr(self, "_sqlconn"):
            self._sqlconn.close()
            del self._sqlconn
        if hasattr(self, "storage"):
            del self.storage

    def begin(self):
        self._sqlconn.run("START TRANSACTION")
        return self._sqlconn

    def commit(self):
        self._sqlconn.run("COMMIT")

    def rollback(self):
        self._sqlconn.run("ROLLBACK")

    def fetchone(self, q, **kw):
        row = self._sqlconn.run(q, **kw)
        if not row:
            return None
        (res,) = row
        return res

    def fetchscalar(self, q, **kw):
        row = self._sqlconn.run(q, **kw)
        if not row:
            return None
        ((res,),) = row
        return res

    @cached_property
    def last_changelog_serial(self):
        return self.db_read_last_changelog_serial()

    def db_read_last_changelog_serial(self):
        q = 'SELECT MAX(serial) FROM changelog'
        res = self.fetchscalar(q)
        return -1 if res is None else res

    def db_read_typedkey(self, relpath):
        q = "SELECT keyname, serial FROM kv WHERE key = :relpath"
        res = self.fetchone(q, relpath=relpath)
        if res is None:
            raise KeyError(relpath)
        (keyname, serial) = res
        return (keyname, serial)

    def _get_relpath_at(self, relpath, serial) -> KeyData:
        (keyname, last_serial) = self.db_read_typedkey(relpath)
        serials_and_values = self._iter_serial_and_value_backwards(relpath, last_serial)
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
        raise KeyError(relpath)

    def _iter_serial_and_value_backwards(
        self, relpath, last_serial
    ) -> Iterator[KeyData]:
        while last_serial >= 0:
            tup = self.get_changes(last_serial).get(relpath)
            if tup is None:
                raise RuntimeError("no transaction entry at %s" % (last_serial))
            keyname, back_serial, val = tup
            yield KeyData(
                relpath=relpath,
                keyname=keyname,
                serial=last_serial,
                back_serial=back_serial,
                value=val,
            )
            last_serial = back_serial

        # we could not find any change below at_serial which means
        # the key didn't exist at that point in time

    def get_relpath_at(self, relpath, serial) -> KeyData:
        result = self._relpath_cache.get((serial, relpath), absent)
        if result is absent:
            result = self._changelog_cache.get((serial, relpath), absent)
        if result is absent:
            changes = self._changelog_cache.get(serial, absent)
            if changes is not absent and relpath in changes:
                (keyname, back_serial, value) = changes[relpath]
                result = KeyData(
                    relpath=relpath,
                    keyname=keyname,
                    serial=serial,
                    back_serial=back_serial,
                    value=value,
                )
        if result is absent:
            result = self._get_relpath_at(relpath, serial)
        if gettotalsizeof(result.value, maxlen=100000) is None:
            # result is big, put it in the changelog cache,
            # which has fewer entries to preserve memory
            self._changelog_cache.put((serial, relpath), result)
        else:
            # result is small
            self._relpath_cache.put((serial, relpath), result)
        return result

    def iter_relpaths_at(self, typedkeys, at_serial):
        keynames = frozenset(k.key_name for k in typedkeys)
        keyname_id_values = {"keynameid%i" % i: k for i, k in enumerate(keynames)}
        q = """
            SELECT key, keyname, serial
            FROM kv
            WHERE serial=:serial AND keyname IN (:keynames)
        """
        q = q.replace(':keynames', ", ".join(':' + x for x in keyname_id_values))
        for serial in range(at_serial, -1, -1):
            rows = self._sqlconn.run(q, serial=serial, **keyname_id_values)
            if not rows:
                continue
            changes = self.get_changes(serial)
            for relpath, keyname, serial in rows:
                (keyname, back_serial, val) = changes[relpath]
                yield KeyData(
                    relpath=relpath,
                    keyname=keyname,
                    serial=serial,
                    back_serial=back_serial,
                    value=val,
                )

    def write_changelog_entry(self, serial, entry):
        threadlog.debug("writing changelog for serial %s", serial)
        q = """
            INSERT INTO changelog (serial, data) VALUES (:serial, :data);"""
        data = dumps(entry)
        self._sqlconn.run(q, serial=serial, data=pg8000.Binary(data))

    def io_file_os_path(self, path):  # noqa: ARG002
        return None

    def io_file_exists(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is not absent:
            return f is not None
        q = "SELECT path FROM files WHERE path = :path"
        return bool(self.fetchscalar(q, path=path.relpath))

    def io_file_set(self, path, content_or_file):
        assert not os.path.isabs(path.relpath)
        assert not path.relpath.endswith("-tmp")
        f = self.dirty_files.get(path.relpath, None)
        if f is None:
            f = SpooledTemporaryFile(max_size=1048576)
        if not isinstance(content_or_file, bytes) and not callable(getattr(content_or_file, "seekable", None)):
            content_or_file = content_or_file.read()
            if len(content_or_file) > 1048576:
                threadlog.warn(
                    "Read %.1f megabytes into memory in postgresql io_file_set for %s, because of unseekable file",
                    len(content_or_file) / 1048576,
                    path.relpath,
                )
        if isinstance(content_or_file, bytes):
            f.write(content_or_file)
            f.seek(0)
        else:
            content_or_file.seek(0)
            shutil.copyfileobj(content_or_file, f)
        self.dirty_files[path.relpath] = f

    def io_file_new_open(self, path):  # noqa: ARG002
        return SpooledTemporaryFile(max_size=1048576)

    def _copy_io_file_open(self, path):
        dirty_file = self.dirty_files.get(path.relpath, absent)
        if dirty_file is None:
            raise OSError
        f = SpooledTemporaryFile()
        if dirty_file is not absent:
            # we need a new file to prevent the dirty_file from being closed
            dirty_file.seek(0)
            shutil.copyfileobj(dirty_file, f)
            dirty_file.seek(0)
            f.seek(0)
            return f
        q = f"""
            COPY (
                SELECT data FROM files WHERE path = {pg8000.native.literal(path.relpath)})
            TO STDOUT WITH (FORMAT binary);"""  # noqa: S608 - we are escaping with literal
        stream = FileIn(f)
        self._sqlconn.run(
            q, stream=stream)
        if stream.got_data:
            f.seek(0)
            return f
        f.close()
        raise OSError(f"File not found at '{path.relpath}'")

    def _copy_io_file_get(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise OSError
        if f is not absent:
            pos = f.tell()
            f.seek(0)
            content = f.read()
            f.seek(pos)
            return content
        with self._copy_io_file_open(path) as f:
            res = f.read()
            if len(res) > 1048576:
                threadlog.warn(
                    "Read %.1f megabytes into memory in postgresql io_file_get for %s",
                    len(res) / 1048576,
                    path.relpath,
                )
            return res

    def _select_io_file_open(self, path):
        dirty_file = self.dirty_files.get(path.relpath, absent)
        if dirty_file is None:
            raise OSError
        if dirty_file is absent:
            return BytesIO(self._select_io_file_get(path))
        f = SpooledTemporaryFile()
        # we need a new file to prevent the dirty_file from being closed
        dirty_file.seek(0)
        shutil.copyfileobj(dirty_file, f)
        dirty_file.seek(0)
        f.seek(0)
        return f

    def _select_io_file_get(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise OSError
        if f is not absent:
            pos = f.tell()
            f.seek(0)
            content = f.read()
            f.seek(pos)
            return content
        q = "SELECT data FROM files WHERE path = :path"
        res = self.fetchscalar(q, path=path.relpath)
        if res is None:
            raise OSError
        return res

    def io_file_size(self, path):
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.get(path.relpath, absent)
        if f is None:
            raise OSError
        if f is not absent:
            pos = f.tell()
            size = f.seek(0, 2)
            f.seek(pos)
            return size
        q = "SELECT size FROM files WHERE path = :path"
        return self.fetchscalar(q, path=path.relpath)

    def io_file_delete(self, path, *, is_last_of_hash):  # noqa: ARG002
        assert not os.path.isabs(path.relpath)
        f = self.dirty_files.pop(path.relpath, None)
        if f is not None:
            f.close()
        self.dirty_files[path.relpath] = None

    def get_raw_changelog_entry(self, serial):
        # because a sequence is used for the next serial, there might be
        # missing serials in the changelog table if there was a conflict
        # during commit
        # this query makes sure we return an empty changelog for missing
        # serials, but only if the serial wasn't used yet by comparing with
        # max(serial) of changelog
        q = r"""
            SELECT
                COALESCE(
                    data,
                    'JK\000\000\000\000@\000\000\000\002Q'::BYTEA) AS data
            FROM changelog
            RIGHT OUTER JOIN (
                SELECT
                    :serial::BIGINT AS serial
                WHERE (
                    :serial::BIGINT <= (SELECT max(serial) FROM changelog))
                ) AS serial
            ON changelog.serial=serial.serial;"""
        return self.fetchscalar(q, serial=serial)

    def get_rel_renames(self, serial):
        if serial == -1:
            return None
        data = self.get_raw_changelog_entry(serial)
        (changes, rel_renames) = loads(data)
        return rel_renames

    def get_changes(self, serial):
        changes = self._changelog_cache.get(serial, absent)
        if changes is absent:
            data = self.get_raw_changelog_entry(serial)
            changes, rel_renames = loads(data)
            # make values in changes read only so no calling site accidentally
            # modifies data
            changes = ensure_deeply_readonly(changes)
            assert isinstance(changes, ReadonlyView)
            self._changelog_cache.put(serial, changes)
        return changes

    def write_transaction(self, io_file):
        return Writer(self.storage, self, io_file)

    def _copy_file_write(self, path, f):
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        q = """
            COPY files (path, size, data) FROM STDIN WITH (FORMAT binary);"""
        f.seek(0)
        self._sqlconn.run(
            q, stream=FileOut(path, f))
        f.close()

    def _insert_file_write(self, path, f):
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        q = """
            INSERT INTO files(path, size, data)
                VALUES (:path, :size, :data);"""
        f.seek(0)
        content = f.read()
        f.close()
        self._sqlconn.run(
            q, path=path, size=len(content), data=pg8000.Binary(content))

    def _file_delete(self, path):
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        q = "DELETE FROM files WHERE path = :path"
        self._sqlconn.run(q, path=path)

    def _lock(self):
        q = 'SELECT pg_advisory_xact_lock(1);'
        self._sqlconn.run(q)

    def _write_dirty_files(self):
        for path, f in self.dirty_files.items():
            if f is None:
                self._file_delete(path)
            else:
                # delete first to avoid conflict
                self._file_delete(path)
                self._file_write(path, f)
        self.dirty_files.clear()

    def commit_files_without_increasing_serial(self):
        self.begin()
        try:
            self._lock()
            self._write_dirty_files()
        except BaseException:
            self.rollback()
            raise
        else:
            self.commit()


def as_bool(value):
    if isinstance(value, bool):
        return value
    if not hasattr(value, "lower"):
        raise ValueError("Unknown boolean value %r." % value)
    if value.lower() in ["false", "no"]:
        return False
    if value.lower() in ["true", "yes"]:
        return True
    raise ValueError("Unknown boolean value '%s'." % value)


@implementer(IStorage)
class Storage:
    SSL_OPT_KEYS = ("ssl_check_hostname", "ssl_ca_certs", "ssl_certfile", "ssl_keyfile")
    database = "devpi"
    host = "localhost"
    port = "5432"
    unix_sock = None
    user = "devpi"
    password = None
    use_copy = True
    ssl_context = None
    expected_schema = dict(
        index=dict(
            kv_serial_idx="""
                CREATE INDEX kv_serial_idx ON kv (serial);
            """),
        sequence=dict(
            changelog_serial_seq="""
                CREATE SEQUENCE changelog_serial_seq
                AS BIGINT
                MINVALUE 0
                START :startserial;
            """),
        table=dict(
            changelog="""
                CREATE TABLE changelog (
                    serial INTEGER PRIMARY KEY,
                    data BYTEA NOT NULL
                )
            """,
            kv="""
                CREATE TABLE kv (
                    key TEXT NOT NULL PRIMARY KEY,
                    keyname TEXT,
                    serial INTEGER
                )
            """,
            files="""
                CREATE TABLE files (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    data BYTEA NOT NULL
                )
            """))

    def __init__(self, basedir, *, notify_on_commit, cache_size, settings):
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

        self.use_copy = as_bool(settings.get("use_copy", os.environ.get(
            "DEVPI_PG_USE_COPY", self.use_copy)))
        self.basedir = basedir
        self._notify_on_commit = notify_on_commit
        if gettotalsizeof(0) is None:
            # old devpi_server version doesn't have a working gettotalsizeof
            changelog_cache_size = cache_size
        else:
            changelog_cache_size = max(1, cache_size // 20)
        relpath_cache_size = max(1, cache_size - changelog_cache_size)
        self._changelog_cache = LRUCache(changelog_cache_size)  # is thread safe
        self._relpath_cache = LRUCache(relpath_cache_size)  # is thread safe
        self.last_commit_timestamp = time.time()
        self.ensure_tables_exist()

    @classmethod
    def exists(cls, basedir, settings):  # noqa: ARG003
        return True

    def perform_crash_recovery(self):
        pass

    def register_key(self, key):
        pass

    def get_connection(self, *, closing=True, write=False, timeout=30):  # noqa: ARG002
        sqlconn = pg8000.native.Connection(
            user=self.user,
            database=self.database,
            host=self.host,
            port=int(self.port),
            unix_sock=self.unix_sock,
            password=self.password,
            ssl_context=self.ssl_context,
            timeout=timeout,
        )
        sqlconn.text_factory = bytes
        conn = Connection(sqlconn, self)
        if closing:
            return contextlib.closing(conn)
        return conn

    def _reflect_schema(self):
        result = {}
        with self.get_connection() as conn:
            sqlconn = conn.begin()
            rows = sqlconn.run("""
                SELECT tablename FROM pg_tables WHERE schemaname='public';""")
            for row in rows:
                result.setdefault("table", {})[row[0]] = ""
            rows = sqlconn.run("""
                SELECT indexname FROM pg_indexes WHERE schemaname='public';""")
            for row in rows:
                result.setdefault("index", {})[row[0]] = ""
            rows = sqlconn.run("""
                SELECT sequencename FROM pg_sequences WHERE schemaname='public';""")
            for row in rows:
                result.setdefault("sequence", {})[row[0]] = ""
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
        with self.get_connection() as conn:
            sqlconn = conn.begin()
            if not schema:
                threadlog.info("DB: Creating schema")
            else:
                threadlog.info("DB: Updating schema")
            if "changelog" not in missing.get("table", {}):
                kw = dict(startserial=conn.db_read_last_changelog_serial() + 1)
            else:
                kw = dict(startserial=0)
            for kind in ('table', 'index', 'sequence'):
                objs = missing.pop(kind, {})
                for name in list(objs):
                    q = objs.pop(name)
                    for k, v in kw.items():
                        q = q.replace(f":{k}", pg8000.native.literal(v))
                    sqlconn.run(q)
                assert not objs
            conn.commit()
        assert not missing


@devpiserver_hookimpl
def devpiserver_describe_storage_backend(settings):
    return StorageInfo(
        name="pg8000",
        description="Postgresql backend",
        exists=Storage.exists,
        storage_cls=Storage,
        connection_cls=Connection,
        writer_cls=Writer,
        storage_factory=Storage,
        settings=settings,
    )


@implementer(IWriter)
class Writer:
    def __init__(self, storage, conn, io_file):
        self.conn = conn
        self.io_file = io_file
        self.storage = storage
        self.rel_renames = []

    def record_set(self, typedkey, value=None, back_serial=None):
        """ record setting typedkey to value (None means it's deleted) """
        assert not isinstance(value, ReadonlyView), value
        # at __exit__ time we write out changes to the _changelog_cache
        # so we protect here against the caller modifying the value later
        value = get_mutable_deepcopy(value)
        self.changes[typedkey.relpath] = (typedkey.key_name, back_serial, value)

    def records_set(self, records) -> None:
        for record in records:
            self.record_set(record.key, record.value, record.back_serial)

    def set_rel_renames(self, rel_renames):
        self.rel_renames = rel_renames

    def _db_insert_typedkey(self, relpath, name, serial):
        q = """
            INSERT INTO kv(key, keyname, serial)
                VALUES (:relpath, :name, :serial);"""
        self.conn._sqlconn.run(q, relpath=relpath, name=name, serial=serial)

    def _db_update_typedkey(self, relpath, name, serial, back_serial):
        q = """
            UPDATE kv SET serial = :serial
            WHERE
                key = :relpath
                AND keyname = :name
                AND serial = :back_serial;"""
        self.conn._sqlconn.run(
            q, relpath=relpath, name=name, serial=serial, back_serial=back_serial
        )

    def __enter__(self):
        self.conn.begin()
        self.conn._lock()
        q = """SELECT nextval('changelog_serial_seq');"""
        self.commit_serial = self.conn.fetchscalar(q)
        self.log = thread_push_log("fswriter%s:" % self.commit_serial)
        self.changes = {}
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
                self.log.info("roll back in %s", commit_serial)
            del self.conn
            del self.storage
            del self.log
        except BaseException:
            self.conn.rollback()
            raise
        finally:
            thread_pop_log("fswriter%s:" % commit_serial)

    def commit(self, commit_serial):
        self.conn._write_dirty_files()
        for relpath, (keyname, back_serial, _value) in self.changes.items():
            if back_serial is None:
                raise RuntimeError
            if back_serial == -1:
                self._db_insert_typedkey(relpath, keyname, commit_serial)
            else:
                self._db_update_typedkey(relpath, keyname, commit_serial, back_serial)
        entry = (self.changes, self.rel_renames)
        self.conn.write_changelog_entry(commit_serial, entry)
        self.conn.commit()
        self.storage.last_commit_timestamp = time.time()
        return LazyChangesFormatter(list(self.changes.keys()), (), ())

    def rollback(self):
        self.conn.rollback()
