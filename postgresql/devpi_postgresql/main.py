from devpi_common.types import cached_property
try:
    from devpi_server import interfaces as ds_interfaces
except ImportError:
    ds_interfaces = None
from devpi_server.fileutil import dumps, loads
try:
    from devpi_server.keyfs import RelpathInfo
    from devpi_server.keyfs import get_relpath_at
except ImportError:
    pass
from devpi_server.log import threadlog, thread_push_log, thread_pop_log
from devpi_server.readonly import ReadonlyView
from devpi_server.readonly import ensure_deeply_readonly, get_mutable_deepcopy
from functools import partial
from io import BytesIO
from io import RawIOBase
from pluggy import HookimplMarker
from repoze.lru import LRUCache
from tempfile import TemporaryFile
from zope.interface import Interface
from zope.interface import implementer
import contextlib
import os
import pg8000.native
import time
from devpi_server.model import ensure_boolean
import ssl


for name in ('IStorageConnection3', 'IStorageConnection2', 'IStorageConnection'):
    IStorageConnection3 = getattr(ds_interfaces, name, Interface)
    if IStorageConnection3 is not Interface:
        break


absent = object()


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
                    raise RuntimeError("Invalid tuple count {tuplecount!r}")
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


@implementer(IStorageConnection3)
class Connection:
    def __init__(self, sqlconn, storage):
        self._sqlconn = sqlconn
        self.dirty_files = {}
        self.changes = {}
        self.storage = storage
        self._changelog_cache = storage._changelog_cache

    def close(self):
        self._sqlconn.close()
        del self._sqlconn
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

    def db_write_typedkey(self, relpath, name, next_serial):
        q = """
            INSERT INTO kv(key, keyname, serial)
                VALUES (:relpath, :name, :next_serial)
            ON CONFLICT (key) DO UPDATE
                SET keyname = EXCLUDED.keyname, serial = EXCLUDED.serial;"""
        self._sqlconn.run(q, relpath=relpath, name=name, next_serial=next_serial)

    def get_relpath_at(self, relpath, serial):
        result = self._changelog_cache.get((serial, relpath), absent)
        if result is absent:
            changes = self._changelog_cache.get(serial, absent)
            if changes is not absent and relpath in changes:
                (keyname, back_serial, value) = changes[relpath]
                result = (serial, back_serial, value)
        if result is absent:
            result = get_relpath_at(self, relpath, serial)
        self._changelog_cache.put((serial, relpath), result)
        return result

    def iter_relpaths_at(self, typedkeys, at_serial):
        keynames = frozenset(k.name for k in typedkeys)
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
            changes = self._changelog_cache.get(serial, absent)
            if changes is absent:
                changes = loads(
                    self.get_raw_changelog_entry(serial))[0]
            for relpath, keyname, serial in rows:
                (keyname, back_serial, val) = changes[relpath]
                yield RelpathInfo(
                    relpath=relpath, keyname=keyname,
                    serial=serial, back_serial=back_serial,
                    value=val)

    def write_changelog_entry(self, serial, entry):
        threadlog.debug("writing changelog for serial %s", serial)
        data = dumps(entry)
        self._sqlconn.run(
            "INSERT INTO changelog (serial, data) VALUES (:serial, :data)",
            serial=serial, data=pg8000.Binary(data))

    def io_file_os_path(self, path):
        return None

    def io_file_exists(self, path):
        assert not os.path.isabs(path)
        q = "SELECT path FROM files WHERE path = :path"
        return bool(self.fetchscalar(q, path=path))

    def io_file_set(self, path, content_or_file):
        assert not os.path.isabs(path)
        assert not path.endswith("-tmp")
        if not isinstance(content_or_file, bytes) and not content_or_file.seekable():
            content_or_file = content_or_file.read()
            if len(content_or_file) > 1048576:
                threadlog.warn(
                    "Read %.1f megabytes into memory in postgresql io_file_set for %s",
                    len(content_or_file) / 1048576, path)
        if isinstance(content_or_file, bytes):
            q = """
                INSERT INTO files(path, size, data)
                    VALUES (:path, :size, :data)
                ON CONFLICT (path) DO UPDATE
                    SET size = EXCLUDED.size, data = EXCLUDED.data;"""
            self._sqlconn.run(
                q, path=path, size=len(content_or_file),
                data=pg8000.Binary(content_or_file))
        else:
            self.io_file_delete(path)
            q = """
                COPY files (path, size, data) FROM STDIN WITH (FORMAT binary);"""
            self._sqlconn.run(
                q, stream=FileOut(path, content_or_file))
        self.dirty_files[path] = True

    def io_file_new_open(self, path):
        return TemporaryFile()

    def io_file_open(self, path):
        f = TemporaryFile()
        q = f"""
            COPY (
                SELECT data FROM files WHERE path = {pg8000.native.literal(path)})
            TO STDOUT WITH (FORMAT binary);"""
        stream = FileIn(f)
        self._sqlconn.run(
            q, stream=stream)
        if stream.got_data:
            f.seek(0)
            return f
        f.close()
        raise IOError(f"File not found at '{path}'")

    def io_file_get(self, path):
        assert not os.path.isabs(path)
        q = "SELECT data FROM files WHERE path = :path"
        res = self.fetchscalar(q, path=path)
        if res is None:
            raise IOError(f"File not found at '{path}'")
        if len(res) > 1048576:
            threadlog.warn(
                "Read %.1f megabytes into memory in postgresql io_file_get for %s",
                len(res) / 1048576, path)
        return res

    def io_file_size(self, path):
        assert not os.path.isabs(path)
        q = "SELECT size FROM files WHERE path = :path"
        return self.fetchscalar(q, path=path)

    def io_file_delete(self, path):
        assert not os.path.isabs(path)
        q = "DELETE FROM files WHERE path = :path"
        self._sqlconn.run(q, path=path)
        self.dirty_files[path] = None

    def get_raw_changelog_entry(self, serial):
        q = "SELECT data FROM changelog WHERE serial = :serial"
        return self.fetchscalar(q, serial=serial)

    def get_changes(self, serial):
        changes = self._changelog_cache.get(serial)
        if changes is None:
            data = self.get_raw_changelog_entry(serial)
            changes, rel_renames = loads(data)
            # make values in changes read only so no calling site accidentally
            # modifies data
            changes = ensure_deeply_readonly(changes)
            self.cache_commit_changes(serial, changes)
        return changes

    def cache_commit_changes(self, serial, changes):
        assert isinstance(changes, ReadonlyView)
        self._changelog_cache.put(serial, changes)

    def write_transaction(self):
        return Writer(self.storage, self)

    def commit_files_without_increasing_serial(self):
        self.commit()


class Storage:
    SSL_OPT_KEYS = ("ssl_check_hostname", "ssl_ca_certs", "ssl_certfile", "ssl_keyfile")
    database = "devpi"
    host = "localhost"
    port = "5432"
    unix_sock = None
    user = "devpi"
    password = None
    ssl_context = None
    expected_schema = dict(
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

    def __init__(self, basedir, notify_on_commit, cache_size, settings=None):
        if settings is None:
            settings = {}
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

        self.basedir = basedir
        self._notify_on_commit = notify_on_commit
        self._changelog_cache = LRUCache(cache_size)  # is thread safe
        self.last_commit_timestamp = time.time()
        self.ensure_tables_exist()
        with self.get_connection() as conn:
            # don't use cached last_changelog_serial here
            self.next_serial = conn.db_read_last_changelog_serial() + 1

    def perform_crash_recovery(self):
        pass

    def get_connection(self, closing=True, write=False):
        sqlconn = pg8000.native.Connection(
            user=self.user,
            database=self.database,
            host=self.host,
            port=int(self.port),
            unix_sock=self.unix_sock,
            password=self.password,
            ssl_context=self.ssl_context,
            timeout=60)
        sqlconn.text_factory = bytes
        conn = Connection(sqlconn, self)
        if write:
            q = 'SELECT pg_advisory_xact_lock(1);'
            conn._sqlconn.run(q)
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
            for kind in ('table', 'index'):
                objs = missing.pop(kind, {})
                for name in list(objs):
                    q = objs.pop(name)
                    sqlconn.run(q)
                assert not objs
            conn.commit()
        assert not missing


@devpiserver_hookimpl
def devpiserver_storage_backend(settings):
    return dict(
        storage=partial(Storage, settings=settings),
        name="pg8000",
        description="Postgresql backend")


class Writer:
    def __init__(self, storage, conn):
        self.conn = conn
        self.storage = storage
        self.changes = {}

    def record_set(self, typedkey, value=None, back_serial=None):
        """ record setting typedkey to value (None means it's deleted) """
        assert not isinstance(value, ReadonlyView), value
        if back_serial is None:
            try:
                _, back_serial = self.conn.db_read_typedkey(typedkey.relpath)
            except KeyError:
                back_serial = -1
        self.conn.db_write_typedkey(typedkey.relpath, typedkey.name,
                                    self.storage.next_serial)
        # at __exit__ time we write out changes to the _changelog_cache
        # so we protect here against the caller modifying the value later
        value = get_mutable_deepcopy(value)
        self.changes[typedkey.relpath] = (typedkey.name, back_serial, value)

    def __enter__(self):
        self.log = thread_push_log("fswriter%s:" % self.storage.next_serial)
        self.conn.begin()
        return self

    def __exit__(self, cls, val, tb):
        thread_pop_log("fswriter%s:" % self.storage.next_serial)
        if cls is None:
            entry = self.changes, []
            self.conn.write_changelog_entry(self.storage.next_serial, entry)
            self.conn.commit()
            commit_serial = self.storage.next_serial
            self.storage.next_serial += 1
            message = "committed: keys: %s"
            args = [",".join(map(repr, list(self.changes)))]
            self.log.info("commited at %s", commit_serial)
            self.log.debug(message, *args)

            self.storage._notify_on_commit(commit_serial)
        else:
            self.conn.rollback()
            self.log.info("roll back at %s", self.storage.next_serial)
        del self.conn
        del self.storage
