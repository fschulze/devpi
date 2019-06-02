from .config import hookimpl
from .keyfs_sqlite_fs import Connection as BaseConnection
from .keyfs_sqlite_fs import Storage as BaseStorage
from .log import threadlog
from .readonly import ReadonlyView
from .readonly import ensure_deeply_readonly
from .fileutil import dumps, loads
from devpi_common.types import cached_property
import sqlite3


notset = object()


class Connection(BaseConnection):
    def _dump(self, relpath=None):
        # for debugging
        rows = self.fetchall("SELECT ROWID, * FROM relpath_info")
        if relpath:
            rows = [x for x in rows if x[0] == relpath]
        print("relpath_info:")
        print("\n".join(repr(row) for row in rows))
        rows = self.fetchall("SELECT * FROM relpath_deleted_at")
        if relpath:
            rows = [x for x in rows if x[0] == relpath]
        print("relpath_deleted_at:")
        print("\n".join(repr(row) for row in rows))
        rows = self.fetchall("SELECT * FROM kvchangelog")
        if relpath:
            rows = [x for x in rows if x[0] == relpath]
        print("kvchangelog:")
        print("\n".join(repr(row) for row in rows))

    def _explain(self, query, *args):
        # for debugging
        c = self._sqlconn.cursor()
        r = c.execute("EXPLAIN " + query, *args)
        return r.fetchall()

    def _explain_query_plan(self, query, *args):
        # for debugging
        c = self._sqlconn.cursor()
        r = c.execute("EXPLAIN QUERY PLAN" + query, *args)
        return r.fetchall()

    def fetchall(self, query, *args):
        c = self._sqlconn.cursor()
        r = c.execute(query, *args)
        return r.fetchall()

    def db_read_last_changelog_serial(self):
        q = 'SELECT max(serial) FROM "renames" LIMIT 1'
        res = self._sqlconn.execute(q).fetchone()[0]
        return -1 if res is None else res

    def db_read_typedkey(self, relpath):
        q = "SELECT ROWID, keyname, serial FROM relpath_info WHERE relpath = ?"
        c = self._sqlconn.cursor()
        row = c.execute(q, (relpath,)).fetchone()
        if row is None:
            raise KeyError(relpath)
        self._relpath_id_cache[relpath] = row[0]
        return tuple(row[1:])

    @cached_property
    def _relpath_id_cache(self):
        return {}

    def db_write_typedkey(self, relpath, name, next_serial):
        q = """
            INSERT OR REPLACE INTO relpath_info (
                ROWID, relpath, keyname, serial)
            VALUES (
                (
                    SELECT ROWID
                    FROM relpath_info
                    WHERE relpath=:relpath
                    UNION
                        SELECT max(ROWID) + 1
                        FROM relpath_info
                        LIMIT 1),
                :relpath,
                :keyname,
                :serial)"""
        c = self._sqlconn.execute(q, dict(
            relpath=relpath, keyname=name, serial=next_serial))
        self._relpath_id_cache[relpath] = c.lastrowid

    def write_changelog_entry(self, serial, entry):
        (changes, renames) = entry
        threadlog.debug("writing changelog for serial %s", serial)
        kvchangelog = []
        for relpath, (keyname, back_serial, value) in changes.items():
            tkey = self.storage.typed_keys[keyname]
            relpath_id = self._relpath_id_cache[relpath]
            if value is None:
                q = "INSERT INTO relpath_deleted_at (relpath_id, serial) VALUES (?, ?)"
                self._sqlconn.execute(q, (relpath_id, serial))
                items = ((b'', None),)
            elif tkey.type == dict:
                old_value = None
                if back_serial > -1:
                    (_, old_value) = self.get_relpath_at(relpath, back_serial)
                if old_value is None:
                    old_value = dict()
                items = []
                all_keys = set(old_value).union(value)
                for k in all_keys:
                    if k not in value:
                        items.append((sqlite3.Binary(dumps(k)), None))
                    else:
                        v = value[k]
                        if k not in old_value or old_value[k] != v:
                            items.append((
                                sqlite3.Binary(dumps(k)),
                                sqlite3.Binary(dumps(v))))
                if not items:
                    items = ((b'', b''),)
            elif tkey.type == set:
                old_value = None
                if back_serial > -1:
                    (_, old_value) = self.get_relpath_at(relpath, back_serial)
                if old_value is None:
                    old_value = set()
                items = []
                all_keys = set(old_value).union(value)
                for k in all_keys:
                    if k not in value:
                        items.append((sqlite3.Binary(dumps(k)), None))
                    elif k not in old_value:
                        items.append((sqlite3.Binary(dumps(k)), b''))
                if not items:
                    items = ((b'', b''),)
            else:
                items = ((b'', sqlite3.Binary(dumps(value))),)
            for key, data in items:
                kvchangelog.append((relpath_id, serial, back_serial, key, data))
        q = """
            INSERT INTO kvchangelog (
                relpath_id, serial, back_serial, key, value)
            VALUES (?, ?, ?, ?, ?)"""
        self._sqlconn.executemany(q, kvchangelog)
        self._sqlconn.execute(
            "INSERT INTO renames (serial, data) VALUES (?, ?)",
            (serial, sqlite3.Binary(dumps(renames))))

    def get_raw_changelog_entry(self, serial):
        q = "SELECT data FROM renames WHERE serial = ?"
        row = self._sqlconn.execute(q, (serial,)).fetchone()
        if row is None:
            return None
        renames = loads(bytes(row[0]))
        changes = self._get_changes_at(serial)
        return dumps((changes, renames))

    def _process_rows(self, rows, serial):
        changes = {}
        for relpath, keyname, key_serial, back_serial, deleted_serial, k, v in rows:
            tkey = self.storage.typed_keys[keyname]
            if relpath not in changes:
                changes[relpath] = dict(
                    keyname=[], key_serial=set(), serial_key=[], back_serial=[], items=[], deleted=set(), empty=False)
            key = changes[relpath]
            if key_serial == deleted_serial and serial != deleted_serial:
                continue
            key['key_serial'].add(key_serial)
            key['keyname'].append(keyname)
            key['back_serial'].append(back_serial)
            if serial == deleted_serial:
                key['items'] = None
            elif tkey.type == dict:
                if k is None:
                    raise RuntimeError("A key can't be None")
                if k == b'' and v is None:
                    key['empty'] = True
                    continue
                if k == b'' and v == b'':
                    # empty value as placeholder when the value didn't change
                    continue
                k = loads(bytes(k))
                if v is None:
                    key['deleted'].add(k)
                    continue
                if k not in key['deleted']:
                    key['items'].append((k, loads(bytes(v))))
                    key['serial_key'].append((serial, k))
            elif tkey.type == set:
                if k is None:
                    raise RuntimeError("A key can't be None")
                if k == b'' and v is None:
                    key['empty'] = True
                    continue
                if k == b'' and v == b'':
                    # empty value as placeholder when the value didn't change
                    continue
                k = loads(bytes(k))
                if v is None:
                    key['deleted'].add(k)
                    continue
                if v != b'':
                    raise RuntimeError("An item in a set can't have a value")
                elif k not in key['deleted']:
                    key['items'].append(k)
                    key['serial_key'].append((serial, k))
            else:
                if k != b'':
                    raise RuntimeError("A plain value can't have a key")
                key['items'].append(loads(bytes(v)))
                key['serial_key'].append((serial, k))
        result = {}
        last_serials = {}
        for relpath, key in changes.items():
            if not key['keyname']:
                # deleted
                continue
            if len(key['serial_key']) != len(set(key['serial_key'])):
                raise RuntimeError
            (keyname,) = set(key['keyname'])
            tkey = self.storage.typed_keys[keyname]
            back_serial = max(key['back_serial'])
            last_serial = max(key['key_serial'])
            items = key['items']
            if items is None:
                value = None
            elif tkey.type == dict:
                if key['empty']:
                    value = dict()
                else:
                    value = dict(items)
            elif tkey.type == set:
                if key['empty']:
                    value = set()
                else:
                    value = set(items)
            else:
                (value,) = items
            result[relpath] = (keyname, back_serial, value)
            last_serials[relpath] = last_serial
        return (result, last_serials)

    def _get_relpath_at(self, relpath, serial):
        q = """
            WITH
                filtered_deleted_at AS (
                    SELECT
                        relpath_info.ROWID,
                        coalesce(max(relpath_deleted_at.serial), -1) AS serial
                    FROM relpath_info
                    LEFT OUTER JOIN relpath_deleted_at ON
                        relpath_deleted_at.relpath_id=relpath_info.ROWID
                        AND relpath_deleted_at.serial<=:serial
                    WHERE
                        relpath_info.relpath=:relpath
                    GROUP BY relpath_info.relpath),
                latest_relpath_keys AS (
                    SELECT
                        relpath_info.ROWID,
                        relpath_info.relpath,
                        relpath_info.keyname,
                        key,
                        coalesce(filtered_deleted_at.serial, -1) AS deleted_serial,
                        max(kvchangelog.serial) AS latest_serial
                    FROM relpath_info
                    JOIN kvchangelog ON
                        kvchangelog.relpath_id=relpath_info.ROWID
                    LEFT OUTER JOIN filtered_deleted_at ON
                        filtered_deleted_at.ROWID=relpath_info.ROWID
                    WHERE
                        kvchangelog.serial>=filtered_deleted_at.serial
                        AND kvchangelog.serial<=:serial
                        AND relpath_info.relpath=:relpath
                    GROUP BY
                        relpath_info.ROWID,
                        relpath_info.relpath,
                        relpath_info.keyname,
                        key,
                        filtered_deleted_at.serial)
            SELECT
                latest_relpath_keys.relpath,
                latest_relpath_keys.keyname,
                kvchangelog.serial,
                back_serial,
                deleted_serial,
                latest_relpath_keys.key,
                value
            FROM latest_relpath_keys
            JOIN kvchangelog ON
                latest_relpath_keys.ROWID=kvchangelog.relpath_id
                AND latest_relpath_keys.key=kvchangelog.key
                AND latest_relpath_keys.latest_serial=kvchangelog.serial
            ORDER BY latest_relpath_keys.ROWID, kvchangelog.serial DESC"""
        rows = self.fetchall(q, dict(relpath=relpath, serial=serial))
        (result, last_serials) = self._process_rows(rows, serial)
        return (last_serials[relpath], result[relpath])

    def get_relpath_at(self, relpath, serial):
        result = self._changelog_cache.get((serial, relpath), notset)
        if result is notset:
            changes = self._changelog_cache.get(serial, notset)
            if changes is not notset and relpath in changes:
                result = (serial, changes[relpath][2])
        if result is notset:
            (last_serial, tup) = self._get_relpath_at(relpath, serial)
            result = (last_serial, ensure_deeply_readonly(tup[2]))
        self._changelog_cache.put((serial, relpath), result)
        return result

    def _get_changes_at(self, serial):
        q = """
            WITH
                changed_relpaths AS (
                    SELECT DISTINCT
                        relpath_info.ROWID,
                        relpath_info.relpath,
                        relpath_info.keyname
                    FROM relpath_info
                    JOIN kvchangelog ON
                        relpath_info.ROWID=kvchangelog.relpath_id
                    WHERE kvchangelog.serial=:serial),
                filtered_deleted_at AS (
                    SELECT
                        changed_relpaths.ROWID,
                        coalesce(max(serial), -1) AS serial
                    FROM changed_relpaths
                    LEFT OUTER JOIN relpath_deleted_at ON
                        relpath_deleted_at.relpath_id=changed_relpaths.ROWID
                        AND relpath_deleted_at.serial<=:serial
                    GROUP BY changed_relpaths.ROWID),
                latest_relpath_keys AS (
                    SELECT
                        changed_relpaths.ROWID,
                        changed_relpaths.relpath,
                        changed_relpaths.keyname,
                        key,
                        filtered_deleted_at.serial AS deleted_serial,
                        max(kvchangelog.serial) AS latest_serial
                    FROM changed_relpaths
                    JOIN kvchangelog ON
                        kvchangelog.relpath_id=changed_relpaths.ROWID
                    JOIN filtered_deleted_at ON
                        filtered_deleted_at.ROWID=changed_relpaths.ROWID
                    WHERE
                        kvchangelog.serial>=filtered_deleted_at.serial
                        AND kvchangelog.serial<=:serial
                    GROUP BY
                        changed_relpaths.ROWID,
                        changed_relpaths.relpath,
                        changed_relpaths.keyname,
                        key,
                        filtered_deleted_at.serial)
            SELECT
                latest_relpath_keys.relpath,
                latest_relpath_keys.keyname,
                kvchangelog.serial,
                back_serial,
                deleted_serial,
                latest_relpath_keys.key,
                value
            FROM latest_relpath_keys
            JOIN kvchangelog ON
                latest_relpath_keys.ROWID=kvchangelog.relpath_id
                AND latest_relpath_keys.key=kvchangelog.key
                AND latest_relpath_keys.latest_serial=kvchangelog.serial
            ORDER BY latest_relpath_keys.ROWID, kvchangelog.serial DESC"""
        rows = self.fetchall(q, dict(serial=serial))
        (result, last_serials) = self._process_rows(rows, serial)
        for relpath, tup in result.items():
            last_serial = last_serials[relpath]
            (_last_serial, _tup) = self._get_relpath_at(relpath, serial)
            if (last_serial, tup) != (_last_serial, _tup):
                raise RuntimeError
        return result

    def get_changes(self, serial):
        changes = self._changelog_cache.get(serial)
        if changes is None:
            # make values in changes read only so no calling site accidentally
            # modifies data
            changes = self._get_changes_at(serial)
            changes = ensure_deeply_readonly(changes)
            assert isinstance(changes, ReadonlyView)
            self._changelog_cache.put(serial, changes)
        return changes


class Storage(BaseStorage):
    Connection = Connection
    db_filename = ".sqlite2"

    def add_key(self, key):
        if not hasattr(self, 'typed_keys'):
            self.typed_keys = {}
        self.typed_keys[key.name] = key

    def ensure_tables_exist(self):
        if self.sqlpath.exists():
            return
        with self.get_connection(write=True) as conn:
            threadlog.info("DB: Creating schema")
            c = conn._sqlconn.cursor()
            c.execute("""
                CREATE TABLE relpath_info (
                    relpath TEXT NOT NULL,
                    keyname TEXT NOT NULL,
                    serial INTEGER NOT NULL
                )
            """)
            c.execute("""
                CREATE UNIQUE INDEX relpath_info_relpath_idx ON relpath_info (relpath);
            """)
            c.execute("""
                CREATE TABLE relpath_deleted_at (
                    relpath_id INTEGER NOT NULL,
                    serial INTEGER NOT NULL,
                    FOREIGN KEY(relpath_id) REFERENCES relpath_info(ROWID)
                )
            """)
            c.execute("""
                CREATE INDEX relpath_deleted_at_relpath_id_idx ON relpath_deleted_at (relpath_id);
            """)
            c.execute("""
                CREATE TABLE kvchangelog (
                    relpath_id INTEGER NOT NULL,
                    serial INTEGER NOT NULL,
                    back_serial INTEGER NOT NULL,
                    key BLOB NOT NULL,
                    value BLOB,
                    FOREIGN KEY(relpath_id) REFERENCES relpath_info(ROWID)
                )
            """)
            c.execute("""
                CREATE INDEX kvchangelog_relpath_id_idx ON kvchangelog (relpath_id);
            """)
            c.execute("""
                CREATE TABLE renames (
                    serial INTEGER NOT NULL PRIMARY KEY,
                    data BLOB NOT NULL
                )
            """)
            conn.commit()


@hookimpl
def devpiserver_storage_backend(settings):
    return dict(
        storage=Storage,
        name="sqlite2",
        description="New SQLite backend with files on the filesystem",
        _test_markers=["storage_with_filesystem"])
