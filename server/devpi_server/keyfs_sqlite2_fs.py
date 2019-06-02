from .config import hookimpl
from .interfaces import IStorageConnection4
from .keyfs_sqlite_fs import Connection as BaseConnection
from .keyfs_sqlite_fs import Storage as BaseStorage
from .keyfs_types import RelpathInfo
from .log import threadlog
from .markers import absent
from .readonly import ensure_deeply_readonly
from .fileutil import dumps, loads
from devpi_common.types import cached_property
from zope.interface import implementer
import sqlite3


def _iter_data_from_rows(typed_keys, rows_iter, debug=False):
    def value_from_items(tkey_type, items):
        if items is None:
            return None
        if tkey_type in (dict, set):
            if empty:
                return tkey_type()
            else:
                return tkey_type(items)
        (value,) = items
        return value

    if debug:
        seen = set()
        serial_key = []
        rows = []
    prev_relpath = None
    keynames = set()
    key_serials = []
    back_serials = []
    deleted = set()
    empty = False
    items = []

    def debug_checks():
        if len(serial_key) != len(set(serial_key)):
            raise RuntimeError

    def debug_checks2():
        if key_serials[0] != max(key_serials):
            raise RuntimeError
        if back_serials[0] != max(back_serials):
            raise RuntimeError

    for row in rows_iter:
        if debug:
            rows.append(row)
        if prev_relpath is None:
            prev_relpath = row[0]
        elif prev_relpath != row[0]:
            if debug:
                seen.add(prev_relpath)
                debug_checks()
                serial_key = []
            # if there are no keynames, then the item was deleted,
            # so we don't yield data back
            if keynames:
                if debug:
                    debug_checks2()
                (keyname,) = keynames
                yield (prev_relpath, keyname, key_serials[0], back_serials[0], value_from_items(typed_keys[keyname].type, items))
            keynames.clear()
            key_serials.clear()
            back_serials.clear()
            deleted.clear()
            empty = False
            items = []
            prev_relpath = row[0]
            if debug:
                rows.clear()
        (relpath, keyname, key_serial, back_serial, deleted_serial, k, v) = row
        tkey_type = typed_keys[keyname].type
        if debug and relpath in seen:
            raise RuntimeError
        key_serials.append(key_serial)
        if key_serial == deleted_serial and key_serial != key_serials[0]:
            continue
        keynames.add(keyname)
        back_serials.append(back_serial)
        if key_serial == deleted_serial:
            items = None
        elif tkey_type in (dict, set):
            if k is None:
                raise RuntimeError("A key can't be None")
            if k == b'' and v is None:
                empty = True
                continue
            if k == b'' and v == b'':
                # empty value as placeholder when the value didn't change
                continue
            k = loads(k)
            if v is None:
                deleted.add(k)
                continue
            if tkey_type is set and v != b'':
                raise RuntimeError("An item in a set can't have a value")
            if k not in deleted:
                if tkey_type is dict:
                    items.append((k, loads(v)))
                elif tkey_type is set:
                    items.append(k)
                else:
                    raise RuntimeError
                if debug:
                    serial_key.append((key_serial, k))
        else:
            if k != b'':
                raise RuntimeError("A plain value can't have a key")
            items.append(loads(v))
            if debug:
                serial_key.append((key_serial, k))
    if prev_relpath is None:
        return
    if debug:
        debug_checks()
    # if there are no keynames, then the item was deleted,
    # so we don't yield data back
    if keynames:
        if debug:
            debug_checks2()
        (keyname,) = keynames
        yield (relpath, keyname, key_serials[0], back_serials[0], value_from_items(tkey_type, items))


@implementer(IStorageConnection4)
class Connection(BaseConnection):
    def _dump(self, relpath=None):
        # for debugging
        rows = self.fetchall("SELECT ROWID, * FROM relpath_info")
        if relpath:
            rows = [x for x in rows if x[0] == relpath]
        print("relpath_info:")  # noqa: T201
        print("\n".join(repr(row) for row in rows))  # noqa: T201
        rows = self.fetchall("SELECT * FROM relpath_deleted_at")
        if relpath:
            rows = [x for x in rows if x[0] == relpath]
        print("relpath_deleted_at:")  # noqa: T201
        print("\n".join(repr(row) for row in rows))  # noqa: T201
        rows = self.fetchall("SELECT * FROM kvchangelog")
        if relpath:
            rows = [x for x in rows if x[0] == relpath]
        print("kvchangelog:")  # noqa: T201
        print("\n".join(repr(row) for row in rows))  # noqa: T201

    def db_read_last_changelog_serial(self):
        q = 'SELECT max(serial) FROM "renames" LIMIT 1'
        res = self.fetchone(q)[0]
        return -1 if res is None else res

    def db_read_typedkey(self, relpath):
        q = "SELECT ROWID, keyname, serial FROM relpath_info WHERE relpath = ?"
        row = self.fetchone(q, (relpath,))
        if row is None:
            raise KeyError(relpath)
        self._relpath_id_cache[relpath] = row[0]
        return tuple(row[1:])

    @cached_property
    def _relpath_id_cache(self):
        return {}

    def db_write_typedkeys(self, data):
        new_typedkeys = []
        updated_typedkeys = []
        relpaths = set()
        for key, keyname, serial, back_serial in data:
            if back_serial == -1:
                new_typedkeys.append(dict(
                    key=key,
                    keyname=keyname,
                    serial=serial))
            else:
                updated_typedkeys.append(dict(
                    key=key,
                    keyname=keyname,
                    serial=serial,
                    back_serial=back_serial))
            relpaths.add(key)
        q = """
            INSERT INTO relpath_info (relpath, keyname, serial)
            VALUES (:key, :keyname, :serial)"""
        self.executemany(q, new_typedkeys)
        q = """
            UPDATE relpath_info SET serial = :serial
            WHERE relpath = :key AND keyname = :keyname AND serial = :back_serial"""
        self.executemany(q, updated_typedkeys)
        questionmarks = ','.join('?' * len(relpaths))
        q = f"""
            SELECT relpath, ROWID FROM relpath_info WHERE relpath IN ({questionmarks})"""
        result = self.fetchall(q, tuple(relpaths))
        assert len(result) == len(relpaths)
        self._relpath_id_cache.update(result)

    def write_changelog_entry(self, serial, entry):
        (changes, renames) = entry
        threadlog.debug("writing changelog for serial %s", serial)
        kvchangelog = []
        for relpath, (keyname, back_serial, value) in changes.items():
            tkey = self.storage.typed_keys[keyname]
            relpath_id = self._relpath_id_cache[relpath]
            if value is None:
                q = "INSERT INTO relpath_deleted_at (relpath_id, serial) VALUES (?, ?)"
                self.execute(q, (relpath_id, serial))
                items = ((b'', None),)
            elif tkey.type is dict:
                old_value = None
                if back_serial > -1:
                    (_, _, old_value) = self.get_relpath_at(relpath, back_serial)
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
                    assert not all_keys
                    assert value == {}
                    items = ((b'', b''),)
            elif tkey.type is set:
                old_value = None
                if back_serial > -1:
                    (_, _, old_value) = self.get_relpath_at(relpath, back_serial)
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
                    assert not all_keys
                    assert value == set()
                    items = ((b'', b''),)
            else:
                items = ((b'', sqlite3.Binary(dumps(value))),)
            for key, data in items:
                kvchangelog.append((relpath_id, serial, back_serial, key, data))
        q = """
            INSERT INTO kvchangelog (
                relpath_id, serial, back_serial, key, value)
            VALUES (?, ?, ?, ?, ?)"""
        self.executemany(q, kvchangelog)
        self.execute(
            "INSERT INTO renames (serial, data) VALUES (?, ?)",
            (serial, sqlite3.Binary(dumps(renames))))

    def get_raw_changelog_entry(self, serial):
        q = "SELECT data FROM renames WHERE serial = ?"
        row = self.fetchone(q, (serial,))
        if row is None:
            return None
        renames = loads(row[0])
        changes = self._get_changes_at(serial)
        return dumps((changes, renames))

    def _iter_data_from_rows(self, rows_iter):
        return _iter_data_from_rows(self.storage.typed_keys, rows_iter)

    def _process_rows(self, rows_iter, at_serial):
        result = {}
        last_serials = {}
        data_iter = self._iter_data_from_rows(rows_iter)
        for (relpath, keyname, last_serial, back_serial, value) in data_iter:
            result[relpath] = (keyname, back_serial, value)
            last_serials[relpath] = last_serial
        return (result, last_serials)

    def _get_relpath_at(self, relpath, serial):
        q = """
            WITH
                filtered_deleted_at AS (
                    SELECT
                        relpath_info.ROWID,
                        relpath_info.relpath,
                        relpath_info.keyname,
                        coalesce(max(relpath_deleted_at.serial), -1) AS deleted_serial
                    FROM relpath_info
                    LEFT OUTER JOIN relpath_deleted_at ON
                        relpath_deleted_at.relpath_id=relpath_info.ROWID
                        AND relpath_deleted_at.serial<=:serial
                    WHERE
                        relpath_info.relpath=:relpath
                    GROUP BY relpath_info.relpath),
                latest_relpath_keys AS (
                    SELECT
                        filtered_deleted_at.ROWID,
                        filtered_deleted_at.relpath,
                        filtered_deleted_at.keyname,
                        key,
                        filtered_deleted_at.deleted_serial,
                        max(kvchangelog.serial) AS latest_serial
                    FROM kvchangelog
                    JOIN filtered_deleted_at ON
                        kvchangelog.relpath_id=filtered_deleted_at.ROWID
                    WHERE
                        kvchangelog.serial>=filtered_deleted_at.deleted_serial
                        AND kvchangelog.serial<=:serial
                    GROUP BY
                        filtered_deleted_at.ROWID,
                        filtered_deleted_at.relpath,
                        filtered_deleted_at.keyname,
                        key,
                        filtered_deleted_at.deleted_serial)
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
                latest_relpath_keys.latest_serial=kvchangelog.serial
                AND latest_relpath_keys.ROWID=kvchangelog.relpath_id
                AND latest_relpath_keys.key=kvchangelog.key
            ORDER BY latest_relpath_keys.ROWID, kvchangelog.serial DESC"""
        rows_iter = self.iterall(q, dict(relpath=relpath, serial=serial))
        (result, last_serials) = self._process_rows(rows_iter, serial)
        return (last_serials[relpath], result[relpath])

    def get_relpath_at(self, relpath, serial):
        result = self._changelog_cache.get((serial, relpath), absent)
        if result is absent:
            changes = self._changelog_cache.get(serial, absent)
            if changes is not absent and relpath in changes:
                (keyname, back_serial, value) = changes[relpath]
                result = (serial, back_serial, value)
        if result is absent:
            (last_serial, tup) = self._get_relpath_at(relpath, serial)
            (keyname, back_serial, value) = tup
            result = (last_serial, back_serial, ensure_deeply_readonly(value))
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
                        changed_relpaths.relpath,
                        changed_relpaths.keyname,
                        coalesce(max(relpath_deleted_at.serial), -1) AS deleted_serial
                    FROM changed_relpaths
                    LEFT OUTER JOIN relpath_deleted_at ON
                        relpath_deleted_at.relpath_id=changed_relpaths.ROWID
                        AND relpath_deleted_at.serial<=:serial
                    GROUP BY changed_relpaths.relpath),
                latest_relpath_keys AS (
                    SELECT
                        filtered_deleted_at.ROWID,
                        filtered_deleted_at.relpath,
                        filtered_deleted_at.keyname,
                        key,
                        filtered_deleted_at.deleted_serial,
                        max(kvchangelog.serial) AS latest_serial
                    FROM kvchangelog
                    JOIN filtered_deleted_at ON
                        kvchangelog.relpath_id=filtered_deleted_at.ROWID
                    WHERE
                        kvchangelog.serial>=filtered_deleted_at.deleted_serial
                        AND kvchangelog.serial<=:serial
                    GROUP BY
                        filtered_deleted_at.ROWID,
                        filtered_deleted_at.relpath,
                        filtered_deleted_at.keyname,
                        key,
                        filtered_deleted_at.deleted_serial)
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
                latest_relpath_keys.latest_serial=kvchangelog.serial
                AND latest_relpath_keys.ROWID=kvchangelog.relpath_id
                AND latest_relpath_keys.key=kvchangelog.key
            ORDER BY latest_relpath_keys.ROWID, kvchangelog.serial DESC"""
        rows_iter = self.iterall(q, dict(serial=serial))
        (result, last_serials) = self._process_rows(rows_iter, serial)
        # for relpath, tup in result.items():
        #     last_serial = last_serials[relpath]
        #     (_last_serial, _tup) = self._get_relpath_at(relpath, serial)
        #     if (last_serial, tup) != (_last_serial, _tup):
        #         raise RuntimeError
        return result

    def get_changes(self, serial):
        changes = self._changelog_cache.get(serial)
        if changes is None:
            # make values in changes read only so no calling site accidentally
            # modifies data
            changes = self._get_changes_at(serial)
            changes = ensure_deeply_readonly(changes)
            self._changelog_cache.put(serial, changes)
        return changes

    def iter_relpaths_at(self, typedkeys, at_serial):
        q = """
            WITH
                filtered_deleted_at AS (
                    SELECT
                        relpath_info.ROWID,
                        relpath_info.relpath,
                        relpath_info.keyname,
                        coalesce(max(relpath_deleted_at.serial), -1) AS deleted_serial
                    FROM relpath_info
                    LEFT OUTER JOIN relpath_deleted_at ON
                        relpath_deleted_at.relpath_id=relpath_info.ROWID
                        AND relpath_deleted_at.serial<=:serial
                    WHERE
                        relpath_info.serial=:serial
                        AND relpath_info.keyname IN (:keynames)
                    GROUP BY relpath_info.relpath),
                latest_relpath_keys AS (
                    SELECT
                        filtered_deleted_at.ROWID,
                        filtered_deleted_at.relpath,
                        filtered_deleted_at.keyname,
                        key,
                        filtered_deleted_at.deleted_serial,
                        max(kvchangelog.serial) AS latest_serial
                    FROM kvchangelog
                    JOIN filtered_deleted_at ON
                        kvchangelog.relpath_id=filtered_deleted_at.ROWID
                    WHERE
                        kvchangelog.serial>=filtered_deleted_at.deleted_serial
                    GROUP BY
                        filtered_deleted_at.ROWID,
                        filtered_deleted_at.relpath,
                        filtered_deleted_at.keyname,
                        key,
                        filtered_deleted_at.deleted_serial)
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
                latest_relpath_keys.latest_serial=kvchangelog.serial
                AND latest_relpath_keys.ROWID=kvchangelog.relpath_id
                AND latest_relpath_keys.key=kvchangelog.key
            JOIN relpath_info ON
                kvchangelog.relpath_id=relpath_info.ROWID
            ORDER BY relpath_info.serial DESC, latest_relpath_keys.ROWID, kvchangelog.serial DESC"""
        keynames = frozenset(k.name for k in typedkeys)
        keyname_id_values = {"keynameid%i" % i: k for i, k in enumerate(keynames)}
        q = q.replace(':keynames', ", ".join(':' + x for x in keyname_id_values))
        for serial in range(at_serial, -1, -1):
            rows = self.fetchall(q, dict(serial=serial, **keyname_id_values))
            if not rows:
                continue
            data_iter = self._iter_data_from_rows(rows)
            for (relpath, keyname, last_serial, back_serial, value) in data_iter:
                yield RelpathInfo(
                    relpath=relpath, keyname=keyname,
                    serial=last_serial, back_serial=back_serial,
                    value=value)


class Storage(BaseStorage):
    Connection = Connection
    db_filename = ".sqlite2"
    expected_schema = dict(
        index=dict(
            relpath_info_relpath_idx="""
                CREATE UNIQUE INDEX relpath_info_relpath_idx ON relpath_info (relpath);
            """,
            relpath_info_serial_idx="""
                CREATE INDEX relpath_info_serial_idx ON relpath_info (serial);
            """,
            relpath_deleted_at_relpath_id_idx="""
                CREATE INDEX relpath_deleted_at_relpath_id_idx ON relpath_deleted_at (relpath_id);
            """,
            kvchangelog_relpath_id_idx="""
                CREATE INDEX kvchangelog_relpath_id_idx ON kvchangelog (relpath_id);
            """,
            kvchangelog_serial_idx="""
                CREATE INDEX kvchangelog_serial_idx ON kvchangelog (serial);
            """,
            kvchangelog_lookup_idx="""
                CREATE INDEX kvchangelog_lookup_idx ON kvchangelog (serial, relpath_id, key);
            """),
        table=dict(
            relpath_info="""
                CREATE TABLE relpath_info (
                    relpath TEXT NOT NULL,
                    keyname TEXT NOT NULL,
                    serial INTEGER NOT NULL
                )
            """,
            relpath_deleted_at="""
                CREATE TABLE relpath_deleted_at (
                    relpath_id INTEGER NOT NULL,
                    serial INTEGER NOT NULL,
                    FOREIGN KEY(relpath_id) REFERENCES relpath_info(ROWID)
                )
            """,
            kvchangelog="""
                CREATE TABLE kvchangelog (
                    relpath_id INTEGER NOT NULL,
                    serial INTEGER NOT NULL,
                    back_serial INTEGER NOT NULL,
                    key BLOB NOT NULL,
                    value BLOB,
                    FOREIGN KEY(relpath_id) REFERENCES relpath_info(ROWID)
                )
            """,
            renames="""
                CREATE TABLE renames (
                    serial INTEGER NOT NULL PRIMARY KEY,
                    data BLOB NOT NULL
                )
            """))

    def __init__(self, basedir, notify_on_commit, cache_size):
        super().__init__(basedir, notify_on_commit, cache_size)
        self.typed_keys = {}

    def add_key(self, key):
        self.typed_keys[key.name] = key


@hookimpl
def devpiserver_storage_backend(settings):
    return dict(
        storage=Storage,
        name="sqlite2",
        description="New SQLite backend with files on the filesystem",
        db_filestore=False,
        hidden=True,
        _test_markers=["storage_with_filesystem"])
