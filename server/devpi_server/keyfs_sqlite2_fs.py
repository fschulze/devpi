from .config import hookimpl
from .interfaces import IStorageConnection3
from .keyfs import RelpathInfo
from .keyfs_sqlite_fs import Connection as BaseConnection
from .keyfs_sqlite_fs import Storage as BaseStorage
from .log import threadlog
from .readonly import ReadonlyView
from .readonly import ensure_deeply_readonly
from .fileutil import dumps, loads
from devpi_common.types import cached_property
from zope.interface import implementer
import sqlite3


absent = object()


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
            if tkey_type == set and v != b'':
                raise RuntimeError("An item in a set can't have a value")
            if k not in deleted:
                if tkey_type == dict:
                    items.append((k, loads(v)))
                elif tkey_type == set:
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


@implementer(IStorageConnection3)
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

    def db_read_last_changelog_serial(self):
        q = 'SELECT max(serial) FROM "renames" LIMIT 1'
        res = self._sqlconn.execute(q).fetchone()[0]
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

    def _db_write_typedkey(self, relpath, name, next_serial):
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
        c = self._sqlconn.cursor()
        c.execute(q, dict(
            relpath=relpath, keyname=name, serial=next_serial))
        self._relpath_id_cache[relpath] = c.lastrowid
        c.close()

    def _write_changelog_entry(self, commit_serial, entry, old_values):
        (changes, renames) = entry
        threadlog.debug("writing changelog for serial %s", commit_serial)
        kvchangelog = []
        for relpath, (keyname, back_serial, value) in changes.items():
            tkey_type = self.storage.typed_keys[keyname].type
            relpath_id = self._relpath_id_cache[relpath]
            if value is None:
                q = "INSERT INTO relpath_deleted_at (relpath_id, serial) VALUES (?, ?)"
                self._sqlconn.execute(q, (relpath_id, commit_serial))
                items = ((b'', None),)
            elif tkey_type in (dict, set):
                old_value = old_values[relpath]
                items = []
                all_keys = set(old_value).union(value)
                for k in all_keys:
                    if k not in value:
                        items.append((sqlite3.Binary(dumps(k)), None))
                    elif tkey_type == set and k not in old_value:
                        items.append((sqlite3.Binary(dumps(k)), b''))
                    elif tkey_type == dict:
                        v = value[k]
                        if k not in old_value or old_value[k] != v:
                            items.append((
                                sqlite3.Binary(dumps(k)),
                                sqlite3.Binary(dumps(v))))
                if not items:
                    items = ((b'', b''),)
            else:
                items = ((b'', sqlite3.Binary(dumps(value))),)
            for key, data in items:
                kvchangelog.append(
                    (relpath_id, commit_serial, back_serial, key, data))
        q = """
            INSERT INTO kvchangelog (
                relpath_id, serial, back_serial, key, value)
            VALUES (?, ?, ?, ?, ?)"""
        self._sqlconn.executemany(q, kvchangelog)
        self._sqlconn.execute(
            "INSERT INTO renames (serial, data) VALUES (?, ?)",
            (commit_serial, sqlite3.Binary(dumps(renames))))

    def write_changelog_entries(self, commit_serial, entry):
        changes = entry[0]
        for relpath, (keyname, back_serial, value) in changes.items():
            if back_serial is None:
                try:
                    (_, back_serial) = self.db_read_typedkey(relpath)
                except KeyError:
                    back_serial = -1
                # update back_serial for _write_changelog_entry
                changes[relpath] = (keyname, back_serial, value)
        old_values = {}
        for relpath, (keyname, back_serial, value) in changes.items():
            tkey_type = self.storage.typed_keys[keyname].type
            if tkey_type in (dict, set):
                old_value = None
                if back_serial > -1:
                    (_, _, old_value) = self.get_relpath_at(relpath, back_serial)
                if old_value is None:
                    old_value = tkey_type()
                old_values[relpath] = old_value
        for relpath, (keyname, back_serial, value) in changes.items():
            self._db_write_typedkey(relpath, keyname, commit_serial)
        self._write_changelog_entry(commit_serial, entry, old_values)

    def get_raw_changelog_entry(self, serial):
        q = "SELECT data FROM renames WHERE serial = ?"
        row = self.fetchone(q, (serial,))
        if row is None:
            return None
        renames = loads(row[0])
        changes = self._changelog_cache.get(serial, absent)
        if changes is absent:
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
        return (
            last_serials[relpath],  # last_serial
            result[relpath][1],  # back_serial
            ensure_deeply_readonly(result[relpath][2]))  # value

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

    def add_key(self, key):
        if not hasattr(self, 'typed_keys'):
            self.typed_keys = {}
        self.typed_keys[key.name] = key


@hookimpl
def devpiserver_storage_backend(settings):
    return dict(
        storage=Storage,
        name="sqlite2",
        description="New SQLite backend with files on the filesystem",
        _test_markers=["storage_with_filesystem"])
