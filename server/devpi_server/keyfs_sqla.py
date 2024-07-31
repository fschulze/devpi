from __future__ import annotations

from .config import hookimpl
from .fileutil import dumps
from .fileutil import loads
from .interfaces import IStorageConnection4
from .interfaces import IWriter2
from .keyfs import KeyfsTimeoutError
from .keyfs_types import PTypedKey
from .keyfs_types import RelpathInfo
from .keyfs_types import TypedKey
from .log import thread_pop_log
from .log import thread_push_log
from .log import threadlog
from .markers import absent
from .mythread import current_thread
from .readonly import ReadonlyView
from .readonly import ensure_deeply_readonly
from .sizeof import gettotalsizeof
from collections import Counter
from contextlib import suppress
from devpi_common.types import cached_property
from functools import partial
from repoze.lru import LRUCache
from typing import TYPE_CHECKING
from zope.interface import implementer
import contextlib
import secrets
import sqlalchemy as sa
import time
import warnings


if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from typing import Any


warnings.simplefilter("error", sa.exc.SADeprecationWarning)
warnings.simplefilter("error", sa.exc.SAWarning)


metadata_obj = sa.MetaData()


ulid_latest_serial_table = sa.Table(
    "ulid_latest_serial",
    metadata_obj,
    sa.Column("ulid", sa.BigInteger, primary_key=True),
    sa.Column("serial", sa.Integer, nullable=False))


relpath_ulid_table = sa.Table(
    "relpath_ulid",
    metadata_obj,
    sa.Column("relpath", sa.String, index=True, nullable=False),
    sa.Column("ulid", sa.BigInteger, nullable=False),
    sa.Column("keytype", sa.String, index=True, nullable=False),
    sa.Column("serial", sa.Integer, nullable=False))


renames_table = sa.Table(
    "renames",
    metadata_obj,
    sa.Column("serial", sa.Integer, primary_key=True),
    sa.Column("data", sa.BINARY, nullable=False))


ulid_changelog_table = sa.Table(
    "ulid_changelog",
    metadata_obj,
    sa.Column("ulid", sa.BigInteger, index=True, nullable=False),
    sa.Column("serial", sa.Integer, index=True, nullable=False),
    sa.Column("back_serial", sa.Integer, nullable=False),
    sa.Column("value", sa.BINARY, nullable=True))


def make_ulid(_randbits=secrets.randbits, _time_ns=time.time_ns):
    return ((_time_ns() // 1_000_000) << 16 | _randbits(16))


@implementer(IStorageConnection4)
class Connection:
    def __init__(self, sqlaconn, storage):
        self._sqlaconn = sqlaconn
        self.storage = storage
        self.dirty_files = {}

    def close(self):
        if hasattr(self, "_sqlaconn"):
            self._sqlaconn.close()
            del self._sqlaconn
        if hasattr(self, "storage"):
            del self.storage

    def commit(self):
        self._sqlaconn.commit()

    def db_read_last_changelog_serial(self) -> int:
        return self._sqlaconn.execute(sa.select(sa.func.coalesce(
            sa.func.max(renames_table.c.serial),
            -1))).scalar()

    def db_read_typedkey(self, relpath: str) -> tuple[str, int]:
        latest_serial_stmt = (
            sa.select(
                relpath_ulid_table.c.relpath,
                sa.func.max(relpath_ulid_table.c.serial).label("serial"))
            .where(
                relpath_ulid_table.c.relpath == relpath)
            .group_by(
                relpath_ulid_table.c.relpath))
        latest_serial_sq = latest_serial_stmt.subquery("latest_serial_sq")
        stmt = (
            sa.select(
                relpath_ulid_table.c.keytype,
                latest_serial_sq.c.serial)
            .join(
                latest_serial_sq,
                sa.and_(
                    relpath_ulid_table.c.relpath == latest_serial_sq.c.relpath,
                    relpath_ulid_table.c.serial == latest_serial_sq.c.serial)))
        row = self._sqlaconn.execute(stmt).one_or_none()
        if row is None:
            raise KeyError(relpath)
        return row

    def _db_write_typedkeys(self, new_typedkeys, updated_typedkeys):
        if new_typedkeys:
            self._sqlaconn.execute(
                sa.insert(relpath_ulid_table),
                new_typedkeys)
            self._sqlaconn.execute(
                sa.insert(ulid_latest_serial_table),
                [dict(serial=x["serial"], ulid=x["ulid"]) for x in new_typedkeys])
        if updated_typedkeys:
            stmt = (
                sa.update(relpath_ulid_table)
                .where(
                    relpath_ulid_table.c.relpath == sa.bindparam("b_relpath"),
                    relpath_ulid_table.c.keytype == sa.bindparam("b_keytype"),
                    relpath_ulid_table.c.serial == sa.bindparam("b_back_serial"))
                .values(serial=sa.bindparam("b_serial"))
                .returning(
                    relpath_ulid_table.c.ulid,
                    relpath_ulid_table.c.serial,
                    sa.bindparam("b_back_serial")))
            if self._sqlaconn.dialect.update_executemany_returning:
                ulid_serials = [
                    dict(b_ulid=x.ulid, b_serial=x.serial, b_back_serial=x.b_back_serial)
                    for x in self._sqlaconn.execute(stmt, updated_typedkeys)]
            else:
                ulid_serials = []
                for updated_typedkey in updated_typedkeys:
                    result = self._sqlaconn.execute(stmt, updated_typedkey).fetchall()
                    if not result:
                        raise RuntimeError
                    ulid_serials.extend(
                        dict(b_ulid=x.ulid, b_serial=x.serial, b_back_serial=x.b_back_serial)
                        for x in result)
            stmt = (
                sa.update(ulid_latest_serial_table)
                .where(
                    ulid_latest_serial_table.c.ulid == sa.bindparam("b_ulid"),
                    ulid_latest_serial_table.c.serial == sa.bindparam("b_back_serial"))
                .values(serial=sa.bindparam("b_serial")))
            self._sqlaconn.execute(stmt, ulid_serials)

    def _get_changes_at(self, serial):
        execute = self._sqlaconn.execute
        stmt = (
            sa.select(
                relpath_ulid_table.c.relpath,
                relpath_ulid_table.c.keytype,
                ulid_changelog_table.c.back_serial)
            .join(
                ulid_changelog_table,
                relpath_ulid_table.c.ulid == ulid_changelog_table.c.ulid)
            .where(
                ulid_changelog_table.c.serial == serial))
        relpath_infos = {x.relpath: x for x in execute(stmt)}
        counts = Counter(relpath_infos.keys())
        if (v := counts.values()) and set(v) != {1}:
            raise RuntimeError
        relpaths_stmt = stmt.with_only_columns(relpath_ulid_table.c.relpath)
        results = {}
        for relpath, info in self._get_relpaths_at(relpaths_stmt, serial).items():
            results[relpath] = (
                relpath_infos[relpath].keytype,
                info["back_serial"],
                info["value"])
        for relpath in set(relpath_infos).difference(results):
            relpath_info = relpath_infos[relpath]
            results[relpath] = (
                relpath_info.keytype, relpath_info.back_serial, None)
        return results

    def get_changes(self, serial: int) -> dict:
        return ensure_deeply_readonly(self._get_changes_at(serial))

    def get_raw_changelog_entry(self, serial: int) -> bytes | None:
        renames = self.get_rel_renames(serial)
        if renames is None:
            return None
        changes = self._get_changes_at(serial)
        return dumps((changes, renames))

    def get_rel_renames(self, serial) -> list | None:
        data = self._sqlaconn.execute(
            sa.select(renames_table.c.data)
            .where(renames_table.c.serial == serial)).scalar()
        return None if data is None else loads(data)

    def _get_relpaths_at(self, relpaths, serial):
        execute = self._sqlaconn.execute
        relpaths_stmt = (
            sa.select(relpath_ulid_table)
            .where(
                relpath_ulid_table.c.relpath.in_(relpaths)))
        deleted_at_stmt = (
            sa.select(
                ulid_changelog_table.c.ulid,
                sa.func.max(ulid_changelog_table.c.serial).label("deleted_serial"))
            .where(
                ulid_changelog_table.c.value.is_(None),
                ulid_changelog_table.c.serial <= serial)
            .group_by(ulid_changelog_table.c.ulid))
        relpaths_sq = relpaths_stmt.subquery("relpaths_sq")
        deleted_at_sq = deleted_at_stmt.subquery("deleted_at_sq")
        relpaths_info_stmt = (
            sa.select(
                relpaths_sq.c.relpath,
                relpaths_sq.c.ulid,
                sa.func.coalesce(
                    deleted_at_sq.c.deleted_serial,
                    -1).label("deleted_serial"))
            .join_from(
                relpaths_sq,
                deleted_at_sq,
                relpaths_sq.c.ulid == deleted_at_sq.c.ulid,
                isouter=True))
        relpaths_info_sq = relpaths_info_stmt.subquery("relpaths_info_sq")
        ulid_relpath_map = {x.ulid: x.relpath for x in execute(relpaths_info_stmt)}
        result = {}
        if not ulid_relpath_map:
            return result
        ulid_changelog_stmt = (
            sa.select(ulid_changelog_table)
            .join_from(
                relpaths_info_sq,
                ulid_changelog_table,
                relpaths_info_sq.c.ulid == ulid_changelog_table.c.ulid)
            .where(
                ulid_changelog_table.c.serial <= serial,
                ulid_changelog_table.c.serial >= relpaths_info_sq.c.deleted_serial))
        ulid_changelog_sq = ulid_changelog_stmt.subquery("ulid_changelog_sq")
        latest_ulid_stmt = (
            sa.select(
                ulid_changelog_sq.c.ulid,
                sa.func.max(ulid_changelog_sq.c.serial).label("serial"))
            .group_by(ulid_changelog_sq.c.ulid))
        latest_ulid_sq = latest_ulid_stmt.subquery("latest_ulid_sq")
        stmt = (
            sa.select(
                ulid_changelog_sq.c.ulid,
                ulid_changelog_sq.c.value)
            .join(
                latest_ulid_sq,
                sa.and_(
                    latest_ulid_sq.c.ulid == ulid_changelog_sq.c.ulid,
                    latest_ulid_sq.c.serial == ulid_changelog_sq.c.serial))
            .order_by(ulid_changelog_sq.c.ulid))
        current_obj = None
        current_ulid = None
        ulid_serials_stmt = (
            sa.select(
                ulid_changelog_sq.c.ulid,
                sa.func.max(ulid_changelog_sq.c.back_serial),
                sa.func.max(ulid_changelog_sq.c.serial))
            .group_by(ulid_changelog_sq.c.ulid))
        ulid_back_serial_map = {}
        ulid_serial_map = {}
        for ulid, back_serial, ulid_serial in execute(ulid_serials_stmt):
            ulid_back_serial_map[ulid] = back_serial
            ulid_serial_map[ulid] = ulid_serial
        rows = execute(stmt).all()
        for ulid, value in rows:
            if current_ulid != ulid:
                current_ulid = ulid
                current_obj = None if value is None else loads(value)
                result[ulid_relpath_map[ulid]] = dict(
                    last_serial=ulid_serial_map[ulid],
                    back_serial=ulid_back_serial_map[ulid],
                    value=current_obj)
        return result

    def get_relpath_at(self, relpath: str, serial: int) -> Any:
        result = self._get_relpaths_at((relpath,), serial)
        if relpath not in result:
            raise KeyError(relpath)
        result = result[relpath]
        return (result["last_serial"], result["back_serial"], ensure_deeply_readonly(result["value"]))

    def iter_relpaths_at(self, typedkeys: list[PTypedKey | TypedKey], serial: int) -> Iterator[RelpathInfo]:
        execute = self._sqlaconn.execute
        keytypes = frozenset(k.name for k in typedkeys)
        stmt = (
            sa.select(
                relpath_ulid_table.c.relpath.distinct(),
                relpath_ulid_table.c.keytype)
            .join(
                relpath_ulid_table,
                relpath_ulid_table.c.ulid == ulid_changelog_table.c.ulid)
            .where(
                ulid_changelog_table.c.serial <= serial,
                relpath_ulid_table.c.keytype.in_(keytypes)))
        relpath_keytype_map = dict(execute(stmt).all())
        relpaths_stmt = stmt.with_only_columns(relpath_ulid_table.c.relpath)
        for relpath, info in self._get_relpaths_at(relpaths_stmt, serial).items():
            keytype = relpath_keytype_map[relpath]
            yield RelpathInfo(
                relpath=relpath, keyname=keytype,
                serial=info["last_serial"], back_serial=info["back_serial"],
                value=info["value"])

    @cached_property
    def last_changelog_serial(self) -> int:
        return self.db_read_last_changelog_serial()

    def rollback(self):
        self._sqlaconn.rollback()

    def _write_records(self, serial, records, renames):
        execute = self._sqlaconn.execute
        threadlog.debug("writing changelog for serial %s", serial)
        ulid_changelog = []
        latest_serial_stmt = (
            sa.select(
                relpath_ulid_table.c.relpath,
                sa.func.max(relpath_ulid_table.c.serial).label("serial"))
            .where(
                relpath_ulid_table.c.relpath.in_(
                    {x.key.relpath for x in records}))
            .group_by(
                relpath_ulid_table.c.relpath))
        latest_serial_sq = latest_serial_stmt.subquery("latest_serial_sq")
        stmt = (
            sa.select(
                relpath_ulid_table.c.relpath,
                relpath_ulid_table.c.ulid)
            .join(
                latest_serial_sq,
                sa.and_(
                    relpath_ulid_table.c.relpath == latest_serial_sq.c.relpath,
                    relpath_ulid_table.c.serial == latest_serial_sq.c.serial)))
        relpath_ulid_map = dict(execute(stmt).all())
        append = ulid_changelog.append
        for record in records:
            ulid = relpath_ulid_map[record.key.relpath]
            if record.value is None:
                append((ulid, serial, record.back_serial, None))
            else:
                append((ulid, serial, record.back_serial, dumps(record.value)))
        if ulid_changelog:
            execute(
                sa.insert(ulid_changelog_table)
                .values(ulid_changelog))
        execute(
            sa.insert(renames_table)
            .values((serial, dumps(renames))))

    def write_transaction(self, io_file: Any = None) -> AbstractContextManager:
        return Writer(self.storage, self, io_file)


class LazyRecordsFormatter:
    __slots__ = ("files_commit", "files_del", "keys")

    def __init__(self, records, files_commit, files_del):
        self.files_commit = files_commit
        self.files_del = files_del
        self.keys = {x.key.relpath for x in records}

    def __str__(self):
        msg = []
        if self.keys:
            msg.append(f"keys: {','.join(sorted(repr(c) for c in self.keys))}")
        if self.files_commit:
            msg.append(f"files_commit: {','.join(self.files_commit)}")
        if self.files_del:
            msg.append(f"files_del: {','.join(self.files_del)}")
        return ", ".join(msg)


@implementer(IWriter2)
class Writer:
    def __init__(self, storage, conn, io_file):
        self.conn = conn
        self.io_file = io_file
        self.storage = storage
        self.records = None
        self.rel_renames = []

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
        records = self.records
        del self.records
        rel_renames = self.rel_renames
        del self.rel_renames
        new_typedkeys = []
        updated_typedkeys = []
        for record in records:
            if record.back_serial is None:
                raise RuntimeError
            assert not isinstance(record.value, ReadonlyView), record.value
            if record.back_serial == -1 or record.old_value is absent:
                new_typedkeys.append(dict(
                    relpath=record.key.relpath,
                    ulid=make_ulid(),
                    keytype=record.key.name,
                    serial=commit_serial))
            else:
                updated_typedkeys.append(dict(
                    b_relpath=record.key.relpath,
                    b_keytype=record.key.name,
                    b_serial=commit_serial,
                    b_back_serial=record.back_serial))
        self.conn._db_write_typedkeys(new_typedkeys, updated_typedkeys)
        del new_typedkeys, updated_typedkeys
        self.conn._write_records(commit_serial, records, rel_renames)
        if self.io_file:
            (files_commit, files_del) = self.io_file.write_dirty_files(rel_renames)
        else:
            (files_commit, files_del) = ([], [])
        self.conn.commit()
        self.storage.last_commit_timestamp = time.time()
        return LazyRecordsFormatter(records, files_commit, files_del)

    def records_set(self, records):
        assert records is not None
        assert self.records is None
        self.records = records

    def rollback(self):
        with suppress(AttributeError):
            del self.records
        if self.io_file:
            self.io_file.drop_dirty_files()
        self.conn.rollback()

    def set_rel_renames(self, rel_renames):
        assert rel_renames is not None
        assert self.rel_renames == []
        self.rel_renames = rel_renames


class Storage:
    def __init__(self, basedir, *, notify_on_commit, cache_size, settings=None):
        self.basedir = basedir
        self.sqlpath = self.basedir / ".sqlite_alchemy"
        if settings is None:
            settings = {}
        self.ro_engine = sa.create_engine(self._url(mode="ro"), echo=False)
        self.rw_engine = sa.create_engine(self._url(mode="rw"), echo=False)
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
        self.typed_keys = {}

    def _url(self, *, mode):
        return f"sqlite+pysqlite:///file:{self.sqlpath}?mode={mode}&timeout=30&uri=true"

    def add_key(self, key):
        self.typed_keys[key.name] = key

    def ensure_tables_exist(self):
        if not self.sqlpath.exists():
            engine = sa.create_engine(self._url(mode="rwc"), echo=False)
            metadata_obj.create_all(engine)

    def get_connection(self, *, closing: bool = True, write: bool = False, timeout: int = 30):
        engine = self.rw_engine if write else self.ro_engine
        sqlaconn = engine.connect()
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


@hookimpl
def devpiserver_storage_backend(settings):
    return dict(
        storage=partial(Storage, settings=settings),
        name="sqla",
        description="SQLAlchemy backend",
        db_filestore=False)
