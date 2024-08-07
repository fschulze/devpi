from __future__ import annotations

from .config import hookimpl
from .filestore_fs import LazyChangesFormatter
from .fileutil import dumps
from .fileutil import loads
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
from contextlib import suppress
from devpi_common.types import cached_property
from typing import TYPE_CHECKING
from zope.interface import implementer
import contextlib
import random
import sqlalchemy as sa
import time
import warnings


if TYPE_CHECKING:
    from .interfaces import IIOFile
    from .keyfs_types import IKeyFSKey
    from .keyfs_types import LocatedKey
    from collections.abc import Iterable
    from collections.abc import Iterator


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


def make_ulid(_randbits=random.getrandbits, _time_ns=time.time_ns):
    ns = _time_ns()
    high_part = (ns // 1_000_000_000) << 26
    low_part = _randbits(26)
    return high_part | low_part


@implementer(IStorageConnection)
class Connection:
    def __init__(self, sqlaconn, storage):
        self._sqlaconn = sqlaconn
        self.storage = storage
        self.dirty_files = {}

    def _dump(self):
        # for debugging
        execute = self._sqlaconn.execute
        for table in (ulid_latest_serial_table, relpath_ulid_table, ulid_changelog_table):
            print(f"{table}:")  # noqa: T201
            for row in execute(sa.select(table)):
                print(f"  {row}")  # noqa: T201

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

    def last_key_serial(self, key: LocatedKey) -> int:
        latest_serial_stmt = (
            sa.select(
                relpath_ulid_table.c.relpath,
                sa.func.max(relpath_ulid_table.c.serial).label("serial"))
            .where(
                relpath_ulid_table.c.relpath == key.relpath)
            .group_by(
                relpath_ulid_table.c.relpath))
        latest_serial_sq = latest_serial_stmt.subquery("latest_serial_sq")
        stmt = (
            sa.select(
                relpath_ulid_table.c.serial)
            .join(
                latest_serial_sq,
                sa.and_(
                    relpath_ulid_table.c.relpath == latest_serial_sq.c.relpath,
                    relpath_ulid_table.c.serial == latest_serial_sq.c.serial)))
        result = self._sqlaconn.execute(stmt).scalar()
        if result is None:
            raise KeyError(key)
        return result

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

    def iter_changes_at(self, serial: int) -> Iterator[KeyData]:
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
        relpaths_stmt = stmt.with_only_columns(relpath_ulid_table.c.relpath)
        yield from self._iter_relpaths_at(relpaths_stmt, serial)

    def get_raw_changelog_entry(self, serial: int) -> bytes | None:
        data = self._sqlaconn.execute(
            sa.select(renames_table.c.data)
            .where(renames_table.c.serial == serial)).scalar()
        if data is None:
            return None
        renames = loads(data)
        changes = {
            c.relpath: (
                c.keyname,
                c.back_serial,
                None if c.value is deleted else c.value)
            for c in self.iter_changes_at(serial)}
        return dumps((changes, renames))

    def iter_rel_renames(self, serial: int) -> Iterator[str]:
        data = self._sqlaconn.execute(
            sa.select(renames_table.c.data)
            .where(renames_table.c.serial == serial)).scalar()
        if data is None:
            return
        yield from loads(data)

    def _iter_relpaths_at(self, relpaths, serial):
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
                relpaths_sq.c.keytype,
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
        ulid_relpath_map = {}
        ulid_keytype_map = {}
        for info in execute(relpaths_info_stmt):
            ulid_relpath_map[info.ulid] = info.relpath
            ulid_keytype_map[info.ulid] = info.keytype
        if not ulid_relpath_map:
            return
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
            yield KeyData(
                relpath=ulid_relpath_map[ulid], keyname=ulid_keytype_map[ulid],
                serial=ulid_serial_map[ulid], back_serial=ulid_back_serial_map[ulid],
                value=deleted if value is None else ensure_deeply_readonly(loads(value)))

    def get_key_at_serial(self, key: LocatedKey, serial: int) -> KeyData:
        results = list(self._iter_relpaths_at((key.relpath,), serial))
        if not results:
            raise KeyError(key)
        (result,) = results
        return result

    def iter_relpaths_at(self, typedkeys: Iterable[IKeyFSKey], at_serial: int) -> Iterator[KeyData]:
        keytypes = frozenset(k.key_name for k in typedkeys)
        stmt = (
            sa.select(
                relpath_ulid_table.c.relpath.distinct(),
                relpath_ulid_table.c.keytype)
            .join(
                relpath_ulid_table,
                relpath_ulid_table.c.ulid == ulid_changelog_table.c.ulid)
            .where(
                ulid_changelog_table.c.serial <= at_serial,
                relpath_ulid_table.c.keytype.in_(keytypes)))
        relpaths_stmt = stmt.with_only_columns(relpath_ulid_table.c.relpath)
        yield from self._iter_relpaths_at(relpaths_stmt, at_serial)

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
            if record.value is deleted:
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

    def write_transaction(self, io_file: IIOFile | None) -> IWriter:
        return Writer(self.storage, self, io_file)


@implementer(IWriter)
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
                    keytype=record.key.key_name,
                    serial=commit_serial))
            else:
                updated_typedkeys.append(dict(
                    b_relpath=record.key.relpath,
                    b_keytype=record.key.key_name,
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
        return LazyChangesFormatter({(x.key.key_name, x.key.relpath) for x in records}, files_commit, files_del)

    def records_set(self, records):
        assert records is not None
        assert self.records is None
        self.records = records

    def rollback(self):
        with suppress(AttributeError):
            del self.records
        self.conn.rollback()

    def set_rel_renames(self, rel_renames):
        assert rel_renames is not None
        assert self.rel_renames == []
        self.rel_renames = rel_renames


@implementer(IStorage)
class Storage:
    db_filename = ".sqlite_alchemy"

    def __init__(self, basedir, *, notify_on_commit, cache_size, settings):  # noqa: ARG002
        self.basedir = basedir
        self.sqlpath = self.basedir / self.db_filename
        self.ro_engine = sa.create_engine(self._url(mode="ro"), echo=False)
        self.rw_engine = sa.create_engine(self._url(mode="rw"), echo=False)
        self._notify_on_commit = notify_on_commit
        self.last_commit_timestamp = time.time()
        self.ensure_tables_exist()

    @classmethod
    def exists(cls, basedir, settings):  # noqa: ARG003
        sqlpath = basedir / cls.db_filename
        return sqlpath.exists()

    def _url(self, *, mode):
        return f"sqlite+pysqlite:///file:{self.sqlpath}?mode={mode}&timeout=30&uri=true"

    def register_key(self, key):
        pass

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

    def perform_crash_recovery(self):
        pass


@hookimpl
def devpiserver_describe_storage_backend(settings):
    return StorageInfo(
        name="sqla",
        description="SQLAlchemy backend",
        exists=Storage.exists,
        hidden=True,
        storage_cls=Storage,
        connection_cls=Connection,
        writer_cls=Writer,
        storage_factory=Storage,
        settings=settings)
