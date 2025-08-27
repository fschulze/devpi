from __future__ import annotations

from .filestore_fs import LazyChangesFormatter
from .fileutil import dumps
from .fileutil import loads
from .interfaces import IWriter
from .keyfs_types import KeyData
from .keyfs_types import ULID
from .log import thread_pop_log
from .log import thread_push_log
from .log import threadlog
from .markers import absent
from .markers import deleted
from .readonly import ReadonlyView
from .readonly import ensure_deeply_readonly
from .sizeof import gettotalsizeof
from collections import defaultdict
from contextlib import suppress
from devpi_common.types import cached_property
from repoze.lru import LRUCache
from sqlalchemy.ext.compiler import compiles
from typing import TYPE_CHECKING
from zope.interface import implementer
import inspect
import sqlalchemy as sa
import time
import warnings


if TYPE_CHECKING:
    from .interfaces import IIOFile
    from .keyfs_types import IKeyFSKey
    from .keyfs_types import LocatedKey
    from .keyfs_types import Record
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from pathlib import Path
    from sqlalchemy.sql.compiler import SQLCompiler
    from sqlalchemy.sql.roles import InElementRole
    from sqlalchemy.types import _Binary
    from types import TracebackType
    from typing import Any
    from typing import Callable
    from typing import IO
    from typing_extensions import Self


warnings.simplefilter("error", sa.exc.SADeprecationWarning)
warnings.simplefilter("error", sa.exc.SAWarning)


class explain(sa.Executable, sa.ClauseElement):
    inherit_cache = True

    def __init__(self, stmt: Any, *, analyze: bool = False) -> None:
        self.statement = stmt
        self.analyze = analyze


@compiles(explain, "sqlite")
def sqlite_explain(element: explain, compiler: SQLCompiler, **kw: Any) -> str:
    text = "EXPLAIN "
    if element.analyze:
        text += "QUERY PLAN "
    text += compiler.process(element.statement, **kw)

    return text


class BaseConnection:
    dirty_files: dict[str, IO | None]
    relpath_ulid_table: sa.Table
    renames_table: sa.Table
    ulid_changelog_table: sa.Table
    ulid_latest_serial_table: sa.Table

    def __init__(self, sqlaconn: sa.engine.Connection, storage: BaseStorage) -> None:
        self._sqlaconn = sqlaconn
        self.storage = storage
        for name, member in inspect.getmembers(storage):
            if isinstance(member, sa.Table):
                setattr(self, name, member)
        self.dirty_files = {}
        self._changelog_cache = storage._changelog_cache
        self._keys = storage._keys
        self._relpath_cache = storage._relpath_cache

    def _dump(self) -> None:
        # for debugging
        execute = self._sqlaconn.execute
        for table in (
            self.ulid_latest_serial_table,
            self.relpath_ulid_table,
            self.ulid_changelog_table,
        ):
            print(f"{table}:")  # noqa: T201
            for row in execute(sa.select(table)):
                print(f"  {row}")  # noqa: T201

    def _ppexplain(self, stmt: sa.Executable) -> None:
        text = str(
            explain(stmt, analyze=True).compile(
                compile_kwargs=dict(literal_binds=True), dialect=self._sqlaconn.dialect
            )
        )
        results = self._sqlaconn.connection.execute(text).fetchall()
        tree = defaultdict(list)
        for row in results:
            tree[row[1]].append(row)
        stack = [0]
        while stack:
            current = tree[stack[-1]]
            while current:
                row = current.pop(0)
                print("  " * (len(stack) - 1), row[-1])  # noqa: T201 - debugging
                if row[0] in tree:
                    stack.append(row[0])
                    break
            if not current:
                stack.pop()

    def _ppsql(self, stmt: sa.Executable) -> None:
        import sqlparse  # type: ignore[import-untyped]

        text = str(
            stmt.compile(  # type: ignore[attr-defined]
                compile_kwargs=dict(literal_binds=True), dialect=self._sqlaconn.dialect
            )
        )
        print(sqlparse.format(text, reindent=True))  # noqa: T201 - debugging

    def _ppresults(self, results: sa.Executable | Iterable) -> None:
        from pprint import pprint

        if isinstance(results, sa.Executable):
            results = self._sqlaconn.execute(results)
        pprint([x._asdict() for x in results])  # noqa: T203 - debugging

    def close(self) -> None:
        if hasattr(self, "_sqlaconn"):
            self._sqlaconn.close()
            del self._sqlaconn
        if hasattr(self, "storage"):
            del self.storage

    def commit(self) -> None:
        self._sqlaconn.commit()

    def db_read_last_changelog_serial(self) -> int:
        result = self._sqlaconn.execute(
            sa.select(sa.func.coalesce(sa.func.max(self.renames_table.c.serial), -1))
        ).scalar()
        assert isinstance(result, int)
        return result

    def get_next_serial(self) -> int:
        raise NotImplementedError

    def last_key_serial(self, key: LocatedKey) -> int:
        latest_serial_stmt = (
            sa.select(
                self.relpath_ulid_table.c.relpath,
                sa.func.max(self.relpath_ulid_table.c.serial).label("serial"),
            )
            .where(self.relpath_ulid_table.c.relpath == key.relpath)
            .group_by(self.relpath_ulid_table.c.relpath)
        )
        latest_serial_sq = latest_serial_stmt.subquery("latest_serial_sq")
        stmt = sa.select(self.relpath_ulid_table.c.serial).join(
            latest_serial_sq,
            sa.and_(
                self.relpath_ulid_table.c.relpath == latest_serial_sq.c.relpath,
                self.relpath_ulid_table.c.serial == latest_serial_sq.c.serial,
            ),
        )
        result = self._sqlaconn.execute(stmt).scalar()
        if result is None:
            raise KeyError(key)
        return result

    def _db_write_typedkeys(
        self, new_typedkeys: Sequence[dict], updated_typedkeys: Sequence[dict]
    ) -> None:
        if new_typedkeys:
            self._sqlaconn.execute(sa.insert(self.relpath_ulid_table), new_typedkeys)
            self._sqlaconn.execute(
                sa.insert(self.ulid_latest_serial_table),
                [dict(serial=x["serial"], ulid=x["ulid"]) for x in new_typedkeys],
            )
        if updated_typedkeys:
            relpath_update_stmt = (
                sa.update(self.relpath_ulid_table)
                .where(
                    self.relpath_ulid_table.c.relpath == sa.bindparam("b_relpath"),
                    self.relpath_ulid_table.c.keytype == sa.bindparam("b_keytype"),
                    self.relpath_ulid_table.c.serial == sa.bindparam("b_back_serial"),
                )
                .values(serial=sa.bindparam("b_serial"))
                .returning(
                    self.relpath_ulid_table.c.ulid,
                    self.relpath_ulid_table.c.serial,
                    sa.bindparam("b_back_serial"),
                )
            )
            if self._sqlaconn.dialect.update_executemany_returning:
                ulid_serials = [
                    dict(
                        b_ulid=x.ulid, b_serial=x.serial, b_back_serial=x.b_back_serial
                    )
                    for x in self._sqlaconn.execute(
                        relpath_update_stmt, updated_typedkeys
                    )
                ]
            else:
                ulid_serials = []
                for updated_typedkey in updated_typedkeys:
                    result = self._sqlaconn.execute(
                        relpath_update_stmt, updated_typedkey
                    ).fetchall()
                    if not result:
                        raise RuntimeError
                    ulid_serials.extend(
                        dict(
                            b_ulid=x.ulid,
                            b_serial=x.serial,
                            b_back_serial=x.b_back_serial,
                        )
                        for x in result
                    )
            ulid_update_stmt = (
                sa.update(self.ulid_latest_serial_table)
                .where(
                    self.ulid_latest_serial_table.c.ulid == sa.bindparam("b_ulid"),
                    self.ulid_latest_serial_table.c.serial
                    == sa.bindparam("b_back_serial"),
                )
                .values(serial=sa.bindparam("b_serial"))
            )
            self._sqlaconn.execute(ulid_update_stmt, ulid_serials)

    def _get_changes_at(self, serial: int) -> dict[str, tuple[str, int, Any]]:
        stmt = (
            sa.select(
                self.relpath_ulid_table.c.relpath,
                self.relpath_ulid_table.c.keytype,
                self.ulid_changelog_table.c.back_serial,
            )
            .join(
                self.ulid_changelog_table,
                self.relpath_ulid_table.c.ulid == self.ulid_changelog_table.c.ulid,
            )
            .where(self.ulid_changelog_table.c.serial == serial)
        )
        relpaths_stmt = stmt.with_only_columns(self.relpath_ulid_table.c.relpath)
        results = {}
        for keydata in self._iter_relpaths_at(relpaths_stmt, serial):
            results[keydata.relpath] = (
                keydata.keyname,
                keydata.back_serial,
                None if keydata.value is deleted else keydata.value,
            )
        return results

    def get_changes(self, serial: int) -> dict:
        return ensure_deeply_readonly(self._get_changes_at(serial))

    def get_raw_changelog_entry(self, serial: int) -> bytes | None:
        renames = self.get_rel_renames(serial)
        if renames is None:
            return None
        changes = self._get_changes_at(serial)
        return dumps((changes, renames))

    def get_rel_renames(self, serial: int) -> list | None:
        data = self._sqlaconn.execute(
            sa.select(self.renames_table.c.data).where(
                self.renames_table.c.serial == serial
            )
        ).scalar()
        return None if data is None else loads(data)

    def _iter_relpaths_at(
        self, relpaths: InElementRole | Iterable[str], serial: int
    ) -> Iterator[KeyData]:
        execute = self._sqlaconn.execute
        relpaths_stmt = sa.select(self.relpath_ulid_table).where(
            self.relpath_ulid_table.c.relpath.in_(relpaths)
        )
        deleted_at_stmt = (
            sa.select(
                self.ulid_changelog_table.c.ulid,
                sa.func.max(self.ulid_changelog_table.c.serial).label("deleted_serial"),
            )
            .where(
                self.ulid_changelog_table.c.value.is_(None),
                self.ulid_changelog_table.c.serial <= serial,
            )
            .group_by(self.ulid_changelog_table.c.ulid)
        )
        relpaths_sq = relpaths_stmt.subquery("relpaths_sq")
        deleted_at_sq = deleted_at_stmt.subquery("deleted_at_sq")
        relpaths_info_stmt = sa.select(
            relpaths_sq.c.relpath,
            relpaths_sq.c.keytype,
            relpaths_sq.c.ulid,
            sa.func.coalesce(deleted_at_sq.c.deleted_serial, -1).label(
                "deleted_serial"
            ),
        ).join_from(
            relpaths_sq,
            deleted_at_sq,
            relpaths_sq.c.ulid == deleted_at_sq.c.ulid,
            isouter=True,
        )
        relpaths_info_sq = relpaths_info_stmt.subquery("relpaths_info_sq")
        ulid_relpath_map = {}
        ulid_keytype_map = {}
        for info in execute(relpaths_info_stmt):
            ulid_relpath_map[info.ulid] = info.relpath
            ulid_keytype_map[info.ulid] = info.keytype
        if not ulid_relpath_map:
            return
        ulid_changelog_stmt = (
            sa.select(self.ulid_changelog_table)
            .join_from(
                relpaths_info_sq,
                self.ulid_changelog_table,
                relpaths_info_sq.c.ulid == self.ulid_changelog_table.c.ulid,
            )
            .where(
                self.ulid_changelog_table.c.serial <= serial,
                self.ulid_changelog_table.c.serial >= relpaths_info_sq.c.deleted_serial,
            )
        )
        ulid_changelog_sq = ulid_changelog_stmt.subquery("ulid_changelog_sq")
        latest_ulid_stmt = sa.select(
            ulid_changelog_sq.c.ulid,
            sa.func.max(ulid_changelog_sq.c.serial).label("serial"),
        ).group_by(ulid_changelog_sq.c.ulid)
        latest_ulid_sq = latest_ulid_stmt.subquery("latest_ulid_sq")
        stmt = (
            sa.select(ulid_changelog_sq.c.ulid, ulid_changelog_sq.c.value)
            .join(
                latest_ulid_sq,
                sa.and_(
                    latest_ulid_sq.c.ulid == ulid_changelog_sq.c.ulid,
                    latest_ulid_sq.c.serial == ulid_changelog_sq.c.serial,
                ),
            )
            .order_by(ulid_changelog_sq.c.ulid)
        )
        ulid_serials_stmt = sa.select(
            ulid_changelog_sq.c.ulid,
            sa.func.max(ulid_changelog_sq.c.back_serial),
            sa.func.max(ulid_changelog_sq.c.serial),
        ).group_by(ulid_changelog_sq.c.ulid)
        ulid_back_serial_map = {}
        ulid_serial_map = {}
        for ulid, back_serial, ulid_serial in execute(ulid_serials_stmt):
            ulid_back_serial_map[ulid] = back_serial
            ulid_serial_map[ulid] = ulid_serial
        rows = execute(stmt).all()
        for ulid, value in rows:
            yield KeyData(
                relpath=ulid_relpath_map[ulid],
                keyname=ulid_keytype_map[ulid],
                serial=ulid_serial_map[ulid],
                back_serial=ulid_back_serial_map[ulid],
                value=deleted
                if value is None
                else ensure_deeply_readonly(loads(value)),
            )

    def get_key_at_serial(self, key: LocatedKey, serial: int) -> KeyData:
        results = list(self._iter_relpaths_at((key.relpath,), serial))
        if not results:
            raise KeyError(key)
        (result,) = results
        return result

    def iter_relpaths_at(
        self, typedkeys: Iterable[IKeyFSKey], at_serial: int
    ) -> Iterator[KeyData]:
        keytypes = frozenset(k.key_name for k in typedkeys)
        stmt = (
            sa.select(
                self.relpath_ulid_table.c.relpath.distinct(),
                self.relpath_ulid_table.c.keytype,
            )
            .join(
                self.relpath_ulid_table,
                self.relpath_ulid_table.c.ulid == self.ulid_changelog_table.c.ulid,
            )
            .where(
                self.ulid_changelog_table.c.serial <= at_serial,
                self.relpath_ulid_table.c.keytype.in_(keytypes),
            )
        )
        relpaths_stmt = stmt.with_only_columns(self.relpath_ulid_table.c.relpath)
        yield from self._iter_relpaths_at(relpaths_stmt, at_serial)

    @cached_property
    def last_changelog_serial(self) -> int:
        return self.db_read_last_changelog_serial()

    def rollback(self) -> None:
        self._sqlaconn.rollback()

    def _write_dirty_files(self) -> tuple[Sequence, Sequence]:
        raise NotImplementedError

    def _write_records(
        self, serial: int, records: Sequence[Record], renames: Sequence[str]
    ) -> None:
        execute = self._sqlaconn.execute
        threadlog.debug("writing changelog for serial %s", serial)
        ulid_changelog: list[tuple[int, int, int, bytes | None]] = []
        latest_serial_stmt = (
            sa.select(
                self.relpath_ulid_table.c.relpath,
                sa.func.max(self.relpath_ulid_table.c.serial).label("serial"),
            )
            .where(
                self.relpath_ulid_table.c.relpath.in_({x.key.relpath for x in records})
            )
            .group_by(self.relpath_ulid_table.c.relpath)
        )
        latest_serial_sq = latest_serial_stmt.subquery("latest_serial_sq")
        stmt = sa.select(
            self.relpath_ulid_table.c.relpath, self.relpath_ulid_table.c.ulid
        ).join(
            latest_serial_sq,
            sa.and_(
                self.relpath_ulid_table.c.relpath == latest_serial_sq.c.relpath,
                self.relpath_ulid_table.c.serial == latest_serial_sq.c.serial,
            ),
        )
        relpath_ulid_map: dict[str, int] = {
            r.relpath: r.ulid for r in execute(stmt).all()
        }
        append = ulid_changelog.append
        for record in records:
            ulid = relpath_ulid_map[record.key.relpath]
            if record.value is deleted:
                append((ulid, serial, record.back_serial, None))
            else:
                append((ulid, serial, record.back_serial, dumps(record.value)))
        if ulid_changelog:
            execute(sa.insert(self.ulid_changelog_table).values(ulid_changelog))
        execute(sa.insert(self.renames_table).values((serial, dumps(renames))))

    def write_transaction(self, io_file: IIOFile | None) -> IWriter:
        return Writer(self.storage, self, io_file)


@implementer(IWriter)
class Writer:
    records: Sequence[Record]
    rel_renames: Sequence[str]

    def __init__(
        self, storage: BaseStorage, conn: BaseConnection, io_file: IIOFile | None
    ) -> None:
        self.conn = conn
        self.io_file = io_file
        self.storage = storage
        self.rel_renames = []

    def __enter__(self) -> Self:
        self.commit_serial = self.conn.get_next_serial()
        self.log = thread_push_log("fswriter%s:" % self.commit_serial)
        return self

    def __exit__(
        self,
        cls: type[BaseException] | None,
        val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
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
        return None

    def commit(self, commit_serial: int) -> LazyChangesFormatter:
        records = self.records
        assert records is not None
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
                new_typedkeys.append(
                    dict(
                        relpath=record.key.relpath,
                        ulid=int(ULID.new()),
                        keytype=record.key.key_name,
                        serial=commit_serial,
                    )
                )
            else:
                updated_typedkeys.append(
                    dict(
                        b_relpath=record.key.relpath,
                        b_keytype=record.key.key_name,
                        b_serial=commit_serial,
                        b_back_serial=record.back_serial,
                    )
                )
        self.conn._db_write_typedkeys(new_typedkeys, updated_typedkeys)
        del new_typedkeys, updated_typedkeys
        self.conn._write_records(commit_serial, records, rel_renames)
        if self.io_file:
            (files_commit, files_del) = self.conn._write_dirty_files()
        else:
            (files_commit, files_del) = ([], [])
        self.conn.commit()
        self.storage.last_commit_timestamp = time.time()
        return LazyChangesFormatter(
            {(x.key.key_name, x.key.relpath) for x in records}, files_commit, files_del
        )

    def records_set(self, records: Sequence[Record]) -> None:
        assert records is not None
        assert not hasattr(self, "records")
        self.records = records

    def rollback(self) -> None:
        with suppress(AttributeError):
            del self.records
        self.conn.rollback()

    def set_rel_renames(self, rel_renames: Sequence[str]) -> None:
        assert rel_renames is not None
        assert self.rel_renames == []
        self.rel_renames = rel_renames


class BaseStorage:
    _keys: dict[str, IKeyFSKey]

    def __init__(
        self,
        basedir: Path,
        *,
        notify_on_commit: Callable,
        cache_size: int,
        settings: dict,  # noqa: ARG002
    ) -> None:
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
        self._keys = {}
        self.last_commit_timestamp = time.time()

    def define_tables(
        self, metadata_obj: sa.MetaData, binary_type: type[_Binary]
    ) -> dict:
        return dict(
            relpath_ulid_table=sa.Table(
                "relpath_ulid",
                metadata_obj,
                sa.Column("relpath", sa.String, index=True, nullable=False),
                sa.Column("ulid", sa.BigInteger, nullable=False),
                sa.Column("keytype", sa.String, index=True, nullable=False),
                sa.Column("serial", sa.Integer, nullable=False),
            ),
            renames_table=sa.Table(
                "renames",
                metadata_obj,
                sa.Column("serial", sa.Integer, primary_key=True),
                sa.Column("data", binary_type, nullable=False),
            ),
            ulid_changelog_table=sa.Table(
                "ulid_changelog",
                metadata_obj,
                sa.Column("ulid", sa.BigInteger, index=True, nullable=False),
                sa.Column("serial", sa.Integer, index=True, nullable=False),
                sa.Column("back_serial", sa.Integer, nullable=False),
                sa.Column("value", binary_type, nullable=True),
            ),
            ulid_latest_serial_table=sa.Table(
                "ulid_latest_serial",
                metadata_obj,
                sa.Column("ulid", sa.BigInteger, primary_key=True),
                sa.Column("serial", sa.Integer, nullable=False),
            ),
        )

    def register_key(self, key: IKeyFSKey) -> None:
        self._keys[key.key_name] = key


def cache_metrics(storage: BaseStorage) -> list[tuple[str, str, object]]:
    result: list[tuple[str, str, object]] = []
    changelog_cache = getattr(storage, "_changelog_cache", None)
    relpath_cache = getattr(storage, "_relpath_cache", None)
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
    result.extend(
        [
            ("devpi_server_storage_cache_evictions", "counter", evictions),
            ("devpi_server_storage_cache_hits", "counter", hits),
            ("devpi_server_storage_cache_lookups", "counter", lookups),
            ("devpi_server_storage_cache_misses", "counter", misses),
            ("devpi_server_storage_cache_size", "gauge", size),
        ]
    )
    if changelog_cache:
        result.extend(
            [
                (
                    "devpi_server_changelog_cache_evictions",
                    "counter",
                    changelog_cache.evictions,
                ),
                ("devpi_server_changelog_cache_hits", "counter", changelog_cache.hits),
                (
                    "devpi_server_changelog_cache_lookups",
                    "counter",
                    changelog_cache.lookups,
                ),
                (
                    "devpi_server_changelog_cache_misses",
                    "counter",
                    changelog_cache.misses,
                ),
                ("devpi_server_changelog_cache_size", "gauge", changelog_cache.size),
                (
                    "devpi_server_changelog_cache_items",
                    "gauge",
                    len(changelog_cache.data) if changelog_cache.data else 0,
                ),
            ]
        )
    if relpath_cache:
        result.extend(
            [
                (
                    "devpi_server_relpath_cache_evictions",
                    "counter",
                    relpath_cache.evictions,
                ),
                ("devpi_server_relpath_cache_hits", "counter", relpath_cache.hits),
                (
                    "devpi_server_relpath_cache_lookups",
                    "counter",
                    relpath_cache.lookups,
                ),
                ("devpi_server_relpath_cache_misses", "counter", relpath_cache.misses),
                ("devpi_server_relpath_cache_size", "gauge", relpath_cache.size),
                (
                    "devpi_server_relpath_cache_items",
                    "gauge",
                    len(relpath_cache.data) if relpath_cache.data else 0,
                ),
            ]
        )
    return result
