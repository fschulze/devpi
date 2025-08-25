from __future__ import annotations

from .filestore_fs import LazyChangesFormatter
from .fileutil import dumps
from .fileutil import loads
from .interfaces import IWriter
from .keyfs_types import KeyData
from .keyfs_types import LocatedKey
from .keyfs_types import PatternedKey
from .keyfs_types import ULID
from .keyfs_types import ULIDKey
from .log import thread_pop_log
from .log import thread_push_log
from .log import threadlog
from .markers import Absent
from .markers import absent
from .markers import deleted
from .readonly import ReadonlyView
from .readonly import ensure_deeply_readonly
from .sizeof import gettotalsizeof
from collections import defaultdict
from contextlib import suppress
from devpi_common.types import cached_property
from itertools import islice
from repoze.lru import LRUCache
from sqlalchemy.ext.compiler import compiles
from typing import TYPE_CHECKING
from typing import cast
from zope.interface import implementer
import inspect
import sqlalchemy as sa
import time
import warnings


if TYPE_CHECKING:
    from .interfaces import IIOFile
    from .keyfs_types import Record
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from pathlib import Path
    from sqlalchemy.sql.compiler import SQLCompiler
    from sqlalchemy.types import _Binary
    from types import TracebackType
    from typing import Any
    from typing import Callable
    from typing import IO
    from typing_extensions import Self


INSERT_BATCH_SIZE = 10000
MAX_SQL_VARIABLES = 100


warnings.simplefilter("error", sa.exc.SADeprecationWarning)
warnings.simplefilter("error", sa.exc.SAWarning)


class Cache:
    def __init__(self, large_cache_size: int, small_cache_size: int) -> None:
        self._large_cache = LRUCache(large_cache_size)
        self._small_cache = LRUCache(small_cache_size)

    def add_keydata(self, serial: int, keydata: KeyData) -> None:
        cache_key = keydata.key.ulid
        if (
            keydata.value is not deleted
            and gettotalsizeof(keydata.value, maxlen=100000) is None
        ):
            # keydata.value is big, put it in the large cache,
            # which has fewer entries to preserve memory
            self._large_cache.put((serial, cache_key), keydata)
        else:
            # keydata is small
            self._small_cache.put((serial, cache_key), keydata)

    def get(self, key: Any, default: Any = None) -> Any:
        result = self._small_cache.get(key, absent)
        if result is absent:
            result = self._large_cache.get(key, absent)
        if result is absent:
            return default
        return result

    def get_keydata(self, key: Any, default: Any = None) -> Any:
        result = self._small_cache.get(key, absent)
        if isinstance(result, (Absent, ULIDKey)):
            result = self._large_cache.get(key, absent)
        if isinstance(result, (Absent, ULIDKey)):
            return default
        return result

    def put(self, key: Any, value: Any) -> None:
        self._small_cache.put(key, value)

    def put_large(self, key: Any, value: Any) -> None:
        self._large_cache.put(key, value)


class explain(sa.Executable, sa.ClauseElement):
    inherit_cache = True

    def __init__(self, stmt: Any, *, analyze: bool = False) -> None:
        self.statement = stmt
        self.analyze = analyze


@compiles(explain, "postgresql")
def pg_explain(element: explain, compiler: SQLCompiler, **kw: Any) -> str:
    text = "EXPLAIN "
    if element.analyze:
        text += "ANALYZE "
    text += compiler.process(element.statement, **kw)

    return text


@compiles(explain, "sqlite")
def sqlite_explain(element: explain, compiler: SQLCompiler, **kw: Any) -> str:
    text = "EXPLAIN "
    if element.analyze:
        text += "QUERY PLAN "
    text += compiler.process(element.statement, **kw)

    return text


class BaseConnection:
    dirty_files: dict[str, IO | None]
    key_ulid_table: sa.Table
    keytype_name_parent_ulid_table: sa.Table
    renames_table: sa.Table
    ulid_changelog_table: sa.Table
    ulid_info_table: sa.Table
    ulid_latest_serial_table: sa.Table

    def __init__(self, sqlaconn: sa.engine.Connection, storage: BaseStorage) -> None:
        self._sqlaconn = sqlaconn
        self.storage = storage
        for name, member in inspect.getmembers(storage):
            if isinstance(member, sa.Table):
                setattr(self, name, member)
        self.dirty_files = {}
        self._cache = storage._cache
        self._keys = storage._keys

    def _dump(self) -> None:
        # for debugging
        execute = self._sqlaconn.execute
        for table in (
            self.ulid_latest_serial_table,
            self.ulid_info_table,
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

    def analyze(self) -> None:
        self._sqlaconn.execute(sa.text("ANALYZE"))

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
        stmt = (
            sa.select(sa.func.max(self.ulid_latest_serial_table.c.latest_serial))
            .join(
                self.ulid_latest_serial_table,
                self.ulid_info_table.c.ulid == self.ulid_latest_serial_table.c.ulid,
            )
            .where(
                sa.tuple_(
                    self.ulid_info_table.c.keytype,
                    self.ulid_info_table.c.name,
                    self.ulid_info_table.c.parent_ulid,
                )
                == sa.tuple_(
                    sa.literal(key.key_name),
                    sa.literal(key.name),
                    sa.literal(key.query_parent_ulid),
                )
            )
            .group_by(
                self.ulid_info_table.c.keytype,
                self.ulid_info_table.c.name,
                self.ulid_info_table.c.parent_ulid,
            )
        )
        result = self._sqlaconn.execute(stmt).scalar()
        if result is None:
            raise KeyError(key)
        return result

    def _db_write_typedkeys(
        self,
        deleted_typedkeys: Sequence[dict],
        new_typedkeys: Sequence[dict],
        updated_typedkeys: Sequence[dict],
    ) -> None:
        if deleted_typedkeys:
            self._sqlaconn.execute(
                sa.update(self.ulid_info_table).where(
                    self.ulid_info_table.c.ulid == sa.bindparam("b_ulid")
                ),
                deleted_typedkeys,
            )
        if new_typedkeys:
            self._sqlaconn.execute(sa.insert(self.ulid_info_table), new_typedkeys)
            self._sqlaconn.execute(
                sa.insert(self.ulid_latest_serial_table),
                [
                    dict(latest_serial=x["added_at_serial"], ulid=x["ulid"])
                    for x in new_typedkeys
                ],
            )
        if updated_typedkeys:
            stmt = (
                sa.update(self.ulid_latest_serial_table)
                .where(self.ulid_latest_serial_table.c.ulid == sa.bindparam("b_ulid"))
                .values(latest_serial=sa.bindparam("b_serial"))
            )
            self._sqlaconn.execute(stmt, updated_typedkeys)

    def iter_changes_at(self, serial: int) -> Iterator[KeyData]:
        stmt = (
            sa.select(
                self.ulid_info_table.c.keytype,
                self.ulid_info_table.c.name,
                self.ulid_info_table.c.parent_ulid,
                self.ulid_changelog_table.c.back_serial,
            )
            .join(
                self.ulid_changelog_table,
                self.ulid_info_table.c.ulid == self.ulid_changelog_table.c.ulid,
            )
            .where(self.ulid_changelog_table.c.serial == serial)
        )
        relpaths_stmt = stmt.with_only_columns(
            self.ulid_info_table.c.keytype,
            self.ulid_info_table.c.name,
            self.ulid_info_table.c.parent_ulid,
        )
        relpaths_ulid_serials_stmt = self._relpaths_ulid_serials_stmt(
            sa.tuple_(
                self.ulid_info_table.c.keytype,
                self.ulid_info_table.c.name,
                self.ulid_info_table.c.parent_ulid,
            ).in_(relpaths_stmt),
            serial=serial,
            with_deleted=True,
        )
        parent_key_names = self.storage.parent_key_names
        for keydata in self._iter_relpaths_at(
            relpaths_ulid_serials_stmt, serial, with_deleted=True
        ):
            if keydata.key.key_name in parent_key_names:
                self._cache.add_keydata(serial, keydata)
            yield keydata

    def get_raw_changelog_entry(self, serial: int) -> bytes | None:
        data = self._sqlaconn.execute(
            sa.select(self.renames_table.c.data).where(
                self.renames_table.c.serial == serial
            )
        ).scalar()
        if data is None:
            return None
        renames = loads(data)
        changes = [
            (
                c.key.key_name,
                c.key.relpath,
                int(c.key.ulid),
                None if c.key.parent_ulid is None else int(c.key.parent_ulid),
                c.back_serial,
                None if c.value is deleted else c.value,
            )
            for c in self.iter_changes_at(serial)
        ]
        return dumps((changes, renames))

    def iter_rel_renames(self, serial: int) -> Iterator[str]:
        data = self._sqlaconn.execute(
            sa.select(self.renames_table.c.data).where(
                self.renames_table.c.serial == serial
            )
        ).scalar()
        if data is None:
            return
        yield from loads(data)

    def _relpaths_ulid_serials_stmt(
        self,
        *whereclause: sa._ColumnExpressionArgument[bool],  # type: ignore[name-defined]
        serial: int,
        with_deleted: bool,
    ) -> sa.Select:
        full_relpaths_ulid_serials_stmt = sa.select(self.ulid_info_table).where(
            *whereclause, self.ulid_info_table.c.added_at_serial <= serial
        )
        if not with_deleted:
            full_relpaths_ulid_serials_stmt = full_relpaths_ulid_serials_stmt.where(
                sa.or_(
                    self.ulid_info_table.c.deleted_at_serial.is_(None),
                    self.ulid_info_table.c.deleted_at_serial > serial,
                )
            )
        full_relpaths_ulid_serials_cte = full_relpaths_ulid_serials_stmt.cte(
            "full_relpaths_ulid_serials_cte"
        )
        relpath_max_added_at_stmt = sa.select(
            full_relpaths_ulid_serials_cte.c.keytype,
            full_relpaths_ulid_serials_cte.c.name,
            full_relpaths_ulid_serials_cte.c.parent_ulid,
            sa.func.max(full_relpaths_ulid_serials_cte.c.added_at_serial).label(
                "max_added_at_serial"
            ),
        ).group_by(
            full_relpaths_ulid_serials_cte.c.keytype,
            full_relpaths_ulid_serials_cte.c.name,
            full_relpaths_ulid_serials_cte.c.parent_ulid,
        )
        relpath_max_added_at_sq = relpath_max_added_at_stmt.subquery(
            "relpath_max_added_at_sq"
        )
        parent_ulid_stmt = sa.select(
            full_relpaths_ulid_serials_cte.c.parent_ulid.distinct().label("parent_ulid")
        ).join_from(
            full_relpaths_ulid_serials_cte,
            self.ulid_changelog_table,
            full_relpaths_ulid_serials_cte.c.parent_ulid
            == self.ulid_changelog_table.c.ulid,
        )
        parent_ulid_sq = parent_ulid_stmt.subquery("parent_ulid_sq")
        parent_ulid_latest_serial_stmt = (
            sa.select(
                parent_ulid_sq.c.parent_ulid,
                sa.func.max(self.ulid_changelog_table.c.serial).label(
                    "latest_parent_serial"
                ),
            )
            .join_from(
                parent_ulid_sq,
                self.ulid_changelog_table,
                parent_ulid_sq.c.parent_ulid == self.ulid_changelog_table.c.ulid,
            )
            .where(self.ulid_changelog_table.c.serial <= serial)
            .group_by(parent_ulid_sq.c.parent_ulid)
        )
        parent_ulid_latest_serial_sq = parent_ulid_latest_serial_stmt.subquery(
            "parent_ulid_latest_serial_sq"
        )
        return (
            sa.select(
                full_relpaths_ulid_serials_cte.c.keytype,
                full_relpaths_ulid_serials_cte.c.name,
                full_relpaths_ulid_serials_cte.c.parent_ulid,
                parent_ulid_latest_serial_sq.c.latest_parent_serial,
                full_relpaths_ulid_serials_cte.c.ulid,
            )
            .join(
                relpath_max_added_at_sq,
                sa.and_(
                    sa.tuple_(
                        full_relpaths_ulid_serials_cte.c.keytype,
                        full_relpaths_ulid_serials_cte.c.name,
                        full_relpaths_ulid_serials_cte.c.parent_ulid,
                    )
                    == sa.tuple_(
                        relpath_max_added_at_sq.c.keytype,
                        relpath_max_added_at_sq.c.name,
                        relpath_max_added_at_sq.c.parent_ulid,
                    ),
                    full_relpaths_ulid_serials_cte.c.added_at_serial
                    == relpath_max_added_at_sq.c.max_added_at_serial,
                ),
            )
            .outerjoin(
                parent_ulid_latest_serial_sq,
                full_relpaths_ulid_serials_cte.c.parent_ulid
                == parent_ulid_latest_serial_sq.c.parent_ulid,
            )
        )

    def _iter_relpaths_at(
        self, relpaths_ulid_serials_stmt: sa.Select, serial: int, *, with_deleted: bool
    ) -> Iterator[KeyData]:
        execute = self._sqlaconn.execute
        relpaths_ulid_serials_cte = relpaths_ulid_serials_stmt.cte(
            "relpaths_ulid_serials_cte"
        )
        ulid_max_serial_stmt = (
            sa.select(
                self.ulid_changelog_table.c.ulid,
                sa.func.max(self.ulid_changelog_table.c.serial).label(
                    "ulid_max_serial"
                ),
            )
            .join(
                relpaths_ulid_serials_cte,
                sa.and_(
                    self.ulid_changelog_table.c.ulid
                    == relpaths_ulid_serials_cte.c.ulid,
                    self.ulid_changelog_table.c.serial <= serial,
                ),
            )
            .group_by(self.ulid_changelog_table.c.ulid)
        )
        ulid_max_serial_sq = ulid_max_serial_stmt.subquery("ulid_max_serial_sq")
        ulid_changelog_stmt = (
            sa.select(
                relpaths_ulid_serials_cte.c.keytype,
                relpaths_ulid_serials_cte.c.name,
                relpaths_ulid_serials_cte.c.parent_ulid,
                self.ulid_changelog_table,
                relpaths_ulid_serials_cte.c.latest_parent_serial,
            )
            .join_from(
                relpaths_ulid_serials_cte,
                self.ulid_changelog_table,
                relpaths_ulid_serials_cte.c.ulid == self.ulid_changelog_table.c.ulid,
            )
            .join(
                ulid_max_serial_sq,
                sa.and_(
                    relpaths_ulid_serials_cte.c.ulid == ulid_max_serial_sq.c.ulid,
                    self.ulid_changelog_table.c.serial
                    == ulid_max_serial_sq.c.ulid_max_serial,
                ),
            )
        )
        if not with_deleted:
            ulid_changelog_stmt = ulid_changelog_stmt.where(
                self.ulid_changelog_table.c.value.isnot(None)
            )
        for row in execute(ulid_changelog_stmt).all():
            yield KeyData(
                key=self._ulidkey_for_row(row, serial),
                serial=row.serial,
                back_serial=row.back_serial,
                value=deleted
                if row.value is None
                else ensure_deeply_readonly(loads(row.value)),
            )

    def _ulidkey_for_row(self, row: sa.Row, serial: int) -> ULIDKey:
        key = self._keys[row.keytype]
        parent_ulid = None if row.parent_ulid < 0 else ULID(row.parent_ulid)
        if parent_ulid:
            parent_key = self.get_ulid_at_serial(
                parent_ulid, serial, parent_serial=row.latest_parent_serial
            ).key
            location = parent_key.relpath
            parent_params = parent_key.params
        else:
            location = ""
            parent_key = None
            parent_params = None
        if isinstance(key, PatternedKey):
            return key.make_ulid_key(
                location,
                row.name,
                ULID(row.ulid),
                parent_key=parent_key,
                parent_params=parent_params,
            )
        assert parent_key is None
        assert parent_params is None
        return key.make_ulid_key(ULID(row.ulid))

    def get_key_at_serial(self, key: ULIDKey, serial: int) -> KeyData:
        assert isinstance(key, ULIDKey)
        try:
            return self.get_ulid_at_serial(key.ulid, serial)
        except KeyError as e:
            raise KeyError(key) from e

    def get_ulid_at_serial(
        self, ulid: ULID, serial: int, *, parent_serial: int = -1
    ) -> KeyData:
        assert isinstance(ulid, ULID)
        cache_key = ulid
        result = self._cache.get_keydata((serial, cache_key), absent)
        if result is absent and parent_serial >= 0:
            result = self._cache.get_keydata((parent_serial, cache_key), absent)
        if result is absent:
            fetch_serial = parent_serial if parent_serial >= 0 else serial
            if fetch_serial < 0:
                raise KeyError(ulid)
            relpaths_ulid_serials_stmt = self._relpaths_ulid_serials_stmt(
                self.ulid_info_table.c.ulid == int(ulid),
                serial=fetch_serial,
                with_deleted=True,
            )
            results = list(
                self._iter_relpaths_at(
                    relpaths_ulid_serials_stmt, fetch_serial, with_deleted=True
                )
            )
            if not results:
                raise KeyError(ulid)
            (result,) = results
            self._cache.add_keydata(fetch_serial, result)
        return result

    emptyset = cast("set", frozenset())

    def _relpaths_ulid_serials_stmt_for_keys(  # noqa: PLR0912
        self,
        keys: Iterable[LocatedKey | PatternedKey | ULIDKey],
        at_serial: int,
        *,
        skip_ulid_keys: set[ULIDKey] = emptyset,
        with_deleted: bool = False,
    ) -> sa.Select:
        execute = self._sqlaconn.execute
        keytypes = {key.key_name for key in keys if isinstance(key, PatternedKey)}
        keytype_name_parent_ulid: set[tuple[str, str, int]] = set()
        key_ulid: set[int] = set()
        for key in keys:
            if key.key_name in keytypes:
                continue
            if isinstance(key, LocatedKey):
                keytype_name_parent_ulid.add(
                    (key.key_name, key.name, key.query_parent_ulid)
                )
            elif isinstance(key, ULIDKey):
                key_ulid.add(int(key.ulid))
        clauses = []
        if keytypes:
            clauses.append(self.ulid_info_table.c.keytype.in_(keytypes))
        is_large_query = (
            len(keytype_name_parent_ulid) + len(key_ulid)
        ) > MAX_SQL_VARIABLES
        if keytype_name_parent_ulid:
            keytype_name_parent_ulid_in: sa.Select | Iterator[sa.Tuple]
            if is_large_query:
                keytype_name_parent_ulid_iter = iter(keytype_name_parent_ulid)
                del keytype_name_parent_ulid
                execute(
                    sa.schema.CreateTable(
                        self.keytype_name_parent_ulid_table, if_not_exists=True
                    )
                )
                execute(self.keytype_name_parent_ulid_table.delete())
                while True:
                    keytype_name_parent_ulid_batch = list(
                        islice(keytype_name_parent_ulid_iter, INSERT_BATCH_SIZE)
                    )
                    if not keytype_name_parent_ulid_batch:
                        break
                    execute(
                        self.keytype_name_parent_ulid_table.insert(),
                        [
                            dict(keytype=key_name, name=name, parent_ulid=parent_ulid)
                            for key_name, name, parent_ulid in keytype_name_parent_ulid_batch
                        ],
                    )
                    del keytype_name_parent_ulid_batch
                keytype_name_parent_ulid_in = sa.select(
                    self.keytype_name_parent_ulid_table.c.keytype,
                    self.keytype_name_parent_ulid_table.c.name,
                    self.keytype_name_parent_ulid_table.c.parent_ulid,
                )
            else:
                keytype_name_parent_ulid_in = (
                    sa.tuple_(
                        sa.literal(key_name),
                        sa.literal(name),
                        sa.literal(parent_ulid),
                    )
                    for (key_name, name, parent_ulid) in keytype_name_parent_ulid
                )
            clauses.append(
                sa.tuple_(
                    self.ulid_info_table.c.keytype,
                    self.ulid_info_table.c.name,
                    self.ulid_info_table.c.parent_ulid,
                ).in_(keytype_name_parent_ulid_in)
            )
        if key_ulid:
            key_ulid_in: sa.Select | Iterator[sa.Tuple]
            if is_large_query:
                key_ulid_iter = iter(key_ulid)
                del key_ulid
                execute(sa.schema.CreateTable(self.key_ulid_table, if_not_exists=True))
                execute(self.key_ulid_table.delete())
                while True:
                    key_ulid_batch = list(islice(key_ulid_iter, INSERT_BATCH_SIZE))
                    if not key_ulid_batch:
                        break
                    execute(
                        self.key_ulid_table.insert(),
                        [dict(ulid=x) for x in key_ulid_batch],
                    )
                    del key_ulid_batch
                key_ulid_in = sa.select(self.key_ulid_table.c.ulid)
            else:
                key_ulid_in = (
                    sa.tuple_(
                        sa.literal(ulid),
                    )
                    for ulid in key_ulid
                )
            clauses.append(
                sa.tuple_(
                    self.ulid_info_table.c.ulid,
                ).in_(key_ulid_in)
            )
        assert clauses
        whereclauses = [sa.or_(*clauses)]
        if skip_ulid_keys:
            whereclauses.append(
                self.ulid_info_table.c.ulid.notin_(int(x.ulid) for x in skip_ulid_keys)
            )
        return self._relpaths_ulid_serials_stmt(
            *whereclauses, serial=at_serial, with_deleted=with_deleted
        )

    def iter_keys_at_serial(
        self,
        keys: Iterable[LocatedKey | PatternedKey],
        at_serial: int,
        *,
        skip_ulid_keys: set[ULIDKey] = emptyset,
        with_deleted: bool,
    ) -> Iterator[KeyData]:
        stmt = self._relpaths_ulid_serials_stmt_for_keys(
            keys, at_serial, skip_ulid_keys=skip_ulid_keys, with_deleted=with_deleted
        )
        yield from self._iter_relpaths_at(stmt, at_serial, with_deleted=with_deleted)

    def iter_ulidkeys_at_serial(
        self,
        keys: Iterable[LocatedKey | PatternedKey],
        at_serial: int,
        *,
        skip_ulid_keys: set[ULIDKey] = emptyset,
        with_deleted: bool,
    ) -> Iterator[ULIDKey]:
        stmt = self._relpaths_ulid_serials_stmt_for_keys(
            keys, at_serial, skip_ulid_keys=skip_ulid_keys, with_deleted=with_deleted
        )
        for result in self._sqlaconn.execute(stmt):
            yield self._ulidkey_for_row(result, at_serial)

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
        ulid_changelog = [
            dict(
                ulid=int(record.key.ulid),
                serial=serial,
                back_serial=record.back_serial,
                value=None if record.value is deleted else dumps(record.value),
            )
            for record in records
        ]
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
        deleted_typedkeys = []
        new_typedkeys = []
        updated_typedkeys = []
        for record in records:
            if record.back_serial is None:
                raise RuntimeError
            assert not isinstance(record.value, ReadonlyView), record.value
            if record.key != record.old_key:
                new_typedkeys.append(
                    dict(
                        name=record.key.name,
                        ulid=int(record.key.ulid),
                        parent_ulid=record.key.query_parent_ulid,
                        keytype=record.key.key_name,
                        added_at_serial=commit_serial,
                    )
                )
            else:
                updated_typedkeys.append(
                    dict(
                        b_ulid=int(record.key.ulid),
                        b_keytype=record.key.key_name,
                        b_serial=commit_serial,
                        b_back_serial=record.back_serial,
                    )
                )
                if record.value is deleted:
                    deleted_typedkeys.append(
                        dict(
                            b_ulid=int(record.key.ulid),
                            deleted_at_serial=commit_serial
                            if record.value is deleted
                            else None,
                        )
                    )
        self.conn._db_write_typedkeys(
            deleted_typedkeys, new_typedkeys, updated_typedkeys
        )
        del new_typedkeys, updated_typedkeys
        self.conn._write_records(commit_serial, records, rel_renames)
        if self.io_file:
            (files_commit, files_del) = self.conn._write_dirty_files()
        else:
            (files_commit, files_del) = ([], [])
        analyze_frequency = (
            100 if commit_serial < 1000 else 1000 if commit_serial < 20000 else 10000
        )
        if ((commit_serial % analyze_frequency) == 0) or len(records) > 10000:
            self.conn.analyze()
        self.conn.commit()
        self.storage.last_commit_timestamp = time.time()
        return LazyChangesFormatter(
            {(x.key.key_name, x.key.name, x.key.parent_ulid) for x in records},
            files_commit,
            files_del,
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
    _keys: dict[str, LocatedKey | PatternedKey]

    def __init__(
        self, basedir: Path, *, notify_on_commit: Callable, settings: dict
    ) -> None:
        self.basedir = basedir
        self._notify_on_commit = notify_on_commit
        self._cache = Cache(
            settings.get("large_cache_size", 500),
            settings.get("small_cache_size", 9500),
        )
        self._keys = {}
        self.last_commit_timestamp = time.time()

    def define_tables(
        self, metadata_obj: sa.MetaData, binary_type: type[_Binary]
    ) -> dict:
        tables = dict(
            key_ulid_table=sa.Table(
                "key_ulid",
                metadata_obj,
                sa.Column("ulid", sa.BigInteger),
                prefixes=["TEMPORARY"],
            ),
            keytype_name_parent_ulid_table=sa.Table(
                "keytype_name_parent_ulid",
                metadata_obj,
                sa.Column("keytype", sa.String),
                sa.Column("name", sa.String),
                sa.Column("parent_ulid", sa.BigInteger),
                prefixes=["TEMPORARY"],
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
                sa.Column("ulid", sa.BigInteger, nullable=False),
                sa.Column("serial", sa.Integer, index=True, nullable=False),
                sa.Column("back_serial", sa.Integer, nullable=False),
                sa.Column("value", binary_type, nullable=True),
            ),
            ulid_info_table=sa.Table(
                "ulid_info",
                metadata_obj,
                sa.Column("ulid", sa.BigInteger, primary_key=True),
                sa.Column("parent_ulid", sa.BigInteger, index=True, nullable=False),
                sa.Column("name", sa.String, nullable=False),
                sa.Column("keytype", sa.String, nullable=False),
                sa.Column("added_at_serial", sa.Integer, nullable=False),
                sa.Column("deleted_at_serial", sa.Integer, nullable=True),
            ),
            ulid_latest_serial_table=sa.Table(
                "ulid_latest_serial",
                metadata_obj,
                sa.Column("ulid", sa.BigInteger, primary_key=True),
                sa.Column("latest_serial", sa.Integer, nullable=False),
            ),
        )
        ulid_info_table = tables["ulid_info_table"]
        sa.Index(
            "ix_ulid_info_keytype_name_parent_ulid",
            ulid_info_table.c.keytype,
            ulid_info_table.c.name,
            ulid_info_table.c.parent_ulid,
        )
        ulid_changelog_table = tables["ulid_changelog_table"]
        sa.Index(
            "ix_ulid_changelog_ulid_serial",
            ulid_changelog_table.c.ulid,
            ulid_changelog_table.c.serial,
        )
        return tables

    @classmethod
    def process_settings(cls, settings: dict[str, Any]) -> dict[str, Any]:
        for key in ("large_cache_size", "small_cache_size"):
            if key not in settings:
                continue
            settings[key] = int(settings[key])
        return settings

    def register_key(self, key: LocatedKey | PatternedKey) -> None:
        self._keys[key.key_name] = key

    @cached_property
    def parent_key_names(self) -> frozenset[str]:
        keys = set(self._keys.values())
        parent_keys: dict[PatternedKey | None, set[PatternedKey]] = defaultdict(set)
        stack: list[PatternedKey | None] = [None]
        while stack:
            parent_key = stack.pop()
            for key in set(keys):
                if not isinstance(key, PatternedKey):
                    continue
                if key.parent_key != parent_key:
                    continue
                parent_keys[parent_key].add(key)
                stack.append(key)
        return frozenset(x.key_name for x in keys if x in parent_keys)


def cache_metrics(storage: BaseStorage) -> list[tuple[str, str, object]]:
    result: list[tuple[str, str, object]] = []
    large_cache = storage._cache._large_cache
    small_cache = storage._cache._small_cache
    if large_cache is None and small_cache is None:
        return result
    if large_cache:
        result.extend(
            [
                (
                    "devpi_server_large_cache_evictions",
                    "counter",
                    large_cache.evictions,
                ),
                ("devpi_server_large_cache_hits", "counter", large_cache.hits),
                ("devpi_server_large_cache_lookups", "counter", large_cache.lookups),
                ("devpi_server_large_cache_misses", "counter", large_cache.misses),
                ("devpi_server_large_cache_size", "gauge", large_cache.size),
                (
                    "devpi_server_large_cache_items",
                    "gauge",
                    len(large_cache.data) if large_cache.data else 0,
                ),
            ]
        )
    if small_cache:
        result.extend(
            [
                (
                    "devpi_server_small_cache_evictions",
                    "counter",
                    small_cache.evictions,
                ),
                ("devpi_server_small_cache_hits", "counter", small_cache.hits),
                ("devpi_server_small_cache_lookups", "counter", small_cache.lookups),
                ("devpi_server_small_cache_misses", "counter", small_cache.misses),
                ("devpi_server_small_cache_size", "gauge", small_cache.size),
                (
                    "devpi_server_small_cache_items",
                    "gauge",
                    len(small_cache.data) if small_cache.data else 0,
                ),
            ]
        )
    return result
