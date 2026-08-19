from __future__ import annotations

from .keyfs import KeyfsTimeoutError
from .keyfs_sqla_base import BaseStorage
from .log import threadlog
from .mythread import current_thread
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import overload
import contextlib
import sqlalchemy as sa
import sqlite3
import sys
import time
import traceback
import weakref


if TYPE_CHECKING:
    from .interfaces import IStorageConnection
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path
    from typing import Any
    from typing import Literal


_write_thread: Any = None


def log_write_thread_trace() -> None:
    if _write_thread is None:
        return
    frame = sys._current_frames()[_write_thread.ident]
    st = "".join(traceback.format_stack(frame))
    request_log_frame = frame
    request_info = ""
    request_tag = "[unknown request]"
    request = None
    while request_log_frame.f_back:
        if "request_log_handler" in request_log_frame.f_code.co_name:
            request_tag = str(request_log_frame.f_locals.get("tag"))
            request = request_log_frame.f_locals.get("request")
        request_log_frame = request_log_frame.f_back
    if request is not None:
        request_info = f" in {request_tag} {request.method} {request.url}"
    threadlog.error(
        "Timeout while waiting for write transaction%s, currently writing thread (%s) trace:\n%s",
        request_info,
        _write_thread.name,
        st,
    )


class LiteBaseStorage(BaseStorage):
    Connection: Callable[[sa.engine.Connection, LiteBaseStorage], IStorageConnection]
    db_filename: str

    def __init__(
        self, basedir: Path, *, notify_on_commit: Callable, settings: dict
    ) -> None:
        super().__init__(basedir, notify_on_commit=notify_on_commit, settings=settings)
        self.sqlpath = self.basedir / self.db_filename
        self.ro_engine = sa.create_engine(
            self._url(mode="ro"),
            echo=False,
            connect_args={"isolation_level": None},
            poolclass=sa.NullPool,
        )
        weakref.finalize(self, self.ro_engine.dispose)
        self.rw_engine = sa.create_engine(
            self._url(mode="rw"),
            echo=False,
            connect_args={"isolation_level": None},
            poolclass=sa.NullPool,
        )
        weakref.finalize(self, self.rw_engine.dispose)
        self.ensure_tables_exist()

    @abstractmethod
    def ensure_tables_exist(self) -> None:
        raise NotImplementedError

    def _execute_conn_pragmas(self, conn: sa.Connection) -> None:
        c = conn.connection.cursor()
        c.execute("PRAGMA busy_timeout=1000")
        c.execute("PRAGMA cache_size = 200000")
        c.close()

    @classmethod
    def exists(cls, basedir: Path, settings: dict) -> bool:  # noqa: ARG003
        sqlpath = basedir / cls.db_filename
        return sqlpath.exists()

    @overload
    def get_connection(
        self, *, closing: Literal[True], write: bool = False, timeout: float = 30
    ) -> AbstractContextManager[IStorageConnection]:
        pass

    @overload
    def get_connection(
        self, *, closing: Literal[False], write: bool = False, timeout: float = 30
    ) -> IStorageConnection:
        pass

    def get_connection(
        self, *, closing: bool = True, write: bool = False, timeout: float = 30
    ) -> IStorageConnection | AbstractContextManager[IStorageConnection]:
        start_time = time.monotonic()
        engine = self.rw_engine if write else self.ro_engine
        sqlaconn = engine.connect()
        self._execute_conn_pragmas(sqlaconn)
        if write:
            global _write_thread  # noqa: PLW0603 - for debugging only
            log_delay: float = 2
            thread = current_thread()
            while 1:
                try:
                    sqlaconn.execute(sa.text("begin immediate"))
                    break
                except sa.exc.OperationalError as e:
                    sqlite_errorcode = (
                        e.orig.sqlite_errorcode
                        if isinstance(e.orig, sqlite3.OperationalError)
                        else None
                    )
                    # another thread may be writing, give it a chance to finish
                    time.sleep(0.1)
                    if hasattr(thread, "exit_if_shutdown"):
                        thread.exit_if_shutdown()
                    elapsed = time.monotonic() - start_time
                    if elapsed >= log_delay:
                        threadlog.warn(
                            "Waiting on database connection for %.6f seconds (SQLite error code: %s)",
                            log_delay,
                            sqlite_errorcode,
                        )
                        log_delay = log_delay * 1.5
                    if elapsed > timeout:
                        # if it takes this long, something is wrong
                        msg = f"Timeout after {int(elapsed)} seconds (SQLite error code: {sqlite_errorcode})."
                        with contextlib.suppress(Exception):
                            log_write_thread_trace()
                        raise KeyfsTimeoutError(msg) from e
            _write_thread = thread
        elapsed = time.monotonic() - start_time
        threadlog.debug(
            "Got %s transaction after %.6f seconds",
            "write" if write else "read",
            elapsed,
        )
        conn = self.Connection(sqlaconn, self)
        if closing:
            return contextlib.closing(conn)
        return conn

    def _url(self, *, mode: str) -> str:
        return f"sqlite+pysqlite:///file:{self.sqlpath}?mode={mode}&timeout=30&uri=true"
