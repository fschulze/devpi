"""
filesystem key/value storage with support for storing and retrieving
basic python types based on parameterizable keys.  Multiple
read Transactions can execute concurrently while at most one
write Transaction is ongoing.  Each Transaction will see a consistent
view of key/values referring to the point in time it was started,
independent from any future changes.
"""
from __future__ import annotations

from . import mythread
from .compat import get_annotations
from .filestore import FileEntry
from .fileutil import read_int_from_file
from .fileutil import write_int_to_file
from .interfaces import IStorageConnection
from .interfaces import IWriter
from .keyfs_types import KeyData
from .keyfs_types import LocatedKey
from .keyfs_types import PatternedKey
from .keyfs_types import Record
from .keyfs_types import SearchKey
from .keyfs_types import ULIDKey
from .log import thread_change_log_prefix
from .log import thread_pop_log
from .log import thread_push_log
from .log import threadlog
from .markers import Absent
from .markers import Deleted
from .markers import absent
from .markers import deleted
from .model import RootModel
from .readonly import ensure_deeply_readonly
from .readonly import get_mutable_deepcopy
from .readonly import is_deeply_readonly
from attrs import frozen
from devpi_common.types import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from typing import overload
import contextlib
import errno
import time


if TYPE_CHECKING:
    from .keyfs_types import KeyFSTypes
    from .keyfs_types import KeyFSTypesRO
    from .keyfs_types import StorageInfo
    from .keyfs_types import ULID
    from .log import TagLogger
    from .mythread import MyThread
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing import Any
    from typing import Callable


class KeyfsTimeoutError(TimeoutError):
    pass


class MissingFileException(Exception):
    def __init__(self, key, serial, missing_source):
        msg = f"missing file '{key}' at serial {serial}"
        if missing_source:
            msg = f"{msg} {missing_source}"
        super().__init__(msg)
        self.key = key
        self.serial = serial


class TxNotificationThread:
    _get_ixconfig_cache: dict[tuple[str, str], dict | None]
    _on_key_change: dict[str, list[Callable]]
    log: TagLogger
    thread: MyThread

    def __init__(self, keyfs):
        self.keyfs = keyfs
        self.cv_new_event_serial = mythread.threading.Condition()
        self.event_serial_path = str(self.keyfs.base_path / ".event_serial")
        self.event_serial_in_sync_at = None
        self._on_key_change = {}

    def on_key_change(self, key, subscriber):
        if mythread.has_active_thread(self):
            raise RuntimeError(
                "cannot register handlers after thread has started")
        key_name = key if isinstance(key, str) else key.key_name
        assert isinstance(key_name, str), key_name
        if not any(
            "KeyChangeEvent" in str(v) for v in get_annotations(subscriber).values()
        ):
            msg = (
                f"The event subscriber {subscriber!r} has no KeyChangeEvent annotation"
            )
            raise RuntimeError(msg)
        self._on_key_change.setdefault(key_name, []).append(subscriber)

    def wait_event_serial(self, serial):
        with threadlog.around("info", "waiting for event-serial %s", serial):
            with self.cv_new_event_serial:
                while serial > self.read_event_serial():
                    self.cv_new_event_serial.wait()

    def read_event_serial(self):
        # the disk serial is kept one higher because pre-2.1.2
        # "event_serial" pointed to the "next event serial to be
        # processed" instead of the now "last processed event serial"
        return read_int_from_file(self.event_serial_path, 0) - 1

    def get_event_serial_timestamp(self):
        f = Path(self.event_serial_path)
        retries = 5
        while retries:
            try:
                return f.stat().st_mtime
            except FileNotFoundError:
                break
            except OSError as e:
                if e.errno not in (errno.EBUSY, errno.ETXTBSY):
                    raise
                retries -= 1
                if not retries:
                    raise
                # let other threads work
                time.sleep(0.001)
        return None

    def write_event_serial(self, event_serial):
        write_int_to_file(event_serial + 1, self.event_serial_path)

    def thread_shutdown(self):
        pass

    def tick(self):
        event_serial = self.read_event_serial()
        while event_serial < self.keyfs.get_current_serial():
            self.thread.exit_if_shutdown()
            event_serial += 1
            self._execute_hooks(event_serial)
            with self.cv_new_event_serial:
                self.write_event_serial(event_serial)
                self.cv_new_event_serial.notify_all()
        serial = self.keyfs.get_current_serial()
        if event_serial >= serial:
            if event_serial == serial:
                self.event_serial_in_sync_at = time.time()
            self.keyfs.wait_tx_serial(
                serial + 1,
                recheck_callback=self.thread.exit_if_shutdown)

    def thread_run(self):
        self.log = thread_push_log("[NOTI]")
        while 1:
            try:
                self.tick()
            except mythread.Shutdown:
                raise
            except MissingFileException as e:
                self.log.warning(
                    "Waiting for file %s in event serial %s", e.key, e.serial
                )
                self.thread.sleep(5)
            except Exception:
                self.log.exception(
                    "Unhandled exception in notification thread.")
                self.thread.sleep(1.0)

    def get_ixconfig(
        self, entry: FileEntry, at_serial: int
    ) -> dict[str, object] | None:
        user = entry.user
        index = entry.index
        if getattr(self, "_get_ixconfig_cache_serial", None) != at_serial:
            self._get_ixconfig_cache = {}
            self._get_ixconfig_cache_serial = at_serial
        cache_key = (user, index)
        if cache_key in self._get_ixconfig_cache:
            return self._get_ixconfig_cache[cache_key]
        with self.keyfs.read_transaction():
            key = self.keyfs.get_key('USER')(user=user)
            value = key.get()
            key = self.keyfs.get_key("INDEX")(user=user, index=index)
            ixconfig = key.get()
        if value is None:
            # the user doesn't exist anymore
            self._get_ixconfig_cache[cache_key] = None
            return None
        if ixconfig is None:
            # the index doesn't exist anymore
            self._get_ixconfig_cache[cache_key] = None
            return None
        self._get_ixconfig_cache[cache_key] = ixconfig
        return ixconfig

    def skip_by_index_config(self, ixconfig: dict[str, object] | None) -> bool:
        if ixconfig is None:
            # the index doesn't exist (anymore)
            return True
        # check if the index uses external URLs now
        return ixconfig.get("type") == "mirror" and bool(
            ixconfig.get("mirror_use_external_urls", False)
        )

    def check_file_change(self, change: KeyData, event_serial: int) -> None:
        missing_source = ""
        key = self.keyfs.get_key_instance(change.key.key_name, change.key.relpath)
        entry = FileEntry(key, change.value)
        if entry.deleted_or_never_fetched or self.skip_by_index_config(
            self.get_ixconfig(entry, event_serial)
        ):
            # file/index removed or mirror related skip
            return
        with self.keyfs.filestore_transaction():
            if entry.file_exists():
                # all good
                return
        # the file is missing, check whether we can ignore it
        serial = self.keyfs.get_current_serial()
        if event_serial < serial:
            # there are newer serials existing
            with self.keyfs.read_transaction() as tx:
                current_val = tx.get(key)
            if current_val is None:
                # entry was deleted
                return
            current_entry = FileEntry(key, current_val)
            if current_entry.deleted_or_never_fetched:
                # the file was removed at some point
                return
            current_ixconfig = self.get_ixconfig(entry, serial)
            if self.skip_by_index_config(current_ixconfig):
                return
            if (
                current_ixconfig
                and current_ixconfig.get("type") == "mirror"
                and current_entry.project is None
            ):
                # this is an old mirror entry with no
                # project info, so this can be ignored
                return
            missing_source = f"current_entry.meta {current_entry.meta!r}"
        missing_source = (
            missing_source if missing_source else f"entry.meta {entry.meta!r}"
        )
        raise MissingFileException(change.key, event_serial, missing_source)

    def _execute_hooks(self, event_serial: int, *, raising: bool = False) -> None:
        self.log.debug("calling hooks for tx%s", event_serial)
        with self.keyfs.get_connection() as conn:
            changes: list[KeyData] = list(conn.iter_changes_at(event_serial))
        # we first check for missing files before we call subscribers
        for change in changes:
            if change.key.key_name in ("STAGEFILE", "PYPIFILE_NOMD5"):
                self.check_file_change(change, event_serial)
        # all files exist or are deleted in a later serial,
        # call subscribers now
        for change in changes:
            subscribers = self._on_key_change.get(change.key.key_name, [])
            if not subscribers:
                continue
            ev = KeyChangeEvent(data=change, at_serial=event_serial)
            for sub in subscribers:
                subname = getattr(sub, "__name__", sub)
                self.log.debug(
                    "%s(key=%r, data=%r at_serial=%r",
                    subname,
                    change.key,
                    change,
                    event_serial,
                )
                try:
                    sub(ev)
                except Exception:
                    if raising:
                        raise
                    self.log.exception(
                        "calling %s failed, serial=%s", sub, event_serial
                    )

        self.log.debug("finished calling all hooks for tx%s", event_serial)


class KeyFS:
    """ singleton storage object. """

    _import_subscriber: Callable | None
    _keys: dict[str, LocatedKey | PatternedKey]

    class ReadOnly(Exception):
        """ attempt to open write transaction while in readonly mode. """

    def __init__(
        self,
        basedir: Path,
        storage_info: StorageInfo,
        *,
        io_file_factory: Callable | None = None,
        readonly: bool = False,
    ) -> None:
        self.base_path = Path(basedir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._keys = {}
        self._threadlocal = mythread.threading.local()
        self._cv_new_transaction = mythread.threading.Condition()
        self._import_subscriber = None
        self.notifier = TxNotificationThread(self)
        self._storage = storage_info.storage_factory(
            self.base_path,
            notify_on_commit=self._notify_on_commit,
            settings={} if storage_info.settings is None else storage_info.settings,
        )
        self.io_file_factory = io_file_factory
        self._readonly = readonly

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.base_path}>"

    def get_connection(self, *, closing=True, write=False, timeout=30):
        conn = self._storage.get_connection(closing=False, write=write, timeout=timeout)
        conn = IStorageConnection(conn)
        if closing:
            return contextlib.closing(conn)
        return conn

    def finalize_init(self):
        if self.io_file_factory is None:
            return
        with self.get_connection() as conn:
            io_file = self.io_file_factory(conn)
            io_file.perform_crash_recovery()

    def import_changes(self, serial: int, changes: Sequence[KeyData]) -> None:
        with contextlib.ExitStack() as cstack:
            conn = cstack.enter_context(self.get_connection(write=True))
            io_file = (
                None
                if self.io_file_factory is None
                else cstack.enter_context(self.io_file_factory(conn))
            )
            fswriter = IWriter(cstack.enter_context(conn.write_transaction(io_file)))
            next_serial = conn.last_changelog_serial + 1
            assert next_serial == serial, (next_serial, serial)
            records: list[Record] = []
            subscriber_changes: dict[ULIDKey, tuple[KeyFSTypesRO | Deleted, int]] = {}
            for change in changes:
                if not isinstance(change, KeyData):
                    raise TypeError
                old_key: ULIDKey | Absent
                old_val: KeyFSTypesRO | Absent | Deleted
                try:
                    old_data: KeyData = conn.get_key_at_serial(change.key, serial - 1)
                    old_key = old_data.key
                    old_val = old_data.value
                except KeyError:
                    old_key = absent
                    old_val = absent
                subscriber_changes[change.key] = (change.value, change.back_serial)
                records.append(
                    Record(
                        change.key,
                        change.mutable_value,
                        change.back_serial,
                        old_key,
                        old_val,
                    )
                )
            fswriter.records_set(records)
        if callable(self._import_subscriber):
            with self.read_transaction(at_serial=serial):
                self._import_subscriber(serial, subscriber_changes)

    def subscribe_on_import(self, subscriber):
        assert self._import_subscriber is None
        self._import_subscriber = subscriber

    def _notify_on_commit(self, serial):
        self.release_all_wait_tx()

    def release_all_wait_tx(self):
        with self._cv_new_transaction:
            self._cv_new_transaction.notify_all()

    def wait_tx_serial(self, serial, *, timeout=None, recheck=0.1, recheck_callback=None):
        """ Return True when the transaction with the serial has been committed.
        Return False if it hasn't happened within a specified timeout.
        If timeout was not specified, we'll wait indefinitely.  In any case,
        this method wakes up every "recheck" seconds to query the database
        in case some other process has produced a commit (in-process commits
        are recognized immediately).
        """
        # we presume that even a few concurrent wait_tx_serial() calls
        # won't cause much pressure on the database.  If that assumption
        # is wrong we have to install a thread which does the
        # db-querying and sets the local condition.
        time_spent = 0

        # recheck time should never be higher than the timeout
        if timeout is not None and recheck > timeout:
            recheck = timeout
        with threadlog.around("debug", "waiting for tx-serial %s", serial):
            with self._cv_new_transaction:
                with self.get_connection() as conn:
                    while serial > conn.db_read_last_changelog_serial():
                        if timeout is not None and time_spent >= timeout:
                            return False
                        self._cv_new_transaction.wait(timeout=recheck)
                        time_spent += recheck
                        if recheck_callback is not None:
                            recheck_callback()
                    return True

    def get_next_serial(self):
        return self.get_current_serial() + 1

    def get_current_serial(self):
        tx = getattr(self._threadlocal, "tx", None)
        if tx is not None:
            return tx.conn.last_changelog_serial
        with self.get_connection(write=False) as conn:
            return conn.last_changelog_serial

    def get_last_commit_timestamp(self):
        return self._storage.last_commit_timestamp

    @property
    def tx(self):
        return self._threadlocal.tx

    def register_key(self, key):
        allowed_chars = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
        key_name = key.key_name
        assert not set(key_name).difference(allowed_chars), (
            f"Invalid key name: {key_name}"
        )
        if key_name in self._keys:
            raise ValueError("Duplicate registration for key named '%s'" % key_name)
        self._keys[key_name] = key
        setattr(self, key_name, key)
        self._storage.register_key(key)
        return key

    def register_anonymous_key(
        self, key_name: str, parent_key: PatternedKey | None, key_type: type[KeyFSTypes]
    ) -> LocatedKey | PatternedKey:
        return self.register_named_key(key_name, "", parent_key, key_type)

    def register_located_key(
        self, key_name: str, location: str, name: str, key_type: type[KeyFSTypes]
    ) -> LocatedKey:
        return self.register_key(
            LocatedKey(self, key_name, f"{location}/{name}", None, key_type)
        )

    def register_named_key(
        self,
        key_name: str,
        pattern_or_name: str,
        parent_key: PatternedKey | None,
        key_type: type[KeyFSTypes],
    ) -> LocatedKey | PatternedKey:
        key: LocatedKey | PatternedKey
        if parent_key is None and "{" not in pattern_or_name:
            key = LocatedKey(self, key_name, pattern_or_name, None, key_type)
        else:
            assert not isinstance(parent_key, LocatedKey)
            key = PatternedKey(self, key_name, pattern_or_name, parent_key, key_type)
        return self.register_key(key)

    def get_key(self, name):
        return self._keys.get(name)

    def get_key_instance(self, keyname, relpath):
        key = self.get_key(keyname)
        if key is not None and not isinstance(key, LocatedKey):
            key = key(**key.extract_params(relpath))
        return key

    def match_key(self, relpath, *key_candidates):
        for key_candidate in key_candidates:
            if not hasattr(key_candidate, "extract_params"):
                return key_candidate
            if params := key_candidate.extract_params(relpath):
                return key_candidate(**params)
        return None

    def _tx_prefix(self, *, filestore=False):
        tx = self._threadlocal.tx
        mode = "F" if filestore else ("W" if tx.write else "R")
        at_serial = getattr(tx, "at_serial", "")
        return "[%stx%s]" % (mode, at_serial)

    def begin_transaction_in_thread(self, write=False, at_serial=None):
        if write and self._readonly:
            raise self.ReadOnly
        assert not hasattr(self._threadlocal, "tx")
        tx = Transaction(self, write=write, at_serial=at_serial)
        self._threadlocal.tx = tx
        thread_push_log(self._tx_prefix())
        return tx

    def clear_transaction(self):
        prefix = self._tx_prefix()
        del self._threadlocal.tx
        thread_pop_log(prefix)

    def restart_as_write_transaction(self):
        if self._readonly:
            raise self.ReadOnly
        tx = self.tx
        if tx.write:
            raise RuntimeError("Can't restart a write transaction.")
        old_prefix = self._tx_prefix()
        tx.restart(write=True)
        thread_change_log_prefix(self._tx_prefix(), old_prefix)

    def restart_read_transaction(self):
        tx = self.tx
        if tx.write:
            raise RuntimeError("Can only restart a read transaction.")
        if tx.at_serial == tx.conn.db_read_last_changelog_serial():
            threadlog.debug(
                "already at current serial, not restarting transactions")
            return
        old_prefix = self._tx_prefix()
        tx.restart(write=False)
        thread_change_log_prefix(self._tx_prefix(), old_prefix)

    def rollback_transaction_in_thread(self):
        try:
            self._threadlocal.tx.rollback()
        finally:
            self.clear_transaction()

    def commit_transaction_in_thread(self):
        try:
            self._threadlocal.tx.commit()
        finally:
            self.clear_transaction()

    @contextlib.contextmanager
    def _filestore_transaction(self):
        tx = FileStoreTransaction(self)
        self._threadlocal.tx = tx
        prefix = self._tx_prefix(filestore=True)
        thread_push_log(prefix)
        try:
            yield tx
        except BaseException:
            try:
                tx.rollback()
            finally:
                del self._threadlocal.tx
                thread_pop_log(prefix)
            raise
        try:
            tx.commit()
        finally:
            del self._threadlocal.tx
            thread_pop_log(prefix)

    @contextlib.contextmanager
    def filestore_transaction(self):
        """Guarantees a transaction able to directly write files.

        An existing transaction is reused.
        """
        tx = getattr(self._threadlocal, "tx", None)
        if tx is not None:
            yield tx
        else:
            with self._filestore_transaction() as tx:
                yield tx

    @contextlib.contextmanager
    def _transaction(self, *, write=False, at_serial=None):
        tx = self.begin_transaction_in_thread(write=write, at_serial=at_serial)
        try:
            yield tx
        except BaseException:
            self.rollback_transaction_in_thread()
            raise
        self.commit_transaction_in_thread()

    @contextlib.contextmanager
    def read_transaction(self, *, at_serial=None, allow_reuse=False):
        tx = getattr(self._threadlocal, 'tx', None)
        if tx is not None:
            if not allow_reuse:
                raise RuntimeError(
                    "Can't open a read transaction "
                    "within a running transaction.")
            if at_serial is not None and tx.at_serial != at_serial:
                msg = (
                    f"Can't open a read transaction at "
                    f"serial {at_serial!r} from within a running "
                    f"transaction at serial {tx.at_serial!r}.")
                raise RuntimeError(msg)
            yield tx
        else:
            with self._transaction(write=False, at_serial=at_serial) as tx:
                yield tx

    @contextlib.contextmanager
    def write_transaction(self, *, allow_restart=False):
        """ Get a write transaction.

        If ``allow_restart`` is ``True`` then an existing read-only transaction is restarted as a write transaction.
        """
        tx = getattr(self._threadlocal, 'tx', None)
        if tx is not None:
            if not tx.write:
                if allow_restart:
                    self.restart_as_write_transaction()
                else:
                    raise self.ReadOnly(
                        "Expected an existing write transaction, "
                        "but there is an existing read transaction.")
            yield tx
        else:
            with self._transaction(write=True) as tx:
                yield tx


@frozen
class KeyChangeEvent:
    data: KeyData
    at_serial: int


class TransactionRootModel(RootModel):
    def __init__(self, xom):
        super().__init__(xom)
        self.model_cache = {}

    def create_user(self, username, password, **kwargs):
        if username in self.model_cache:
            assert self.model_cache[username] is None
        self.model_cache[username] = super().create_user(
            username, password, **kwargs)
        return self.model_cache[username]

    def create_stage(self, user, index, type="stage", **kwargs):
        key = (user.name, index)
        if key in self.model_cache:
            assert self.model_cache[key] is None
        self.model_cache[key] = super().create_stage(
            user, index, type=type, **kwargs)
        return self.model_cache[key]

    def delete_user(self, username):
        if username in self.model_cache:
            assert self.model_cache[username] is not None
            del self.model_cache[username]
        super().delete_user(username)

    def delete_stage(self, username, index):
        super().delete_stage(username, index)
        key = (username, index)
        if key in self.model_cache:
            assert self.model_cache[key] is not None
            del self.model_cache[key]

    def get_user(self, name):
        if name not in self.model_cache:
            self.model_cache[name] = super().get_user(name)
        return self.model_cache[name]

    def getstage(self, user, index=None):
        if index is None:
            user = user.strip('/')
            (user, index) = user.split('/')
        key = (user, index)
        if key not in self.model_cache:
            self.model_cache[key] = super().getstage(user, index)
        return self.model_cache[key]


class FileStoreTransaction:
    def __init__(self, keyfs):
        self.keyfs = keyfs
        self.closed = False
        self.write = True

    @cached_property
    def conn(self):
        return self.keyfs.get_connection(write=True, closing=False)

    @cached_property
    def io_file(self):
        return self.keyfs.io_file_factory(self.conn)

    def _close(self):
        if self.closed:
            # We can reach this when the transaction is restarted and there
            # is an exception after the commit and before the assignment of
            # the __dict__. The ``transaction`` context manager will call
            # ``rollback``, which then arrives here.
            return
        threadlog.debug("closing filestore transaction")
        self.conn.close()
        self.closed = True

    def commit(self):
        self.io_file.commit()
        self._close()

    def rollback(self):
        self.io_file.rollback()
        self.conn.rollback()
        threadlog.debug("filestore transaction rollback")
        self._close()


class KeysChecker:
    def __init__(self, keyfs, keys):
        all_key_names: set[str] = set()
        key_names: set[str] = {k.key_name for k in keys if isinstance(k, PatternedKey)}
        key_name_locations: set[tuple[str, str]] = set()
        key_name_relpaths: set[tuple[str, str]] = set()
        for key in keys:
            assert isinstance(key, (LocatedKey, PatternedKey, SearchKey))
            all_key_names.add(key.key_name)
            if isinstance(key, SearchKey) and key.key_name not in key_names:
                key_name_locations.add((key.key_name, key.location))
            if isinstance(key, LocatedKey) and key.key_name not in key_names:
                key_name_relpaths.add((key.key_name, key.relpath))
        self._dirty = keyfs._dirty
        self._original = keyfs._original
        self._ulid_keys = keyfs._ulid_keys
        self.all_key_names = all_key_names
        self.key_name_locations = key_name_locations
        self.key_name_relpaths = key_name_relpaths
        self.key_names = key_names
        self.processed_ulid_keys = set()
        self.ulidkeys_to_skip = set()

    def iter_ulidkey_values(self):
        all_key_names = self.all_key_names
        key_name_locations = self.key_name_locations
        key_name_relpaths = self.key_name_relpaths
        key_names = self.key_names
        processed_ulid_keys = self.processed_ulid_keys
        ulidkeys_to_skip = self.ulidkeys_to_skip
        for ulid_key in self._dirty:
            if (
                ulid_key in processed_ulid_keys
                or ulid_key.key_name not in all_key_names
            ):
                continue
            processed_ulid_keys.add(ulid_key)
            value = self._dirty[ulid_key]
            if isinstance(value, Deleted):
                continue
            if (
                ulid_key.key_name in key_names
                or (ulid_key.key_name, ulid_key.location) in key_name_locations
                or (ulid_key.key_name, ulid_key.relpath) in key_name_relpaths
            ):
                yield (ulid_key, ensure_deeply_readonly(value))
                ulidkeys_to_skip.add(ulid_key)
        for ulid_key in self._original:
            if (
                ulid_key in processed_ulid_keys
                or ulid_key.key_name not in all_key_names
            ):
                continue
            processed_ulid_keys.add(ulid_key)
            (back_serial, old_ulid_key, old_value) = self._original[ulid_key]
            if isinstance(old_ulid_key, Absent) or isinstance(old_value, Absent):
                continue
            if (
                old_ulid_key.key_name in key_names
                or (old_ulid_key.key_name, old_ulid_key.location) in key_name_locations
                or (old_ulid_key.key_name, old_ulid_key.relpath) in key_name_relpaths
            ):
                yield (old_ulid_key, old_value)
                ulidkeys_to_skip.add(old_ulid_key)
        # we don't check _ulid_keys here, because we need the values

    def iter_ulidkeys(self):
        for k, _v in self.iter_ulidkey_values():
            yield k
        all_key_names = self.all_key_names
        key_name_locations = self.key_name_locations
        key_name_relpaths = self.key_name_relpaths
        key_names = self.key_names
        processed_ulid_keys = self.processed_ulid_keys
        ulidkeys_to_skip = self.ulidkeys_to_skip
        for ulid_key in self._ulid_keys.values():
            if (
                isinstance(ulid_key, (Absent, Deleted))
                or ulid_key in processed_ulid_keys
                or ulid_key.key_name not in all_key_names
            ):
                continue
            processed_ulid_keys.add(ulid_key)
            # if we get here, then the ulid_key was iterated over before,
            # but the original value wasn't fetch
            # we can still be sure it exists, because we checked _dirty and
            # _original first and the key will only be here when iterated over
            if (
                ulid_key.key_name in key_names
                or (ulid_key.key_name, ulid_key.location) in key_name_locations
                or (ulid_key.key_name, ulid_key.relpath) in key_name_relpaths
            ):
                yield ulid_key
                ulidkeys_to_skip.add(ulid_key)


class Transaction:
    _dirty: dict[ULIDKey, KeyFSTypes | Deleted]
    _got_all: set[LocatedKey | PatternedKey | SearchKey]
    _model: TransactionRootModel | Absent
    _original: dict[ULIDKey, tuple[int, ULIDKey | Absent, KeyFSTypesRO | Absent]]
    _ulid_keys: dict[LocatedKey, ULIDKey | Absent | Deleted]

    def __init__(self, keyfs, *, at_serial=None, write=False):
        if write and at_serial:
            raise RuntimeError(
                "Can't open write transaction with 'at_serial'.")
        self.keyfs = keyfs
        self.commit_serial = None
        self.write = write
        if self.write:
            # open connection immediately
            self.conn  # noqa: B018
        if at_serial is None:
            at_serial = self.conn.last_changelog_serial
        self.at_serial = at_serial
        self._got_all = set()
        self._original = {}
        self._ulid_keys = {}
        self._dirty = {}
        self.closed = False
        self.doomed = False
        self._model = absent
        self._finished_listeners = []
        self._success_listeners = []

    @cached_property
    def conn(self):
        return self.keyfs.get_connection(
            write=self.write, closing=False)

    @cached_property
    def io_file(self):
        return self.keyfs.io_file_factory(self.conn)

    def get_model(self, xom):
        if self._model is absent:
            self._model = TransactionRootModel(xom)
        return self._model

    def iter_keys_at_serial(
        self, keys: LocatedKey, at_serial: int
    ) -> Iterator[KeyData]:
        return self.conn.iter_keys_at_serial(
            keys, at_serial, fill_cache=False, with_deleted=True
        )

    def iter_serial_and_value_backwards(
        self, key: LocatedKey | ULIDKey, last_serial: int
    ) -> Iterator[tuple[int, KeyFSTypesRO]]:
        if not isinstance(key, ULIDKey):
            key = self.resolve(key, fetch=True)
        while last_serial >= 0:
            data = self.conn.get_key_at_serial(key, last_serial)
            yield (data.last_serial, data.value)
            last_serial = data.back_serial

    def iter_ulidkey_values_for(
        self,
        keys: Sequence[LocatedKey | PatternedKey | SearchKey],
        *,
        fill_cache: bool = True,
        with_deleted: bool = False,
    ) -> Iterator[tuple[ULIDKey, KeyFSTypesRO]]:
        keys_checker = KeysChecker(self, keys)
        yield from keys_checker.iter_ulidkey_values()
        if all(k in self._got_all for k in keys):
            return
        processed_ulid_keys = keys_checker.processed_ulid_keys
        for keydata in self.conn.iter_keys_at_serial(
            keys,
            self.at_serial,
            skip_ulid_keys=keys_checker.ulidkeys_to_skip,
            fill_cache=fill_cache,
            with_deleted=with_deleted,
        ):
            key = keydata.key
            if key in processed_ulid_keys:
                continue
            if fill_cache and key not in self._original:
                self._original[key] = (keydata.serial, key, keydata.value)
                _key = self.keyfs._keys[key.key_name]
                if not isinstance(_key, LocatedKey):
                    _key = _key(**key.params)
                self._ulid_keys[_key] = key
            yield (key, keydata.value)
        if fill_cache:
            for _key in keys:
                self._got_all.add(_key)

    def iter_ulidkeys_for(
        self,
        keys: Sequence[LocatedKey | PatternedKey | SearchKey],
        *,
        fill_cache: bool = True,
        with_deleted: bool = False,
    ) -> Iterator[ULIDKey]:
        keys_checker = KeysChecker(self, keys)
        yield from keys_checker.iter_ulidkeys()
        if all(k in self._got_all for k in keys):
            return
        processed_ulid_keys = keys_checker.processed_ulid_keys
        for key in self.conn.iter_ulidkeys_at_serial(
            keys,
            self.at_serial,
            skip_ulid_keys=keys_checker.ulidkeys_to_skip,
            fill_cache=fill_cache,
            with_deleted=with_deleted,
        ):
            if key in processed_ulid_keys:
                continue
            _key = self.keyfs._keys[key.key_name]
            if not isinstance(_key, LocatedKey):
                _key = _key(**key.params)
            self._ulid_keys[_key] = key
            yield key
        for _key in keys:
            self._got_all.add(_key)

    def get_last_serial_and_value_at(
        self, key: LocatedKey | ULIDKey, at_serial: int
    ) -> tuple[int, ULIDKey | Absent, KeyFSTypesRO | Absent]:
        if not isinstance(key, ULIDKey):
            try:
                key = self.resolve_at(key, at_serial)
            except KeyError:
                return (-1, absent, absent)
        try:
            data = self.conn.get_key_at_serial(key, at_serial)
        except KeyError:
            return (-1, absent, absent)
        return (data.last_serial, data.key, data.value)

    def get_value_at(self, key: LocatedKey, at_serial: int) -> KeyFSTypesRO:
        (last_serial, val) = self.last_serial_and_value_at(key, at_serial)
        return val

    def back_serial(self, key: LocatedKey | ULIDKey) -> int:
        if not isinstance(key, ULIDKey):
            try:
                key = self.resolve(key, fetch=True)
            except KeyError:
                return -1
        (back_serial, ulid_key, val) = self.get_original(key)
        return back_serial

    def last_serial(self, key: LocatedKey | ULIDKey) -> int:
        if not isinstance(key, ULIDKey):
            try:
                key = self.resolve(key, fetch=True)
            except KeyError:
                return -1
        if key in self._dirty:
            return self.at_serial
        (last_serial, ulid_key, val) = self.get_original(key)
        return last_serial

    def last_serial_and_value_at(
        self, key: LocatedKey | ULIDKey, at_serial: int
    ) -> tuple[int, KeyFSTypesRO]:
        if not isinstance(key, ULIDKey):
            key = self.resolve_at(key, at_serial)
        data = self.conn.get_key_at_serial(key, at_serial)
        if data.value is deleted:
            raise KeyError(key)  # was deleted
        return (data.last_serial, data.value)

    def is_dirty(self, key: LocatedKey | ULIDKey) -> bool:
        if not isinstance(key, ULIDKey):
            try:
                key = self.resolve(key, fetch=False)
            except KeyError:
                return False
        return key in self._dirty

    def get_original(
        self, key: LocatedKey | ULIDKey
    ) -> tuple[int, ULIDKey | Absent, KeyFSTypesRO | Absent]:
        """ Return original value from start of transaction,
            without changes from current transaction."""
        if not isinstance(key, ULIDKey):
            key = self.resolve(key, fetch=True)
        if key not in self._original:
            (serial, ulid_key, val) = self.get_last_serial_and_value_at(
                key, self.at_serial
            )
            if val not in (absent, deleted):
                assert isinstance(ulid_key, ULIDKey)
                assert is_deeply_readonly(val)
            self._original[key] = (serial, ulid_key, val)
        (serial, ulid_key, val) = self._original[key]
        return (serial, ulid_key, val)

    def _get(
        self, key: LocatedKey | ULIDKey
    ) -> tuple[ULIDKey | Absent, KeyFSTypes | KeyFSTypesRO | Absent | Deleted]:
        val: KeyFSTypes | KeyFSTypesRO | Absent | Deleted
        if not isinstance(key, ULIDKey):
            try:
                key = self.resolve(key, fetch=True)
            except KeyError:
                val = absent
                if isinstance(key, LocatedKey) and isinstance(
                    self._ulid_keys.get(key), Deleted
                ):
                    val = deleted
                return (absent, val)
        ulid_key: ULIDKey | Absent
        if key in self._dirty:
            val = self._dirty[key]
            ulid_key = key
        else:
            (back_serial, ulid_key, val) = self.get_original(key)
        return (ulid_key, val)

    @overload
    def get(self, key: LocatedKey, default: Absent) -> KeyFSTypesRO: ...

    @overload
    def get(self, key: ULIDKey, default: Absent) -> KeyFSTypesRO: ...

    def get(self, key: LocatedKey | ULIDKey, default: Any = absent) -> Any:
        """Return current read-only value referenced by key."""
        (ulid_key, val) = self._get(key)
        if val in (absent, deleted):
            val = (
                # for convenience we return an empty instance if no default is given
                key.key_type() if default is absent else default
            )
        return ensure_deeply_readonly(val)

    def get_mutable(self, key: LocatedKey | ULIDKey) -> KeyFSTypes:
        """Return current mutable value referenced by key."""
        (ulid_key, val) = self._get(key)
        if val in (absent, deleted):
            # for convenience we return an empty instance
            val = key.key_type()
        return get_mutable_deepcopy(val)

    def exists(self, key: LocatedKey | ULIDKey) -> bool:
        if not isinstance(key, ULIDKey):
            try:
                key = self.resolve(key, fetch=True)
            except KeyError:
                return False
        val: KeyFSTypes | KeyFSTypesRO | Absent | Deleted
        if key in self._dirty:
            val = self._dirty[key]
        else:
            (serial, ulid_key, val) = self.get_original(key)
        return val not in (absent, deleted)

    def delete(self, key: LocatedKey | ULIDKey) -> None:
        if not self.write:
            raise self.keyfs.ReadOnly
        if isinstance(key, ULIDKey):
            ulid_key = key
        else:
            try:
                ulid_key = self.resolve(key, fetch=True)
            except KeyError:
                return
        (serial, old_ulid_key, val) = self.get_original(ulid_key)
        if isinstance(old_ulid_key, Absent):
            self._dirty.pop(ulid_key, Absent)
            if isinstance(key, LocatedKey):
                self._ulid_keys.pop(key, None)
        else:
            self._dirty[ulid_key] = deleted

    def key_for_ulid(self, ulid: ULID) -> ULIDKey:
        keydata = self.conn.get_ulid_at_serial(ulid, self.at_serial)
        if keydata.key not in self._original:
            self._original[keydata.key] = (
                keydata.last_serial,
                keydata.key,
                keydata.value,
            )
        return keydata.key

    def _new_absent_ulidkey(self, key):
        new_ulid_key = self._new_ulidkey(key)
        self._ulid_keys[key] = new_ulid_key
        self._original[new_ulid_key] = (-1, absent, absent)
        return new_ulid_key

    def _new_ulidkey(self, key):
        while True:
            new_ulid_key = key.new_ulid()
            # when rapidly generating new ulid keys, we can get
            # duplicates due to the birthday paradox
            if new_ulid_key not in self._original:
                return new_ulid_key

    def resolve(self, key: LocatedKey, *, fetch: bool) -> ULIDKey:
        result = self.resolve_keys((key,), fetch=fetch, new_for_missing=False)
        possible_ulid_key = next(result, None)
        # exhaust the iterator to let it update its caches
        if possible_ulid_key is not None and next(result, None) is not None:
            raise RuntimeError("Got additional keys")
        (_key, ulid_key) = (
            (key, absent) if possible_ulid_key is None else possible_ulid_key
        )
        if isinstance(ulid_key, (Absent, Deleted)):
            raise KeyError(key)
        assert _key == key
        return ulid_key

    def resolve_at(self, key: LocatedKey, at_serial: int) -> ULIDKey:
        result = self.conn.iter_ulidkeys_at_serial(
            (key,), at_serial=at_serial, fill_cache=False, with_deleted=True
        )
        fetched_ulid_key: ULIDKey | Absent = next(result, absent)
        if isinstance(fetched_ulid_key, Absent):
            raise KeyError(key)
        # exhaust the iterator to let it update its caches
        if next(result, absent) is not absent:
            raise RuntimeError("Got additional keys")
        return fetched_ulid_key

    def resolve_keys(  # noqa: PLR0912
        self,
        keys: Iterable[LocatedKey],
        *,
        fetch: bool,
        fill_cache: bool = False,
        new_for_missing: bool,
    ) -> Iterator[tuple[LocatedKey, ULIDKey | Absent | Deleted]]:
        missing = set()
        for key in keys:
            assert isinstance(key, LocatedKey)
            _ulid_key = self._ulid_keys.get(key, None)
            if _ulid_key is None:
                missing.add(key)
                continue
            yield (
                (key, self._new_absent_ulidkey(key))
                if _ulid_key is absent and new_for_missing
                else (key, _ulid_key)
            )
        if not missing:
            return
        processed = set()
        if fetch:
            for keydata in self.conn.iter_keys_at_serial(
                missing,
                at_serial=self.at_serial,
                fill_cache=fill_cache,
                with_deleted=True,
            ):
                ulid_key = keydata.key
                _key = self.keyfs._keys[ulid_key.key_name]
                if not isinstance(_key, LocatedKey):
                    _key = _key(**ulid_key.params)
                if (parent_key := ulid_key.parent_key) is not None and not self.exists(
                    parent_key
                ):
                    # parent was deleted
                    ulid_key = deleted
                else:
                    self._original[ulid_key] = (
                        keydata.last_serial,
                        ulid_key,
                        keydata.value,
                    )
                self._ulid_keys[_key] = ulid_key
                processed.add(_key)
                yield (_key, ulid_key)
        else:
            for ulid_key in self.conn.iter_ulidkeys_at_serial(
                missing,
                at_serial=self.at_serial,
                fill_cache=fill_cache,
                with_deleted=True,
            ):
                _key = self.keyfs._keys[ulid_key.key_name]
                if not isinstance(_key, LocatedKey):
                    _key = _key(**ulid_key.params)
                if (
                    parent_key := ulid_key.parent_key
                ) is not None and not parent_key.exists():
                    # parent was deleted
                    self._ulid_keys[_key] = deleted
                    _u_key = absent
                else:
                    _u_key = ulid_key
                processed.add(_key)
                yield (_key, _u_key)
        if new_for_missing:
            for key in missing.difference(processed):
                new_ulid_key = self._new_absent_ulidkey(key)
                self._ulid_keys[key] = new_ulid_key
                yield (key, new_ulid_key)
        else:
            for _key in missing.difference(processed):
                self._ulid_keys[key] = absent
                yield (_key, absent)

    def set(self, key: LocatedKey | ULIDKey, val: KeyFSTypes) -> None:
        if not isinstance(val, key.key_type) and not issubclass(
            key.key_type, type(val)
        ):
            raise TypeError(
                "%r requires value of type %r, got %r"
                % (key.relpath, key.key_type.__name__, type(val).__name__)
            )
        if not self.write:
            raise self.keyfs.ReadOnly
        # sanity check for dictionaries: we always want to have unicode
        # keys, not bytes
        if key.key_type is dict:
            check_unicode_keys(val)
        assert val not in (None, absent, deleted)
        (old_ulid_key, old_val) = self._get(key)
        if not isinstance(key, ULIDKey):
            if old_val in (absent, deleted) or isinstance(old_ulid_key, Absent):
                cache_key = self._new_ulidkey(key)
            else:
                cache_key = old_ulid_key
            self._ulid_keys[key] = cache_key
        elif old_ulid_key is not absent and old_val in (absent, deleted):
            cache_key = self._new_ulidkey(key)
            for lkey, ulid_key in self._ulid_keys.items():
                if ulid_key == key:
                    self._ulid_keys[lkey] = cache_key
                    break
        else:
            cache_key = key
        self._dirty[cache_key] = val

    def commit(self):
        threadlog.debug(
            "_original %s, _dirty %s, _ulid_keys %s",
            len(self._original),
            len(self._dirty),
            len(self._ulid_keys),
        )
        if self.doomed:
            threadlog.debug("closing doomed transaction")
            result = self._close()
            self._run_listeners(self._finished_listeners)
            return result
        if not self.write:
            result = self._close()
            self._run_listeners(self._finished_listeners)
            return result
        # no longer needed
        self._got_all.clear()
        self._ulid_keys.clear()
        records = []
        while self._dirty:
            (key, val) = self._dirty.popitem()
            assert val is not absent
            (back_serial, old_ulid_key, old_val) = self.get_original(key)
            if val == old_val and key == old_ulid_key:
                continue
            if val is deleted and old_val in (absent, deleted):
                continue
            records.append(Record(key, val, back_serial, old_ulid_key, old_val))
        if not records and not self.io_file.is_dirty():
            threadlog.debug("nothing to commit, just closing tx")
            result = self._close()
            self._run_listeners(self._finished_listeners)
            return result
        # no longer needed
        self._original.clear()
        with contextlib.ExitStack() as cstack:
            cstack.callback(self._close)
            with self.io_file, self.conn.write_transaction(self.io_file) as writer:
                writer.set_rel_renames(self.io_file.get_rel_renames())
                writer.records_set(records)
                commit_serial = writer.commit_serial
            self.commit_serial = commit_serial
            self._run_listeners(self._success_listeners)
            self._run_listeners(self._finished_listeners)
        return commit_serial

    def on_commit_success(self, callback):
        self._success_listeners.append(callback)

    def on_finished(self, callback):
        self._finished_listeners.append(callback)

    def _run_listeners(self, listeners):
        for listener in listeners:
            try:
                listener()
            except Exception:
                threadlog.exception("Error calling %s", listener)

    def _close(self):
        if self.closed:
            # We can reach this when the transaction is restarted and there
            # is an exception after the commit and before the assignment of
            # the __dict__. The ``transaction`` context manager will call
            # ``rollback``, which then arrives here.
            return
        try:
            threadlog.debug("closing transaction at %s", self.at_serial)
            del self._got_all
            del self._model
            del self._original
            del self._ulid_keys
            del self._dirty
        finally:
            self.conn.close()
            self.closed = True
        return self.at_serial

    def rollback(self):
        try:
            self.conn.rollback()
            if self.keyfs.io_file_factory is not None:
                self.io_file.rollback()
            threadlog.debug("transaction rollback at %s" % (self.at_serial))
        finally:
            result = self._close()
        self._run_listeners(self._finished_listeners)
        return result

    def restart(self, write=False):
        if self.write:
            raise RuntimeError("Can't restart a write transaction.")
        self.commit()
        threadlog.debug(
            "restarting %s transaction afresh as %s transaction",
            "write" if self.write else "read",
            "write" if write else "read")
        try:
            newtx = self.__class__(self.keyfs, write=write)
        except BaseException:
            self.doomed = True
            raise
        self._close()
        self.__dict__ = newtx.__dict__

    def doom(self):
        """ mark as doomed to automatically rollback any changes """
        self.doomed = True


def check_unicode_keys(d):
    for key, val in d.items():
        assert not isinstance(key, bytes), repr(key)
        # not allowing bytes seems ok for now, we might need to relax that
        # it certainly helps to get unicode clean
        assert not isinstance(val, bytes), repr(key) + "=" + repr(val)
        if isinstance(val, dict):
            check_unicode_keys(val)
