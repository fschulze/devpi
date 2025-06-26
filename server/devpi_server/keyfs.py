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
from .filestore import FileEntry
from .fileutil import read_int_from_file
from .fileutil import write_int_to_file
from .interfaces import IStorageConnection
from .interfaces import IWriter
from .keyfs_types import LocatedKey
from .keyfs_types import NamedKey
from .keyfs_types import NamedKeyFactory
from .keyfs_types import Record
from .log import thread_change_log_prefix
from .log import thread_pop_log
from .log import thread_push_log
from .log import threadlog
from .markers import absent
from .markers import deleted
from .model import RootModel
from .readonly import ensure_deeply_readonly
from .readonly import get_mutable_deepcopy
from .readonly import is_deeply_readonly
from devpi_common.types import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
import contextlib
import errno
import time


if TYPE_CHECKING:
    from .keyfs_types import KeyFSTypesRO
    from .markers import Absent
    from .mythread import MyThread


class KeyfsTimeoutError(TimeoutError):
    pass


class MissingFileException(Exception):
    def __init__(self, relpath, serial):
        msg = "missing file '%s' at serial %s" % (relpath, serial)
        super(MissingFileException, self).__init__(msg)
        self.relpath = relpath
        self.serial = serial


class TxNotificationThread:
    _get_ixconfig_cache: dict[tuple[str, str], dict | None]
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
            self._execute_hooks(event_serial, self.log)
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
                    "Waiting for file %s in event serial %s",
                    e.relpath, e.serial)
                self.thread.sleep(5)
            except Exception:
                self.log.exception(
                    "Unhandled exception in notification thread.")
                self.thread.sleep(1.0)

    def get_ixconfig(self, entry, event_serial):
        user = entry.user
        index = entry.index
        if getattr(self, '_get_ixconfig_cache_serial', None) != event_serial:
            self._get_ixconfig_cache = {}
            self._get_ixconfig_cache_serial = event_serial
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

    def _execute_hooks(self, event_serial, log, raising=False):
        log.debug("calling hooks for tx%s", event_serial)
        with self.keyfs.get_connection() as conn:
            changes = conn.get_changes(event_serial)
        # we first check for missing files before we call subscribers
        for relpath, (keyname, _back_serial, val) in changes.items():
            if keyname in ("STAGEFILE", "PYPIFILE_NOMD5"):
                key = self.keyfs.get_key_instance(keyname, relpath)
                entry = FileEntry(key, val)
                if entry.meta == {} or entry.last_modified is None:
                    # the file was removed
                    continue
                ixconfig = self.get_ixconfig(entry, event_serial)
                if ixconfig is None:
                    # the index doesn't exist (anymore)
                    continue
                if ixconfig.get("type") == "mirror" and ixconfig.get(
                    "mirror_use_external_urls", False
                ):
                    # the index uses external URLs now
                    continue
                with self.keyfs.filestore_transaction():
                    if entry.file_exists():
                        # all good
                        continue
                # the file is missing, check whether we can ignore it
                serial = self.keyfs.get_current_serial()
                if event_serial < serial:
                    # there are newer serials existing
                    with self.keyfs.read_transaction() as tx:
                        current_val = tx.get(key)
                    if current_val is None:
                        # entry was deleted
                        continue
                    current_entry = FileEntry(key, current_val)
                    if current_entry.meta == {} or current_entry.last_modified is None:
                        # the file was removed at some point
                        continue
                    current_ixconfig = self.get_ixconfig(entry, serial)
                    if current_ixconfig is None:
                        # the index doesn't exist (anymore)
                        continue
                    if current_ixconfig.get("type") == "mirror":
                        if current_ixconfig.get("mirror_use_external_urls", False):
                            # the index uses external URLs now
                            continue
                        if current_entry.project is None:
                            # this is an old mirror entry with no
                            # project info, so this can be ignored
                            continue
                    log.debug("missing current_entry.meta %r", current_entry.meta)
                log.debug("missing entry.meta %r", entry.meta)
                raise MissingFileException(relpath, event_serial)
        # all files exist or are deleted in a later serial,
        # call subscribers now
        for relpath, (keyname, back_serial, val) in changes.items():
            subscribers = self._on_key_change.get(keyname, [])
            if not subscribers:
                continue
            key = self.keyfs.get_key_instance(keyname, relpath)
            ev = KeyChangeEvent(key, val, event_serial, back_serial)
            for sub in subscribers:
                subname = getattr(sub, "__name__", sub)
                log.debug(
                    "%s(key=%r, at_serial=%r, back_serial=%r",
                    subname,
                    ev.typedkey,
                    event_serial,
                    ev.back_serial,
                )
                try:
                    sub(ev)
                except Exception:
                    if raising:
                        raise
                    log.exception("calling %s failed, serial=%s", sub, event_serial)

        log.debug("finished calling all hooks for tx%s", event_serial)


class KeyFS:
    """ singleton storage object. """
    class ReadOnly(Exception):
        """ attempt to open write transaction while in readonly mode. """

    def __init__(
        self,
        basedir,
        storage_info,
        *,
        io_file_factory=None,
        readonly=False,
        cache_size=10000,
    ):
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
            cache_size=cache_size,
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
        with self.get_connection() as conn:
            io_file = self.io_file_factory(conn)
            io_file.perform_crash_recovery()

    def import_changes(self, serial, changes):
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
            records = []
            subscriber_changes = {}
            for relpath, (keyname, back_serial, val) in changes.items():
                try:
                    old_val = conn.get_relpath_at(relpath, serial - 1).value
                except KeyError:
                    old_val = absent
                typedkey = self.get_key_instance(keyname, relpath)
                subscriber_changes[typedkey] = (val, back_serial)
                records.append(
                    Record(
                        typedkey,
                        deleted if val is None else get_mutable_deepcopy(val),
                        back_serial,
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

    def register_located_key(self, key_name, location, name, key_type):
        return self.register_key(LocatedKey(self, key_name, location, name, key_type))

    def register_named_key(self, key_name, pattern_or_name, parent_key, key_type):
        key: LocatedKey | NamedKey | NamedKeyFactory
        if "{" in pattern_or_name:
            key = NamedKeyFactory(self, key_name, pattern_or_name, parent_key, key_type)
        elif parent_key is None:
            key = LocatedKey(self, key_name, "", pattern_or_name, key_type)
        else:
            key = NamedKey(self, key_name, pattern_or_name, parent_key, key_type)
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
            raise self.ReadOnly()
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
            raise self.ReadOnly()
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


class KeyChangeEvent:
    def __init__(self, typedkey, value, at_serial, back_serial):
        self.typedkey = typedkey
        self.value = value
        self.at_serial = at_serial
        self.back_serial = back_serial


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


class Transaction:
    _model: TransactionRootModel | Absent

    def __init__(self, keyfs, at_serial=None, write=False):
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
        self._original = {}
        self.cache = {}
        self.dirty = set()
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

    def iter_relpaths_at(self, typedkeys, at_serial):
        return self.conn.iter_relpaths_at(typedkeys, at_serial)

    def iter_serial_and_value_backwards(self, relpath, last_serial):
        while last_serial >= 0:
            data = self.conn.get_relpath_at(relpath, last_serial)
            yield (data.last_serial, data.value)
            last_serial = data.back_serial

    def get_last_serial_and_value_at(
        self,
        typedkey: LocatedKey,
        at_serial: int,
    ) -> tuple[int, KeyFSTypesRO | Absent]:
        relpath = typedkey.relpath
        try:
            data = self.conn.get_relpath_at(relpath, at_serial)
        except KeyError:
            return (-1, absent)
        return (data.last_serial, data.value)

    def get_value_at(self, typedkey: LocatedKey, at_serial: int) -> KeyFSTypesRO | None:
        (last_serial, val) = self.last_serial_and_value_at(typedkey, at_serial)
        return val

    def last_serial(self, typedkey: LocatedKey) -> int:
        if typedkey in self.cache:
            return self.at_serial
        (last_serial, val) = self.get_original(typedkey)
        return last_serial

    def last_serial_and_value_at(self, typedkey, at_serial):
        relpath = typedkey.relpath
        data = self.conn.get_relpath_at(relpath, at_serial)
        if data.value is deleted:
            raise KeyError(relpath)  # was deleted
        return (data.last_serial, data.value)

    def is_dirty(self, typedkey):
        return typedkey in self.dirty

    def get_original(self, typedkey):
        """ Return original value from start of transaction,
            without changes from current transaction."""
        if typedkey not in self._original:
            (serial, val) = self.get_last_serial_and_value_at(typedkey, self.at_serial)
            if val not in (absent, deleted):
                assert is_deeply_readonly(val)
            self._original[typedkey] = (serial, val)
        return self._original[typedkey]

    def _get(self, typedkey):
        assert isinstance(typedkey, LocatedKey)
        if typedkey in self.cache:
            val = self.cache[typedkey]
        else:
            (back_serial, val) = self.get_original(typedkey)
        if val in (absent, deleted):
            # for convenience we return an empty instance
            val = typedkey.type()
        return val

    def get(self, typedkey):
        """Return current read-only value referenced by typedkey."""
        return ensure_deeply_readonly(self._get(typedkey))

    def get_mutable(self, typedkey):
        """Return current mutable value referenced by typedkey."""
        return get_mutable_deepcopy(self._get(typedkey))

    def exists(self, typedkey):
        if typedkey in self.cache:
            val = self.cache[typedkey]
            return val not in (absent, deleted)
        (serial, val) = self.get_original(typedkey)
        return val not in (absent, deleted)

    def delete(self, typedkey):
        if not self.write:
            raise self.keyfs.ReadOnly()
        self.cache[typedkey] = deleted
        self.dirty.add(typedkey)

    def set(self, typedkey, val):  # noqa: A003
        if not self.write:
            raise self.keyfs.ReadOnly()
        # sanity check for dictionaries: we always want to have unicode
        # keys, not bytes
        if typedkey.type is dict:
            check_unicode_keys(val)
        assert val is not None
        self.cache[typedkey] = val
        self.dirty.add(typedkey)

    def commit(self):
        threadlog.debug(
            "_original %s, cache %s, dirty %s",
            len(self._original),
            len(self.cache),
            len(self.dirty),
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
        records = []
        for typedkey in self.dirty:
            val = self.cache[typedkey]
            assert val is not absent
            (back_serial, old_val) = self.get_original(typedkey)
            if val == old_val:
                continue
            if val is deleted and old_val in (absent, deleted):
                continue
            records.append(Record(typedkey, val, back_serial, old_val))
        if not records and not self.io_file.is_dirty():
            threadlog.debug("nothing to commit, just closing tx")
            result = self._close()
            self._run_listeners(self._finished_listeners)
            return result
        with contextlib.ExitStack() as cstack:
            cstack.callback(self._close)
            cstack.enter_context(self.io_file)
            fswriter = IWriter(
                cstack.enter_context(self.conn.write_transaction(self.io_file))
            )
            fswriter.set_rel_renames(self.io_file.get_rel_renames())
            fswriter.records_set(records)
            commit_serial = fswriter.commit_serial
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
            del self._model
            del self._original
            del self.cache
            del self.dirty
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
