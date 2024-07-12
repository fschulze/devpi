from .fileutil import get_write_file_ensure_dir
from .fileutil import rename
from .interfaces import IIOFile
from .keyfs_types import FilePathInfo
from .log import threadlog
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from zope.interface import Interface
from zope.interface import alsoProvides
from zope.interface import implementer
import os
import re
import shutil
import sys
import threading


class ITempStorageFile(Interface):
    """ Marker interface. """


class DirtyFile:
    def __init__(self, path):
        self.path = path
        # use hash of path, pid and thread id to prevent conflicts
        key = f"{path}{os.getpid()}{threading.current_thread().ident}"
        digest = sha256(key.encode('utf-8')).hexdigest()
        if sys.platform == 'win32':
            # on windows we have to shorten the digest, otherwise we reach
            # the 260 chars file path limit too quickly
            digest = digest[:8]
        self.tmppath = f"{path}-{digest}-tmp"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.path}>"

    @classmethod
    def from_content(cls, path, content_or_file):
        self = DirtyFile(path)
        if hasattr(content_or_file, "devpi_srcpath"):
            dirname = os.path.dirname(self.tmppath)
            if not os.path.exists(dirname):
                # ignore file exists errors
                # one reason for that error is a race condition where
                # another thread tries to create the same folder
                with suppress(FileExistsError):
                    os.makedirs(dirname)
            os.link(content_or_file.devpi_srcpath, self.tmppath)
        else:
            with get_write_file_ensure_dir(self.tmppath) as f:
                if isinstance(content_or_file, bytes):
                    f.write(content_or_file)
                else:
                    assert content_or_file.seekable()
                    content_or_file.seek(0)
                    shutil.copyfileobj(content_or_file, f)
        return self


@implementer(IIOFile)
class FSIOFile:
    def __init__(self, conn, settings):
        self.conn = conn
        self.settings = settings
        self.basedir = Path(self.conn.storage.basedir)
        self._dirty_files = {}

    def __enter__(self):
        return self

    def __exit__(self, cls, val, tb):
        if cls is not None:
            self.rollback()
            return False
        rel_renames = self.get_rel_renames()
        (files_commit, files_del) = self.write_dirty_files(rel_renames)
        if files_commit or files_del:
            threadlog.debug(
                "wrote files: %s",
                LazyChangesFormatter({}, files_commit, files_del))
        return True

    def delete(self, path, *, is_last_of_hash):  # noqa: ARG002
        assert isinstance(path, FilePathInfo)
        path = str(self.basedir / path.relpath)
        old = self._dirty_files.get(path)
        if old is not None:
            os.remove(old.tmppath)
        self._dirty_files[path] = None

    def exists(self, path):
        assert isinstance(path, FilePathInfo)
        path = str(self.basedir / path.relpath)
        if path in self._dirty_files:
            dirty_file = self._dirty_files[path]
            if dirty_file is None:
                return False
            path = dirty_file.tmppath
        return os.path.exists(path)

    def get_content(self, path):
        assert isinstance(path, FilePathInfo)
        path = str(self.basedir / path.relpath)
        if path in self._dirty_files:
            dirty_file = self._dirty_files[path]
            if dirty_file is None:
                raise OSError
            path = dirty_file.tmppath
        with open(path, "rb") as f:
            data = f.read()
            if len(data) > 1048576:
                threadlog.warn(
                    "Read %.1f megabytes into memory in get_content for %s",
                    len(data) / 1048576, path)
            return data

    def is_dirty(self):
        return bool(self._dirty_files)

    def is_path_dirty(self, path):
        return path in self._dirty_files

    def new_open(self, path):
        assert isinstance(path, FilePathInfo)
        assert not self.exists(path)
        path = str(self.basedir / path.relpath)
        assert not path.endswith("-tmp")
        f = get_write_file_ensure_dir(DirtyFile(path).tmppath)
        alsoProvides(f, ITempStorageFile)
        return f

    def open_read(self, path):
        assert isinstance(path, FilePathInfo)
        path = str(self.basedir / path.relpath)
        if path in self._dirty_files:
            dirty_file = self._dirty_files[path]
            if dirty_file is None:
                raise OSError
            path = dirty_file.tmppath
        return open(path, "rb")  # noqa: SIM115

    def os_path(self, path):
        assert isinstance(path, FilePathInfo)
        return str(self.basedir / path.relpath)

    def set_content(self, path, content_or_file):
        assert isinstance(path, FilePathInfo)
        path = str(self.basedir / path.relpath)
        assert not path.endswith("-tmp")
        if ITempStorageFile.providedBy(content_or_file):
            self._dirty_files[path] = DirtyFile(path)
        else:
            self._dirty_files[path] = DirtyFile.from_content(path, content_or_file)

    def size(self, path):
        assert isinstance(path, FilePathInfo)
        path = str(self.basedir / path.relpath)
        if path in self._dirty_files:
            dirty_file = self._dirty_files[path]
            if dirty_file is None:
                return None
            path = dirty_file.tmppath
        with suppress(OSError):
            return os.path.getsize(path)

    def commit(self):
        rel_renames = self.get_rel_renames()
        (files_commit, files_del) = self.write_dirty_files(rel_renames)
        if files_commit or files_del:
            threadlog.debug(
                "wrote files without increasing serial: %s",
                LazyChangesFormatter({}, files_commit, files_del))

    def get_rel_renames(self):
        pending_renames = []
        for path, dirty_file in self._dirty_files.items():
            if dirty_file is None:
                pending_renames.append((None, path))
            else:
                pending_renames.append((dirty_file.tmppath, path))
        basedir = str(self.basedir)
        return list(make_rel_renames(basedir, pending_renames))

    def perform_crash_recovery(self):
        rel_renames = self.conn.get_rel_renames(self.conn.last_changelog_serial)
        if rel_renames:
            check_pending_renames(str(self.basedir), rel_renames)

    def rollback(self):
        for dirty_file in self._dirty_files.values():
            if dirty_file is not None:
                os.remove(dirty_file.tmppath)
        self._dirty_files.clear()

    def write_dirty_files(self, rel_renames):
        basedir = str(self.basedir)
        # If we crash in the remainder, the next restart will
        # - call check_pending_renames which will replay any remaining
        #   renames from the changelog entry, and
        # - initialize next_serial from the max committed serial + 1
        result = commit_renames(basedir, rel_renames)
        self._dirty_files.clear()
        return result


class LazyChangesFormatter:
    __slots__ = ('files_commit', 'files_del', 'keys')

    def __init__(self, changes, files_commit, files_del):
        self.files_commit = files_commit
        self.files_del = files_del
        self.keys = changes.keys()

    def __str__(self):
        msg = []
        if self.keys:
            msg.append(f"keys: {','.join(repr(c) for c in self.keys)}")
        if self.files_commit:
            msg.append(f"files_commit: {','.join(self.files_commit)}")
        if self.files_del:
            msg.append(f"files_del: {','.join(self.files_del)}")
        return ", ".join(msg)


def check_pending_renames(basedir, pending_relnames):
    for relpath in pending_relnames:
        path = os.path.join(basedir, relpath)
        suffix = tmpsuffix_for_path(relpath)
        if suffix is not None:
            suffix_len = len(suffix)
            dst = path[:-suffix_len]
            if os.path.exists(path):
                rename(path, dst)
                threadlog.warn("completed file-commit from crashed tx: %s",
                               dst)
            elif not os.path.exists(dst):
                raise OSError("missing file %s" % dst)
        else:
            with suppress(OSError):
                os.remove(path)  # was already removed
                threadlog.warn("completed file-del from crashed tx: %s", path)


def commit_renames(basedir, pending_renames):
    files_del = []
    files_commit = []
    for relpath in pending_renames:
        path = os.path.join(basedir, relpath)
        suffix = tmpsuffix_for_path(relpath)
        if suffix is not None:
            suffix_len = len(suffix)
            rename(path, path[:-suffix_len])
            files_commit.append(relpath[:-suffix_len])
        else:
            with suppress(OSError):
                os.remove(path)
            files_del.append(relpath)
    return (files_commit, files_del)


def make_rel_renames(basedir, pending_renames):
    # produce a list of strings which are
    # - paths relative to basedir
    # - if they have "-tmp" at the end it means they should be renamed
    #   to the path without the "-tmp" suffix
    # - if they don't have "-tmp" they should be removed
    for source, dest in pending_renames:
        if source is not None:
            assert source.startswith(dest)
            assert source.endswith("-tmp")
            yield source[len(basedir) + 1:]
        else:
            assert dest.startswith(basedir)
            yield dest[len(basedir) + 1:]


tmp_file_matcher = re.compile(r"(.*?)(-[0-9a-fA-F]{8,64})?(-tmp)$")


def tmpsuffix_for_path(path):
    # ends with -tmp and includes hash since temp files are written directly
    # to disk instead of being kept in memory
    m = tmp_file_matcher.match(path)
    if m is not None:
        return m.group(2) + m.group(3) if m.group(2) else m.group(3)
    return None
