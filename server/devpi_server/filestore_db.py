from .interfaces import IIOFile
from zope.interface import implementer


@implementer(IIOFile)
class DBIOFile:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, cls, val, tb):
        if cls is not None:
            self.rollback()
            return False
        return True

    def commit(self):
        return self.conn.commit_files_without_increasing_serial()

    def delete(self, path, *, is_last_of_hash):
        return self.conn.io_file_delete(path, is_last_of_hash=is_last_of_hash)

    def exists(self, path):
        return self.conn.io_file_exists(path)

    def get_content(self, path):
        return self.conn.io_file_get(path)

    def get_rel_renames(self):
        return []

    def is_dirty(self):
        return bool(self.conn.dirty_files)

    def is_path_dirty(self, path):
        return path in self.conn.dirty_files

    def new_open(self, path):
        return self.conn.io_file_new_open(path)

    def open_read(self, path):
        return self.conn.io_file_open(path)

    def os_path(self, path):
        return self.conn.io_file_os_path(path)

    def perform_crash_recovery(self):
        return self.conn.storage.perform_crash_recovery()

    def rollback(self):
        return self.conn.rollback()

    def set_content(self, path, content_or_file):
        return self.conn.io_file_set(path, content_or_file)

    def size(self, path):
        return self.conn.io_file_size(path)
