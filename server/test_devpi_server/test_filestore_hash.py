from devpi_server.filestore_hash import check_pending_renames
from devpi_server.filestore_hash import commit_renames
from devpi_server.filestore_hash import make_rel_renames
from devpi_server.keyfs_types import FilePathInfo
from pathlib import Path
import os
import pytest


class TestRenameFileLogic:
    def test_new_content_nocrash(self, tmpdir):
        file1 = tmpdir.join("file1")
        file1_tmp = file1 + "-tmp"
        file1.write("hello")
        file1_tmp.write("this")
        pending_renames = [(Path(str(file1_tmp)), Path(str(file1)))]
        rel_renames = make_rel_renames(str(tmpdir), pending_renames)
        commit_renames(str(tmpdir), rel_renames)
        assert file1.check()
        assert file1.read() == "this"
        assert not file1_tmp.exists()
        check_pending_renames(str(tmpdir), rel_renames)
        assert file1.check()
        assert file1.read() == "this"
        assert not file1_tmp.exists()

    def test_new_content_crash(self, tmpdir, caplog):
        file1 = tmpdir.join("file1")
        file1_tmp = file1 + "-tmp"
        file1.write("hello")
        file1_tmp.write("this")
        pending_renames = [(Path(str(file1_tmp)), Path(str(file1)))]
        rel_renames = make_rel_renames(str(tmpdir), pending_renames)
        # we don't call perform_pending_renames, simulating a crash
        assert file1.read() == "hello"
        assert file1_tmp.exists()
        check_pending_renames(str(tmpdir), rel_renames)
        assert file1.check()
        assert file1.read() == "this"
        assert not file1_tmp.exists()
        assert len(caplog.getrecords(".*completed.*file-commit.*")) == 1

    def test_remove_nocrash(self, tmpdir):
        file1 = tmpdir.join("file1")
        file1.write("hello")
        pending_renames = [(None, Path(str(file1)))]
        rel_renames = make_rel_renames(str(tmpdir), pending_renames)
        commit_renames(str(tmpdir), rel_renames)
        assert not file1.exists()
        check_pending_renames(str(tmpdir), rel_renames)
        assert not file1.exists()

    def test_remove_crash(self, tmpdir, caplog):
        file1 = tmpdir.join("file1")
        file1.write("hello")
        pending_renames = [(None, Path(str(file1)))]
        rel_renames = make_rel_renames(str(tmpdir), pending_renames)
        # we don't call perform_pending_renames, simulating a crash
        assert file1.exists()
        check_pending_renames(str(tmpdir), rel_renames)
        assert not file1.exists()
        assert len(caplog.getrecords(".*completed.*file-del.*")) == 1

    @pytest.mark.storage_with_filesystem
    @pytest.mark.notransaction
    @pytest.mark.xfail
    def test_dirty_files_removed_on_rollback(self, file_digest, keyfs):
        content = b'foo'
        content_hash = file_digest(content)
        with pytest.raises(RuntimeError), keyfs.read_transaction() as tx:  # noqa: PT012
            tx.io_file.set_content(FilePathInfo('foo', content_hash), content)
            tmppath = tx.io_file._dirty_files[Path(keyfs.basedir.join('+files', content_hash[:3], content_hash[3:]).strpath)].tmppath
            assert os.path.exists(tmppath)
            # abort transaction
            raise RuntimeError
        assert not os.path.exists(tmppath)
