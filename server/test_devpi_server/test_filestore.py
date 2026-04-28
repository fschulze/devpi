from devpi_server.filestore import get_hashes
from devpi_server.filestore import get_size
from devpi_server.model.remote import iter_cache_remote_file
from devpi_server.model.simpleapi import SimpleInfo
from io import BytesIO
from pathlib import Path
from webob.headers import ResponseHeaders
import pytest


zip_types = ("application/zip", "application/x-zip-compressed")


@pytest.fixture
def filestore(xom):
    return xom.filestore


@pytest.mark.notransaction
def test_cache_remote_file(filestore, http, gen, xom):
    with filestore.keyfs.write_transaction():
        link = gen.pypi_package_link("pytest-1.8.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.best_available_hash_spec is None
        assert not entry.hashes
        assert not entry.file_exists()
        headers = ResponseHeaders({
            "content-length": "3",
            "last-modified": "Thu, 25 Nov 2010 20:00:27 GMT"})
        http.url2response[link.url] = dict(
            status_code=200, headers=headers, raw=BytesIO(b"123")
        )
        stage = xom.model.getstage('root/pypi')
        list(iter_cache_remote_file(stage, entry, entry.url))
        rheaders = entry.gethttpheaders()
        assert rheaders["content-length"] == "3"
        assert rheaders["content-type"] in zip_types
        assert rheaders["last-modified"] == headers["last-modified"]
        content = entry.file_get_content()
        assert content == b"123"

    # reget entry and check about content
    with filestore.keyfs.read_transaction():
        entry = filestore.get_file_entry(entry.relpath)
        assert entry.file_exists()
        assert entry.hashes.get_default_spec() == get_hashes(content).get_default_spec()
        assert entry.file_size() == 3
        rheaders = entry.gethttpheaders()
        assert entry.file_get_content() == b"123"


@pytest.mark.notransaction
@pytest.mark.storage_with_filesystem
def test_file_tx_commit(filestore, gen, xom):
    filestore.keyfs.begin_transaction_in_thread(write=True)
    link = gen.pypi_package_link("pytest-1.8.zip", hash_spec=False)
    info = SimpleInfo.from_url(link)
    entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
    assert not entry.file_exists()
    content = b"123"
    entry.file_set_content(content, hashes=get_hashes(content), size=get_size(content))
    assert entry.file_exists()
    filepath = Path(entry.file_os_path(_raises=False))
    assert not filepath.exists()
    assert entry.file_get_content() == content
    # commit existing data and start new transaction
    filestore.keyfs.commit_transaction_in_thread()
    filestore.keyfs.begin_transaction_in_thread(write=True)
    assert filepath.exists()
    entry.file_delete(is_last_of_hash=True)
    assert filepath.exists()
    assert not entry.file_exists()
    filestore.keyfs.commit_transaction_in_thread()
    assert not filepath.exists()


@pytest.mark.notransaction
@pytest.mark.storage_with_filesystem
def test_file_tx_rollback(filestore, gen, xom):
    filestore.keyfs.begin_transaction_in_thread(write=True)
    link = gen.pypi_package_link("pytest-1.8.zip", hash_spec=False)
    info = SimpleInfo.from_url(link)
    entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
    assert not entry.file_exists()
    content = b"123"
    entry.file_set_content(content, hashes=get_hashes(content), size=get_size(content))
    assert entry.file_exists()
    filepath = Path(entry.file_os_path(_raises=False))
    assert not filepath.exists()
    assert entry.file_get_content() == content
    filestore.keyfs.rollback_transaction_in_thread()
    assert not filepath.exists()


@pytest.mark.notransaction
def test_store_and_iter(filestore, model):
    with filestore.keyfs.write_transaction():
        user = model.create_user("user", "")
        model.create_stage(user, "index")
        content = b"hello"
        hashes = get_hashes(content)
        entry = filestore.store(
            "user",
            "index",
            "something-1.0.zip",
            content,
            hashes=hashes,
            size=get_size(content),
        )
        assert entry.hashes.get_default_spec() == hashes.get_default_spec()
        assert entry.file_exists()
    with filestore.keyfs.read_transaction():
        entry2 = filestore.get_file_entry(entry.relpath)
        assert entry2.basename == "something-1.0.zip"
        assert entry2.file_exists()
        assert entry2.best_available_hash_spec == entry.best_available_hash_spec
        assert entry2.hashes == entry.hashes
        assert entry2.last_modified
        assert entry2.file_get_content() == content


def test_maplink_nochange(filestore, gen, xom):
    filestore.keyfs.restart_as_write_transaction()
    link = gen.pypi_package_link("pytest-1.2.zip")
    info = SimpleInfo.from_url(link)
    entry1 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
    filestore.keyfs.commit_transaction_in_thread()
    last_serial = filestore.keyfs.get_current_serial()

    # start a new write transaction
    filestore.keyfs.begin_transaction_in_thread(write=True)
    info = SimpleInfo.from_url(link)
    entry2 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
    assert entry1.relpath == entry2.relpath
    assert entry1.basename == entry2.basename == "pytest-1.2.zip"
    assert isinstance(entry1.best_available_hash_spec, str)
    filestore.keyfs.commit_transaction_in_thread()
    assert filestore.keyfs.get_current_serial() == last_serial
