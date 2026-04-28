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


@pytest.mark.writetransaction
class TestSimpleInfo:
    def test_maplink_deterministic(self, gen, xom):
        link = gen.pypi_package_link("pytest-1.2.zip")
        info = SimpleInfo.from_url(link)
        entry1 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        entry2 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry1.relpath == entry2.relpath
        assert entry1.basename == entry2.basename == "pytest-1.2.zip"
        assert isinstance(entry1.best_available_hash_spec, str)

    @pytest.mark.parametrize("hash_type", ["sha256", "md5"])
    def test_maplink_splithashdir_issue78(self, gen, hash_type, xom):
        link = gen.pypi_package_link("pytest-1.2.zip", hash_type=hash_type)
        info = SimpleInfo.from_url(link)
        entry1 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        # check md5 directory structure (issue78)
        parts = entry1.relpath.split("/")
        parent2 = parts[-2]
        parent1 = parts[-3]
        assert parent1 == link.hash_value[:3]
        assert parent2 == link.hash_value[3:16]
        assert hash_type == link.hash_type

    def test_maplink(self, gen, xom):
        link = gen.pypi_package_link("pytest-1.2.zip")
        info = SimpleInfo.from_url(link)
        entry1 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        entry2 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert not entry1.file_exists()
        assert not entry2.file_exists()
        assert entry1 == entry2
        assert entry1.relpath.endswith("/pytest-1.2.zip")
        assert entry1.best_available_hash_spec == link.hash_spec
        assert entry1.project == "pytest"

    @pytest.mark.parametrize(
        ("releasename", "project", "version"),
        [
            ("pytest-2.3.4.zip", "pytest", "2.3.4"),
            ("pytest-2.3.4-py27.egg", "pytest", "2.3.4"),
            ("dddttt-0.1.dev38-py2.7.egg", "dddttt", "0.1.dev38"),
            ("devpi-0.9.5.dev1-cp26-none-linux_x86_64.whl", "devpi", "0.9.5.dev1"),
            ("wheel-0.21.0-py2.py3-none-any.whl", "wheel", "0.21.0"),
            ("green-0.4.0-py2.5-win32.egg", "green", "0.4.0"),
            ("Candela-0.2.1.macosx-10.4-x86_64.exe", "Candela", "0.2.1"),
            (
                "Cambiatuscromos-0.1.1alpha.linux-x86_64.exe",
                "Cambiatuscromos",
                "0.1.1alpha",
            ),
            ("Aesthete-0.4.2.win32.exe", "Aesthete", "0.4.2"),
            ("DTL-1.0.5.win-amd64.exe", "DTL", "1.0.5"),
            ("Cheetah-2.2.2-1.x86_64.rpm", "Cheetah", "2.2.2"),
            ("Cheetah-2.2.2-1.src.rpm", "Cheetah", "2.2.2"),
            ("Cheetah-2.2.2-1.x85.rpm", "Cheetah", "2.2.2"),
            ("Cheetah-2.2.2.dev1-1.0.x85.rpm", "Cheetah", "2.2.2.dev1"),
            ("Cheetah-2.2.2.dev1-1.0.noarch.rpm", "Cheetah", "2.2.2.dev1"),
            ("deferargs.tar.gz", "", ""),
            ("hello-1.0.doc.zip", "hello", "1.0"),
            ("Twisted-12.0.0.win32-py2.7.msi", "Twisted", "12.0.0"),
            ("django_ipware-0.0.8-py3-none-any.whl", "django_ipware", "0.0.8"),
            (
                "my-binary-package-name-1-4-3-yip-0.9.tar.gz",
                "my-binary-package-name-1-4-3-yip",
                "0.9",
            ),
            (
                "my-binary-package-name-1-4-3-yip-0.9+deadbeef.tar.gz",
                "my-binary-package-name-1-4-3-yip",
                "0.9+deadbeef",
            ),
            ("cffi-1.6.0-pp251-pypy_41-macosx_10_11_x86_64.whl", "cffi", "1.6.0"),
            (
                "argon2_cffi-18.2.0.dev0.0-pp2510-pypy_41-macosx_10_13_x86_64.whl",
                "argon2_cffi",
                "18.2.0.dev0.0",
            ),
        ],
    )
    def test_maplink_project_version(self, gen, releasename, project, version, xom):
        link = gen.pypi_package_link(releasename)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", project)
        assert entry.relpath.endswith("/" + releasename)
        assert entry.project == project
        assert entry.version == version

    def test_maplink_project_bad_archive(self, gen, xom):
        link = gen.pypi_package_link("pytest-1.0.foo")
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.relpath.endswith("/pytest-1.0.foo")
        assert entry.project == "pytest"
        # the unknown file type prevents us from getting the version
        assert entry.version is None

    def test_maplink_replaced_release_not_cached_yet(self, gen, xom):
        link = gen.pypi_package_link("pytest-1.2.zip")
        info = SimpleInfo.from_url(link)
        entry1 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert not entry1.file_exists()
        assert entry1.best_available_hash_spec is not None
        assert entry1.best_available_hash_spec == link.hash_spec
        newlink = gen.pypi_package_link("pytest-1.2.zip")
        newinfo = SimpleInfo.from_url(newlink)
        entry2 = newinfo.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry2.best_available_hash_spec is not None
        assert entry2.best_available_hash_spec == newlink.hash_spec

    def test_maplink_replaced_release_already_cached(self, gen, xom):
        content1 = b"somedata"
        hashes1 = get_hashes(content1)
        md5_1 = get_hashes(content1, hash_types=("md5",))
        hashes1.update(md5_1)
        link1 = gen.pypi_package_link("pytest-1.2.zip", hash_spec=md5_1.get_spec("md5"))
        info1 = SimpleInfo.from_url(link1)
        entry1 = info1.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        # pseudo-write a release file with a specific hash_spec
        entry1.file_set_content(content1, hashes=hashes1, size=get_size(content1))
        assert entry1.file_exists()
        # make sure the entry has the same hash_spec as the external link
        assert entry1.best_available_hash_spec is not None
        assert entry1.hashes.get_spec("md5") == link1.hash_spec
        assert entry1.hashes.get_spec("md5") == link1.hash_spec

        # now replace the hash of the link and check again
        content2 = b"otherdata"
        md5_2 = get_hashes(content2, hash_types=("md5",))
        link2 = gen.pypi_package_link("pytest-1.2.zip", hash_spec=md5_2.get_spec("md5"))
        info2 = SimpleInfo.from_url(link2)
        entry2 = info2.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry2.best_available_hash_spec is not None
        assert entry2.best_available_hash_spec == link2.hash_spec
        assert entry2.hashes.get_spec("md5") == link2.hash_spec
        assert not entry2.file_exists()

    @pytest.mark.storage_with_filesystem
    def test_file_exists_new_hash(self, filestore, gen, xom):
        content1 = b"somedata"
        md5_1 = get_hashes(content1, hash_types=("md5",))
        link1 = gen.pypi_package_link("pytest-1.2.zip", hash_spec=md5_1.get_spec("md5"))
        info1 = SimpleInfo.from_url(link1)
        entry1 = info1.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        # write a wrong file outside the transaction
        filepath = Path(entry1.file_os_path())
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as f:
            f.write("othercontent")
        filestore.keyfs.rollback_transaction_in_thread()
        filestore.keyfs.begin_transaction_in_thread(write=True)
        # maplink doesn't validate the checksum (anymore)
        entry2 = info1.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry2.file_exists()
        content2 = entry2.file_get_content()
        assert content1 != content2
        assert isinstance(
            entry2.hashes.exception_for(content2, entry2.relpath), ValueError
        )
        assert entry2.hashes.exception_for(content1, entry2.relpath) is None
        filestore.keyfs.commit_transaction_in_thread()
        assert Path(filepath).exists()

    def test_file_delete(self, gen, xom):
        link = gen.pypi_package_link("pytest-1.2.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry1 = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        content = b""
        entry1.file_set_content(
            content, hashes=get_hashes(content), size=get_size(content)
        )
        assert entry1.file_exists()
        entry1.file_delete(is_last_of_hash=True)
        assert not entry1.file_exists()

    def test_relpathentry(self, filestore, gen, xom):
        link = gen.pypi_package_link("pytest-1.7.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.url == link.url
        assert not entry.file_exists()
        entry._hashes = hashes = get_hashes(b"")
        hash_spec = hashes.get_default_spec()
        assert not entry.file_exists()
        content = b""
        entry.file_set_content(content, hashes=hashes, size=get_size(content))
        assert entry.file_exists()
        assert entry.url == link.url
        assert entry.hashes.get_default_spec() == hashes.get_default_spec()

        # reget
        entry = filestore.get_mutable_file_entry(entry.relpath)
        assert entry.file_exists()
        assert entry.url == link.url
        assert entry.best_available_hash_spec == hash_spec
        assert entry.hashes == hashes
        entry.delete()
        assert not entry.file_exists()

    def test_iterfile_remote_no_headers(self, http, gen, xom):
        link = gen.pypi_package_link("pytest-1.8.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.best_available_hash_spec is None
        assert not entry.hashes
        headers = ResponseHeaders({})
        http.url2response[link.url] = dict(
            status_code=200, headers=headers, raw=BytesIO(b"123")
        )
        stage = xom.model.getstage("root/pypi")
        list(iter_cache_remote_file(stage, entry, entry.url))
        rheaders = entry.gethttpheaders()
        assert rheaders["content-length"] == "3"
        assert rheaders["content-type"] in zip_types
        assert entry.file_get_content() == b"123"

    def test_iterfile_remote_empty_content_type_header(self, http, gen, xom):
        link = gen.pypi_package_link("pytest-1.8.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.best_available_hash_spec is None
        assert not entry.hashes
        headers = ResponseHeaders({"Content-Type": ""})
        http.url2response[link.url] = dict(
            status_code=200, headers=headers, raw=BytesIO(b"123")
        )
        stage = xom.model.getstage("root/pypi")
        list(iter_cache_remote_file(stage, entry, entry.url))
        rheaders = entry.gethttpheaders()
        assert rheaders["content-length"] == "3"
        assert rheaders["content-type"] in zip_types
        assert entry.file_get_content() == b"123"

    def test_iterfile_remote_error_size_mismatch(self, http, gen, xom):
        link = gen.pypi_package_link("pytest-3.0.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.best_available_hash_spec is None
        assert not entry.hashes
        headers = ResponseHeaders(
            {
                "content-length": "3",
                "last-modified": "Thu, 25 Nov 2010 20:00:27 GMT",
                "content-type": "application/zip",
            }
        )
        stage = xom.model.getstage("root/pypi")
        http.url2response[link.url] = dict(
            status_code=200, headers=headers, raw=BytesIO(b"1")
        )
        with pytest.raises(
            ValueError, match=r"got 1 bytes of.*from remote, expected 3"
        ):
            list(iter_cache_remote_file(stage, entry, entry.url))

    def test_iterfile_remote_nosize(self, filestore, http, gen, xom):
        link = gen.pypi_package_link("pytest-3.0.zip", hash_spec=False)
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.best_available_hash_spec is None
        assert not entry.hashes
        headers = ResponseHeaders(
            {"last-modified": "Thu, 25 Nov 2010 20:00:27 GMT", "content-length": ""}
        )
        assert entry.file_size() is None
        http.url2response[link.url] = dict(
            status_code=200, headers=headers, raw=BytesIO(b"1")
        )
        stage = xom.model.getstage("root/pypi")
        list(iter_cache_remote_file(stage, entry, entry.url))
        assert entry.file_get_content() == b"1"
        entry2 = filestore.get_file_entry(entry.relpath)
        assert entry2.file_size() == 1
        rheaders = entry.gethttpheaders()
        assert rheaders["last-modified"] == headers["last-modified"]
        assert rheaders["content-type"] in zip_types

    def test_iterfile_remote_error_md5(self, http, gen, xom):
        link = gen.pypi_package_link("pytest-3.0.zip")
        info = SimpleInfo.from_url(link)
        entry = info.make_mutable_entry(xom.keyfs.schema, "root", "pypi", "pytest")
        assert entry.best_available_hash_spec is not None
        assert entry.best_available_hash_spec == link.hash_spec
        assert entry.hashes
        headers = ResponseHeaders(
            {
                "content-length": "3",
                "last-modified": "Thu, 25 Nov 2010 20:00:27 GMT",
                "content-type": "application/zip",
            }
        )
        http.url2response[link.url_nofrag] = dict(
            status_code=200, headers=headers, raw=BytesIO(b"123")
        )
        stage = xom.model.getstage("root/pypi")
        with pytest.raises(ValueError, match=link.md5):
            list(iter_cache_remote_file(stage, entry, entry.url))
        assert not entry.file_exists()
