from devpi_server.keyfs import KeyFS
from devpi_server.keyfs_sqlite2_fs import _iter_data_from_rows
from devpi_server.keyfs_types import FilePathInfo
from functools import partial


class DictKey:
    type = dict


class SetKey:
    type = set


def test_iter_data_from_rows():
    rows = [
        ('root/pypi/+f/de7/cdf515ea90ab0/AX3 model extras-1.0.2.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x03urlQ', b'N\x00\x00\x00\x91https://test-files.pythonhosted.org/packages/74/09/bf8717767cdd03d3ba59d698ab112614fd2be9e3a29abfaf4f3d2b8c54f2/AX3%20model%20extras-1.0.2.tar.gzQ'),
        ('root/pypi/+f/de7/cdf515ea90ab0/AX3 model extras-1.0.2.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07projectQ', b'N\x00\x00\x00\x10ax3-model-extrasQ'),
        ('root/pypi/+f/de7/cdf515ea90ab0/AX3 model extras-1.0.2.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07versionQ', b'N\x00\x00\x00\x051.0.2Q'),
        ('root/pypi/+f/de7/cdf515ea90ab0/AX3 model extras-1.0.2.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\thash_specQ', b'N\x00\x00\x00Gsha256=de7cdf515ea90ab05f145bc66e63ff4ba5e4f4d6d75d5db08db088eb0e1d7474Q'),
        ('root/pypi/+f/7fd/008c6703e8a45/AX3_model_extras-1.0.3-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x03urlQ', b'N\x00\x00\x00\x97https://test-files.pythonhosted.org/packages/9b/01/aa30a69e83d4ee484536fc34a96e53020decb5a5f5e1addbdbf2b0ac0486/AX3_model_extras-1.0.3-py3-none-any.whlQ'),
        ('root/pypi/+f/7fd/008c6703e8a45/AX3_model_extras-1.0.3-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07projectQ', b'N\x00\x00\x00\x10ax3-model-extrasQ'),
        ('root/pypi/+f/7fd/008c6703e8a45/AX3_model_extras-1.0.3-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07versionQ', b'N\x00\x00\x00\x051.0.3Q'),
        ('root/pypi/+f/7fd/008c6703e8a45/AX3_model_extras-1.0.3-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\thash_specQ', b'N\x00\x00\x00Gsha256=7fd008c6703e8a459359994d119d6ad9366b69aae38ca6382f0812320535847fQ'),
        ('root/pypi/+f/134/abd1b076e8a7d/AX3 model extras-1.0.3.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x03urlQ', b'N\x00\x00\x00\x91https://test-files.pythonhosted.org/packages/70/37/38d72593a7f9e6c15a096f5bc5ef6436b7c440edaf020397f8c4de621136/AX3%20model%20extras-1.0.3.tar.gzQ'),
        ('root/pypi/+f/134/abd1b076e8a7d/AX3 model extras-1.0.3.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07projectQ', b'N\x00\x00\x00\x10ax3-model-extrasQ'),
        ('root/pypi/+f/134/abd1b076e8a7d/AX3 model extras-1.0.3.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07versionQ', b'N\x00\x00\x00\x051.0.3Q'),
        ('root/pypi/+f/134/abd1b076e8a7d/AX3 model extras-1.0.3.tar.gz', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\thash_specQ', b'N\x00\x00\x00Gsha256=134abd1b076e8a7d4086ee1e43f291a33a2dcfc52d396c08f5392dfad1f8fb4bQ'),
        ('root/pypi/+f/0c3/095e22df11ca3/AX3_model_extras-1.0.2-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x03urlQ', b'N\x00\x00\x00\x97https://test-files.pythonhosted.org/packages/db/1f/c09df18ec9897b60677ed39b8a8fa30480812202d97184a02e994fc7e5ff/AX3_model_extras-1.0.2-py3-none-any.whlQ'),
        ('root/pypi/+f/0c3/095e22df11ca3/AX3_model_extras-1.0.2-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07projectQ', b'N\x00\x00\x00\x10ax3-model-extrasQ'),
        ('root/pypi/+f/0c3/095e22df11ca3/AX3_model_extras-1.0.2-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\x07versionQ', b'N\x00\x00\x00\x051.0.2Q'),
        ('root/pypi/+f/0c3/095e22df11ca3/AX3_model_extras-1.0.2-py3-none-any.whl', 'STAGEFILE', 363798, -1, -1, b'N\x00\x00\x00\thash_specQ', b'N\x00\x00\x00Gsha256=0c3095e22df11ca38995f7c12245dcba20f3138ea0a2b3e8157ea6cc0fbce610Q')]
    typed_keys = dict(STAGEFILE=DictKey())
    result = list(_iter_data_from_rows(typed_keys, rows, debug=True))
    expected = [
        (
            'root/pypi/+f/de7/cdf515ea90ab0/AX3 model extras-1.0.2.tar.gz',
            'STAGEFILE',
            363798,
            -1, dict(
                url='https://test-files.pythonhosted.org/packages/74/09/bf8717767cdd03d3ba59d698ab112614fd2be9e3a29abfaf4f3d2b8c54f2/AX3%20model%20extras-1.0.2.tar.gz',
                project='ax3-model-extras',
                version='1.0.2',
                hash_spec='sha256=de7cdf515ea90ab05f145bc66e63ff4ba5e4f4d6d75d5db08db088eb0e1d7474')),
        (
            'root/pypi/+f/7fd/008c6703e8a45/AX3_model_extras-1.0.3-py3-none-any.whl',
            'STAGEFILE',
            363798,
            -1, dict(
                url='https://test-files.pythonhosted.org/packages/9b/01/aa30a69e83d4ee484536fc34a96e53020decb5a5f5e1addbdbf2b0ac0486/AX3_model_extras-1.0.3-py3-none-any.whl',
                project='ax3-model-extras',
                version='1.0.3',
                hash_spec='sha256=7fd008c6703e8a459359994d119d6ad9366b69aae38ca6382f0812320535847f')),
        (
            'root/pypi/+f/134/abd1b076e8a7d/AX3 model extras-1.0.3.tar.gz',
            'STAGEFILE',
            363798,
            -1, dict(
                url='https://test-files.pythonhosted.org/packages/70/37/38d72593a7f9e6c15a096f5bc5ef6436b7c440edaf020397f8c4de621136/AX3%20model%20extras-1.0.3.tar.gz',
                project='ax3-model-extras',
                version='1.0.3',
                hash_spec='sha256=134abd1b076e8a7d4086ee1e43f291a33a2dcfc52d396c08f5392dfad1f8fb4b')),
        (
            'root/pypi/+f/0c3/095e22df11ca3/AX3_model_extras-1.0.2-py3-none-any.whl',
            'STAGEFILE',
            363798,
            -1, dict(
                url='https://test-files.pythonhosted.org/packages/db/1f/c09df18ec9897b60677ed39b8a8fa30480812202d97184a02e994fc7e5ff/AX3_model_extras-1.0.2-py3-none-any.whl',
                project='ax3-model-extras',
                version='1.0.2',
                hash_spec='sha256=0c3095e22df11ca38995f7c12245dcba20f3138ea0a2b3e8157ea6cc0fbce610'))]
    assert expected == result


def test_iter_data_empty_dict():
    rows = [
        ('user1/dev/hello/1.0/.config', 'PROJVERSION', 6, 5, -1, b'N\x00\x00\x00\x07versionQ', None)]
    typed_keys = dict(PROJVERSION=DictKey())
    result = list(_iter_data_from_rows(typed_keys, rows, debug=True))
    expected = [('user1/dev/hello/1.0/.config', 'PROJVERSION', 6, 5, dict())]
    assert expected == result


def test_iter_data_empty_set():
    rows = [
        ('hello/world2/hello/.versions', 'PROJVERSIONS', 18, 14, -1, b'N\x00\x00\x00\x031.0Q', None)]
    typed_keys = dict(PROJVERSIONS=SetKey())
    result = list(_iter_data_from_rows(typed_keys, rows, debug=True))
    expected = [('hello/world2/hello/.versions', 'PROJVERSIONS', 18, 14, set())]
    assert expected == result


def test_iter_data_from_rows_deleted():
    rows = [
        ('hpk/dev/+f/3fa/8515b0eac6193/detox-0.11.tar.gz', 'STAGEFILE', 123795, 123791, 123795, b'', None)]
    typed_keys = dict(STAGEFILE=DictKey())
    result = list(_iter_data_from_rows(typed_keys, rows, debug=True))
    expected = [('hpk/dev/+f/3fa/8515b0eac6193/detox-0.11.tar.gz', 'STAGEFILE', 123795, 123791, None)]
    assert expected == result


def test_iter_data_from_rows_deleted_and_set_again():
    rows = [
        ('hello', 'NAME', 2, 1, 1, b'F\x00\x00\x00\x02Q', b'F\x00\x00\x00\x02Q'),
        ('hello', 'NAME', 1, 0, 1, b'', None)]
    typed_keys = dict(NAME=DictKey())
    result = list(_iter_data_from_rows(typed_keys, rows, debug=True))
    expected = [('hello', 'NAME', 2, 1, {2: 2})]
    assert expected == result


def test_keyfs_sqlite2_fs(gen_path, file_digest, sorted_serverdir):
    from devpi_server import keyfs_sqlite2_fs
    from devpi_server.filestore_fs import FSIOFile
    tmp = gen_path()
    storage = keyfs_sqlite2_fs.Storage
    io_file_factory = partial(FSIOFile, settings={})
    keyfs = KeyFS(tmp, storage, io_file_factory=io_file_factory)
    content = b'bar'
    file_path_info = FilePathInfo('foo', file_digest(content))
    with keyfs.write_transaction() as tx:
        assert tx.io_file.os_path(file_path_info) == str(tmp / 'foo')
        tx.io_file.set_content(file_path_info, content)
        tx.conn._sqlconn.commit()
    with keyfs.read_transaction() as tx:
        assert tx.io_file.get_content(file_path_info) == content
        with open(tx.io_file.os_path(file_path_info), 'rb') as f:
            assert f.read() == content
    assert sorted_serverdir(tmp) == ['.sqlite2', 'foo']


def test_keyfs_sqlite2_hash(gen_path, file_digest, sorted_serverdir):
    from devpi_server import keyfs_sqlite2_fs
    from devpi_server.filestore_hash import HashIOFile
    tmp = gen_path()
    storage = keyfs_sqlite2_fs.Storage
    io_file_factory = partial(HashIOFile, settings={})
    keyfs = KeyFS(tmp, storage, io_file_factory=io_file_factory)
    content = b'bar'
    content_hash = file_digest(content)
    file_path_info = FilePathInfo('foo', content_hash)
    with keyfs.write_transaction() as tx:
        assert tx.io_file.os_path(file_path_info) == tmp.joinpath('+files', content_hash[:3], content_hash[3:])
        tx.io_file.set_content(file_path_info, content)
        tx.conn._sqlconn.commit()
    with keyfs.read_transaction() as tx:
        assert tx.io_file.get_content(file_path_info) == content
        with open(tx.io_file.os_path(file_path_info), 'rb') as f:
            assert f.read() == content
    assert sorted_serverdir(tmp) == ['+files', '.sqlite2']
    assert sorted(x.name for x in tmp.joinpath('+files').iterdir()) == [content_hash[:3]]
    assert sorted(x.name for x in tmp.joinpath('+files', content_hash[:3]).iterdir()) == [content_hash[3:]]
