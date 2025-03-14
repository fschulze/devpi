import os
import subprocess
import pytest
import sys
from devpi_common.viewhelp import ViewLinkStore
from devpi.test import DevIndex
from devpi.test import find_sdist_and_wheels
from devpi.test import prepare_toxrun_args
from devpi.test import post_tox_json_report
from pathlib import Path
from textwrap import dedent


def test_post_tox_json_report(loghub, mock_http_api):
    mock_http_api.set("http://devpi.net", result={})
    post_tox_json_report(loghub, "http://devpi.net", {"hello": "123"})
    assert len(mock_http_api.called) == 1
    loghub._getmatcher().fnmatch_lines("""
        *posting*
        *success*
    """)
    loghub._getmatcher().no_fnmatch_line('*200 OK*')


def test_post_tox_json_report_skip(loghub, mock_http_api):
    mock_http_api.set("http://devpi.net", message="custom skip")
    post_tox_json_report(loghub, "http://devpi.net", {"hello": "123"})
    assert len(mock_http_api.called) == 1
    loghub._getmatcher().fnmatch_lines("""
        *posting*
        *success*
        custom skip
    """)
    loghub._getmatcher().no_fnmatch_line('*200 OK*')


def test_post_tox_json_report_error(loghub, mock_http_api):
    mock_http_api.set("http://devpi.net/+tests", reason="Not Found", status=404)
    with pytest.raises(SystemExit) as excinfo:
        post_tox_json_report(loghub, "http://devpi.net/+tests", {"hello": "123"})
    assert excinfo.value.code == 1
    assert len(mock_http_api.called) == 1
    loghub._getmatcher().fnmatch_lines("""
        *posting*
        *404 Not Found*
    """)


def test_post_tox_json_report_forbidden(loghub, mock_http_api):
    mock_http_api.set("http://devpi.net/foo/bar/", reason="Forbidden", status=403)
    with pytest.raises(SystemExit) as excinfo:
        post_tox_json_report(loghub, "http://devpi.net/foo/bar/", {"hello": "123"})
    assert excinfo.value.code == 1
    assert len(mock_http_api.called) == 1
    loghub._getmatcher().fnmatch_lines("""
        *posting*
        *403 Forbidden
    """)


def test_post_tox_json_report_forbidden_msg(loghub, mock_http_api):
    mock_http_api.set(
        "http://devpi.net/foo/bar/", reason="Forbidden", status=403,
        message="custom forbidden")
    with pytest.raises(SystemExit) as excinfo:
        post_tox_json_report(loghub, "http://devpi.net/foo/bar/", {"hello": "123"})
    assert excinfo.value.code == 1
    assert len(mock_http_api.called) == 1
    loghub._getmatcher().fnmatch_lines("""
        *posting*
        *403 Forbidden: custom forbidden
    """)


@pytest.fixture
def pseudo_current():
    class Current:
        simpleindex = "http://pseudo/user/index/"
    return Current


def contains_sublist(list1, sublist):
    len_sublist = len(sublist)
    assert len_sublist <= len(list1)
    for i in range(len(list1)):
        if list1[i:i+len_sublist] == sublist:
            return True
    return False


def test_passthrough_args_toxargs(makehub, tmpdir, pseudo_current):
    hub = makehub(["test", "--tox-args", "-- -x", "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    tmpdir.ensure("tox.ini")
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert args[-2:] == ["--", "-x"]


@pytest.mark.parametrize("other_index", ["root/pypi", "/"])
def test_index_option(create_and_upload, devpi, monkeypatch, out_devpi, other_index):
    import re

    def runtox(self, *args, **kwargs):
        self.hub.info("Mocked tests ... %r %r" % (args, kwargs))
    monkeypatch.setattr("devpi.test.DevIndex.runtox", runtox)

    create_and_upload("exa-1.0")

    # remember username
    out = out_devpi("use")
    (url, user) = re.search(
        r'(https?://.+?)\s+\(logged in as (.+?)\)', out.stdout.str()).groups()

    # go to other index
    devpi("use", other_index)

    out = out_devpi("test", "--index", "%s/dev" % user, "exa")
    out.stdout.fnmatch_lines("""
        received*/%s/dev/*exa-1.0*
        unpacking*
        Mocked tests ...*""" % user)

    # forget current server index
    devpi("use", "--delete")

    out = out_devpi("test", "--index", url, "exa")
    out.stdout.fnmatch_lines("""
        received*/%s/dev/*exa-1.0*
        unpacking*
        Mocked tests ...*""" % user)


@pytest.mark.parametrize('basic_auth', [None, ('root', 'verysecret')])
def test_download_and_unpack(makehub, tmpdir, pseudo_current, monkeypatch,
                             basic_auth):
    class FakeHTTP(object):
        class Response(object):
            def __init__(self, content=b'archive'):
                self.status_code = 200
                self.content = content

        def __init__(self):
            self.last_get = None

        def get(self, *args, **kwargs):
            self.last_get = (args, kwargs)
            return self.Response()

    class FakeUnpack(object):
        def __init__(self):
            self.called = False

        def unpack(self):
            self.called = True

    hub = makehub(['test', '-epy27', 'somepkg'])
    hub.current.reconfigure(dict(
        index='http://dev/foo/bar',
        login='http://dev/+login',
        pypisubmit='http://dev/foo/bar'))
    if basic_auth:
        hub.current.set_basic_auth(*basic_auth)
    index = DevIndex(hub, tmpdir, pseudo_current)

    fake_http = FakeHTTP()
    hub.http.get = fake_http.get
    fake_unpack = FakeUnpack()
    monkeypatch.setattr('devpi.test.UnpackedPackage.unpack',
                        fake_unpack.unpack)

    links = [
        {'href': 'http://dev/foo/bar/prep1-1.0.tar.gz', 'rel': 'releasefile'},
    ]
    store = ViewLinkStore('http://something/index',
                          {'+links': links, 'name': 'prep1', 'version': '1.0'})
    link = store.get_link(rel='releasefile')

    index.download_and_unpack('1.0', link)
    assert fake_unpack.called
    args, kwargs = fake_http.last_get
    assert args[0] == 'http://dev/foo/bar/prep1-1.0.tar.gz'
    if basic_auth:
        assert kwargs['auth'] == basic_auth
    else:
        assert kwargs.get('auth') is None


def test_toxini(makehub, tmpdir, pseudo_current):
    toxini = tmpdir.ensure("new-tox.ini")
    hub = makehub(["test", "-c", toxini, "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    tmpdir.ensure("tox.ini")
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(toxini)])


def test_passthrough_args_env(makehub, tmpdir, pseudo_current):
    hub = makehub(["test", "-epy27", "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    tmpdir.ensure("tox.ini")
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-epy27"])


def test_fallback_ini(makehub, tmpdir, pseudo_current):
    p = tmpdir.ensure("mytox.ini")
    hub = makehub(["test", "--fallback-ini", str(p), "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(p)])
    p2 = tmpdir.ensure("tox.ini")
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(p2)])


def test_fallback_ini_relative(makehub, tmpdir, pseudo_current):
    p = tmpdir.ensure("mytox.ini")
    hub = makehub(["test", "--fallback-ini", "mytox.ini", "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(p)])
    p2 = tmpdir.ensure("tox.ini")
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(p2)])


def test_pyproject_toml(makehub, tmpdir, pseudo_current):
    p = tmpdir.join("pyproject.toml")
    p.write_text(dedent("""
        [project]
        keywords = [
            "foo",
            "bar",
        ]

        [tool.tox]
        """), "utf-8")
    hub = makehub(["test", "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(p)])


def test_setup_cfg(makehub, tmpdir, pseudo_current):
    p = tmpdir.join("setup.cfg")
    p.write_text("[tox:tox]", "utf-8")
    hub = makehub(["test", "somepkg"])
    index = DevIndex(hub, tmpdir, pseudo_current)
    args = index.get_tox_args(unpack_path=Path(tmpdir.strpath))
    assert contains_sublist(args, ["-c", str(p)])


class TestWheel:
    def test_find_wheels_and_sdist(self, loghub):
        vl = ViewLinkStore("http://something/index", {"+links": [
            {"href": "http://b/pytest-2.7.0.zip", "rel": "releasefile"},
            {"href": "http://b/pytest-2.7.0.tar.gz", "rel": "releasefile"},
            {"href": "http://b/pytest-2.7.0-py2.py3-none-any.whl", "rel": "releasefile"},
        ]})
        links = vl.get_links(rel="releasefile")
        sdist_links, wheel_links = find_sdist_and_wheels(loghub, links)
        assert len(sdist_links) == 2
        assert sdist_links[0].basename.endswith(".tar.gz")
        assert sdist_links[1].basename.endswith(".zip")
        assert len(wheel_links) == 1
        assert wheel_links[0].basename == "pytest-2.7.0-py2.py3-none-any.whl"

    def test_find_wheels_and_no_sdist(self, loghub):
        vl = ViewLinkStore("http://something/index", {"+links": [
            {"href": "http://b/pytest-2.7.0-py2.py3-none-any.whl", "rel": "releasefile"},
        ]})
        links = vl.get_links(rel="releasefile")
        with pytest.raises(SystemExit):
            find_sdist_and_wheels(loghub, links)

        loghub._getmatcher().fnmatch_lines("""
            *need at least one sdist*
        """)

    def test_find_wheels_not_universal(self, loghub):
        vl = ViewLinkStore("http://something/index", {"+links": [
            {"href": "http://b/pytest-2.7.0.tar.gz", "rel": "releasefile"},
            {"href": "http://b/pytest-2.7.0-py26-none-any.whl", "rel": "releasefile"},
        ]})
        links = vl.get_links(rel="releasefile")
        (sdist_links, wheel_links) = find_sdist_and_wheels(loghub, links)
        assert len(sdist_links) == 1
        assert sdist_links[0].basename.endswith(".tar.gz")
        assert len(wheel_links) == 0
        loghub._getmatcher().fnmatch_lines("""
            *only universal wheels*
        """)

    def test_find_wheels_non_universal(self, loghub):
        vl = ViewLinkStore("http://something/index", {"+links": [
            {"href": "http://b/pytest-2.7.0.tar.gz", "rel": "releasefile"},
            {"href": "http://b/pytest-2.7.0-py2-none-any.whl", "rel": "releasefile"},
        ]})
        links = vl.get_links(rel="releasefile")
        (sdist_links, wheel_links) = find_sdist_and_wheels(
            loghub, links, universal_only=False)
        assert len(sdist_links) == 1
        assert sdist_links[0].basename.endswith(".tar.gz")
        assert len(wheel_links) == 1
        assert wheel_links[0].basename.endswith("py2-none-any.whl")
        assert 'only universal wheels' not in '\n'.join(loghub._getmatcher().lines)

    @pytest.mark.skipif("config.option.fast")
    @pytest.mark.parametrize("pkgname", (
        "prep1",
        "prep-dash",
        "prep_under",
        "prep.dot",
        "prep.dot-dash",
        "prep.dot_under",
        "Upper"))
    def test_prepare_toxrun_args(self, loghub, pkgname, pseudo_current, tmpdir, reqmock, initproj):
        initproj((pkgname, "1.0"), filedefs={})
        subprocess.check_call(["python", "setup.py", "sdist", "--formats=gztar,zip"])
        subprocess.check_call(["python", "setup.py", "bdist_wheel", "--universal"])
        vl_links = []
        for p in Path("dist").iterdir():
            url = f"http://b/{p.name}"
            vl_links.append(dict(href=url, rel="releasefile"))
            if 'py2.py3' in url:
                vl_links.append(dict(href=url.replace('py2.py3', 'py2'), rel="releasefile"))
                vl_links.append(dict(href=url.replace('py2.py3', 'py3'), rel="releasefile"))
            reqmock.mockresponse(
                url,
                code=200, data=p.read_bytes(), method="GET")
        vl = ViewLinkStore(
            "http://something/index",
            {"+links": vl_links, "name": pkgname, "version": "1.0"})
        links = vl.get_links(rel="releasefile")
        sdist_links, wheel_links = find_sdist_and_wheels(loghub, links)
        dev_index = DevIndex(loghub, tmpdir, pseudo_current)
        toxrunargs = prepare_toxrun_args(dev_index, vl, sdist_links, wheel_links)
        assert len(toxrunargs) == 3
        sdist1, sdist2, wheel1 = toxrunargs
        pkgname_norm = pkgname.replace('-', '_').replace('.', '_').lower()
        pkgname_whl = pkgname.replace('-', '_')
        assert sdist1[0].basename in {
            f"{pkgname}-1.0.tar.gz",
            f"{pkgname_norm}-1.0.tar.gz"}
        assert str(sdist1[1].path_unpacked).endswith((
            "targz" + os.sep + f"{pkgname}-1.0",
            "targz" + os.sep + f"{pkgname_norm}-1.0"))
        assert sdist2[0].basename in {
            f"{pkgname}-1.0.zip",
            f"{pkgname_norm}-1.0.zip"}
        assert str(sdist2[1].path_unpacked).endswith((
            "zip" + os.sep + f"{pkgname}-1.0",
            "zip" + os.sep + f"{pkgname_norm}-1.0"))
        assert wheel1[0].basename in {
            f"{pkgname}-1.0-py2.py3-none-any.whl",
            f"{pkgname_norm}-1.0-py2.py3-none-any.whl",
            f"{pkgname_whl}-1.0-py2.py3-none-any.whl"}
        assert str(wheel1[1].path_unpacked).endswith(wheel1[0].basename)

    @pytest.mark.skipif("config.option.fast")
    def test_prepare_toxrun_args_select(self, loghub, pseudo_current, tmpdir, reqmock, initproj):
        # test that we can explicitly select a non universal wheel
        pyver = "py%s" % sys.version_info[0]
        vl = ViewLinkStore("http://something/index", {"+links": [
            {"href": "http://b/prep_under-1.0.tar.gz", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0-%s-none-any.whl" % pyver, "rel": "releasefile"},
        ], "name": "prep-under", "version": "1.0"})
        links = vl.get_links(rel="releasefile")
        sdist_links, wheel_links = find_sdist_and_wheels(
            loghub, links, universal_only=False)
        dev_index = DevIndex(loghub, tmpdir, pseudo_current)

        initproj("prep_under-1.0", filedefs={})
        subprocess.check_call(["python", "setup.py", "sdist", "--formats=gztar"])
        subprocess.check_call(["python", "setup.py", "bdist_wheel"])
        for p in Path("dist").iterdir():
            reqmock.mockresponse(
                f"http://b/{p.name}",
                code=200, data=p.read_bytes(), method="GET")
        toxrunargs = prepare_toxrun_args(
            dev_index, vl, sdist_links, wheel_links, select=pyver)
        assert len(toxrunargs) == 1
        (wheel1,) = toxrunargs
        assert wheel1[0].basename == "prep_under-1.0-%s-none-any.whl" % pyver
        assert str(wheel1[1].path_unpacked).endswith(wheel1[0].basename)

    def test_wheels_only_download_selected(self, loghub, monkeypatch, pseudo_current, reqmock, tmpdir):
        vl = ViewLinkStore("http://something/index", {"+links": [
            {"href": "http://b/prep_under-1.0-cp37-cp37m-macosx_10_10_x86_64.whl", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0-cp38-cp38-macosx_11_0_arm64.whl", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0-cp39-cp39-macosx_11_0_arm64.whl", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "rel": "releasefile"},
            {"href": "http://b/prep_under-1.0.tar.gz", "rel": "releasefile"},
        ], "name": "prep-under", "version": "1.0"})
        links = vl.get_links(rel="releasefile")
        (sdist_links, wheel_links) = find_sdist_and_wheels(
            loghub, links, universal_only=False)
        dev_index = DevIndex(loghub, tmpdir, pseudo_current)
        monkeypatch.setattr("devpi.test.UnpackedPackage.unpack", lambda s: None)
        reqmock.mockresponse(
            "http://b/prep_under-1.0.tar.gz",
            code=200, method="GET", data=b"source")
        reqmock.mockresponse(
            "http://b/prep_under-1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            code=200, method="GET", data=b"wheel")

        toxrunargs = prepare_toxrun_args(
            dev_index, vl, sdist_links, wheel_links,
            select="(?:.*39)(?:.*linux)(?:.*whl)")
        ((wheel_link, wheel, wheel_sdist),) = toxrunargs
        assert wheel_link.basename == "prep_under-1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        assert wheel.path_archive.name == "prep_under-1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        assert wheel_sdist.path_archive.name == "prep_under-1.0.tar.gz"

    def test_wheels_and_sdist(self, out_devpi, create_and_upload):
        create_and_upload("exa-1.0", filedefs={
            "tox.ini": """
              [testenv]
              commands = python -c "print('ok')"
            """,
            "setup.cfg": """
                [bdist_wheel]
                universal = True
            """})
        result = out_devpi("test", "-epy", "--debug", "exa==1.0")
        assert result.ret == 0
        result.stdout.fnmatch_lines("""*exa-1.0.*""")
        result = out_devpi("list", "-f", "exa")
        assert result.ret == 0
        result.stdout.re_match_lines_random(r"""
            .*exa-1\.0-.+\.whl
            .*tests.*passed
            .*exa-1\.0\.(tar\.gz|zip)
            .*tests.*passed
        """)


class TestFunctional:
    def test_main_nopackage(self, capfd, out_devpi):
        out_devpi("test", "--debug", "notexists73", ret=1)
        (out, err) = capfd.readouterr()
        assert "could not find/receive links for notexists73" in out

    def test_main_example(self, out_devpi, create_and_upload):
        result = out_devpi("index", "bases=root/pypi")
        assert result.ret == 0
        create_and_upload("exa-1.0", filedefs={
            "tox.ini": """
              [testenv]
              commands = python -c "print('ok')"
            """,
        })
        result = out_devpi("test", "--debug", "exa")
        assert result.ret == 0
        result = out_devpi("list", "-f", "exa")
        assert result.ret == 0
        result.stdout.fnmatch_lines("""*tests passed*""")

    def test_main_example_with_basic_auth(self, initproj, devpi, out_devpi):
        initproj('exa-1.0', {
            'tox.ini': """
            [testenv]
            commands = python -c "print('ok')"
            """,
        })
        hub = devpi('upload')
        hub.current.set_basic_auth('root', 'verysecret')

        result = out_devpi('test', 'exa')
        assert result.ret == 0
        expected_output = (
            'Using existing basic auth*',
            '*password*might be exposed*',
            '*//root:verysecret@*',
        )
        result.stdout.fnmatch_lines(expected_output)

    def test_no_post(self, out_devpi, create_and_upload, monkeypatch):
        def post(*args, **kwargs):
            0 / 0  # noqa: B018

        create_and_upload("exa-1.0", filedefs={
            "tox.ini": """
              [testenv]
              commands = python -c "print('ok')"
            """})
        monkeypatch.setattr("devpi.test.post_tox_json_report", post)
        result = out_devpi("test", "--no-upload", "exa")
        assert result.ret == 0

    def test_specific_version(self, out_devpi, create_and_upload):
        result = out_devpi("index", "bases=root/pypi")
        assert result.ret == 0
        create_and_upload("exa-1.0", filedefs={
            "tox.ini": """
              [testenv]
              commands = python -c "print('ok')"
            """,
        })
        create_and_upload("exa-1.1", filedefs={
            "tox.ini": """
              [testenv]
              commands = python -c "print('ok')"
            """,
        })
        result = out_devpi("test", "--debug", "exa==1.0")
        assert result.ret == 0
        result.stdout.fnmatch_lines("""*exa-1.0.*""")
        result = out_devpi("list", "-f", "exa")
        assert result.ret == 0
        result.stdout.fnmatch_lines("""
            *exa-1.1.*
            *exa-1.0.*
            *tests passed*""")

    def test_pkgname_with_dashes(self, out_devpi, create_and_upload):
        result = out_devpi("index", "bases=root/pypi")
        assert result.ret == 0
        create_and_upload(("my-pkg-123", "1.0"), filedefs={
            "tox.ini": """
              [testenv]
              commands = python -c "print('ok')"
            """,
        })
        result = out_devpi("test", "--debug", "my-pkg-123")
        assert result.ret == 0
        result = out_devpi("list", "-f", "my-pkg-123")
        assert result.ret == 0
        result.stdout.fnmatch_lines("""*tests passed*""")

    def test_test_hides_auth_in_url(self, capsys, create_and_upload, monkeypatch, devpi):
        create_and_upload(("foo", "1.0"), filedefs={
            "tox.ini": """
                [testenv]
                commands = python -c "print('ok')"
            """})
        calls = []

        def subprocess_call(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr('subprocess.call', subprocess_call)
        devpi("test", "--no-upload", "foo")
        assert len(calls) == 1
        (out, err) = capsys.readouterr()
        (line,) = [x for x in out.splitlines() if 'PIP_INDEX_URL' in x]
        expected = 'http://****:****@localhost'
        assert expected in line
