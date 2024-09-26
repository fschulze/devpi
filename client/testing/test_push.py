import pytest
import json
import textwrap
from devpi.main import Hub
from devpi.main import initmain
from devpi.push import parse_target, PyPIPush, DevpiPush
from io import StringIO
from subprocess import check_output


def runproc(cmd):
    args = cmd.split()
    return check_output(args)


def test_parse_target_devpi(loghub):
    class args:
        target = "user/name"
        index = None
    res = parse_target(loghub, args)
    assert isinstance(res, DevpiPush)


def test_parse_target_pypi(tmpdir, loghub):
    p = tmpdir.join("pypirc")
    p.write(textwrap.dedent("""
        [distutils]
        index-servers = whatever

        [whatever]
        repository: http://anotherserver
        username: test
        password: testp
    """))

    class args:
        target = "pypi:whatever"
        pypirc = str(p)
        index = None

    res = parse_target(loghub, args)
    assert isinstance(res, PyPIPush)
    assert res.user == "test"
    assert res.password == "testp"
    assert res.posturl == "http://anotherserver"


def test_parse_target_pypi_default_repository(tmpdir, loghub):
    p = tmpdir.join("pypirc")
    p.write(textwrap.dedent("""
        [distutils]
        index-servers = whatever

        [whatever]
        username: test
        password: testp
    """))

    class args:
        target = "pypi:whatever"
        pypirc = str(p)
        index = None

    res = parse_target(loghub, args)
    assert isinstance(res, PyPIPush)
    assert res.user == "test"
    assert res.password == "testp"
    assert res.posturl == "https://upload.pypi.org/legacy/"


def test_push_devpi(loghub, mock_http_api):
    class args:
        target = "user/name"
        index = None
    pusher = parse_target(loghub, args)
    mock_http_api.set(loghub.current.index, result={})
    pusher.execute(loghub, "pytest", "2.3.5", req_options={})
    dict(name="pytest", version="2.3.5", targetindex="user/name")
    ((methodname, _url, kwargs),) = mock_http_api.called
    assert methodname == "push"
    kvdict = dict(kwargs["kvdict"])
    assert kvdict.pop("name") == "pytest"
    assert kvdict.pop("version") == "2.3.5"
    assert kvdict.pop("targetindex") == "user/name"
    assert not kvdict


def test_push_devpi_no_docs(mock_http_api, tmpdir):
    out = StringIO()
    (hub, push) = initmain([
        "devpitest",
        "--clientdir", tmpdir.join("client").strpath,
        "push",
        "--no-docs",
        "pytest==2.3.5",
        "user/name"], file=out)
    mock_http_api.set(hub.current.index, result={})
    with pytest.raises(SystemExit):
        push(hub, hub.args)
    assert not mock_http_api.called
    assert "server doesn't support the '--no-docs' option" in out.getvalue()
    hub.current.features = {'push-no-docs'}
    push(hub, hub.args)
    ((methodname, _url, kwargs),) = mock_http_api.called
    assert methodname == "push"
    kvdict = dict(kwargs["kvdict"])
    assert kvdict.pop("name") == "pytest"
    assert kvdict.pop("version") == "2.3.5"
    assert kvdict.pop("targetindex") == "user/name"
    assert kvdict.pop("no_docs") is True
    assert not kvdict


def test_push_devpi_no_docs_and_only_docs(mock_http_api, tmpdir):
    out = StringIO()
    (hub, push) = initmain([
        "devpitest",
        "--clientdir", tmpdir.join("client").strpath,
        "push",
        "--no-docs",
        "--only-docs",
        "pytest==2.3.5",
        "user/name"], file=out)
    mock_http_api.set(hub.current.index, result={})
    with pytest.raises(SystemExit):
        push(hub, hub.args)
    assert not mock_http_api.called
    assert "Can't use '--no-docs' and '--only-docs' together." in out.getvalue()


def test_push_devpi_only_docs(mock_http_api, tmpdir):
    out = StringIO()
    (hub, push) = initmain([
        "devpitest",
        "--clientdir", tmpdir.join("client").strpath,
        "push",
        "--only-docs",
        "pytest==2.3.5",
        "user/name"], file=out)
    mock_http_api.set(hub.current.index, result={})
    with pytest.raises(SystemExit):
        push(hub, hub.args)
    assert not mock_http_api.called
    assert "server doesn't support the '--only-docs' option" in out.getvalue()
    hub.current.features = {'push-only-docs'}
    push(hub, hub.args)
    ((methodname, _url, kwargs),) = mock_http_api.called
    assert methodname == "push"
    kvdict = dict(kwargs["kvdict"])
    assert kvdict.pop("name") == "pytest"
    assert kvdict.pop("version") == "2.3.5"
    assert kvdict.pop("targetindex") == "user/name"
    assert kvdict.pop("only_docs") is True
    assert not kvdict


def test_push_devpi_index_option(loghub, mock_http_api):
    class args:
        target = "user/name"
        index = "src/dev"
    pusher = parse_target(loghub, args)
    mock_http_api.set("src/dev", result={})
    pusher.execute(loghub, "pytest", "2.3.5", req_options={})
    dict(name="pytest", version="2.3.5", targetindex="user/name")
    assert len(mock_http_api.called) == 1


def test_push_devpi_index_option_with_environment(loghub, monkeypatch, mock_http_api):
    loghub.args.target = "user/name"
    loghub.args.index = "src/dev"
    monkeypatch.setenv("DEVPI_INDEX", "http://devpi/user/dev")
    mock_http_api.set(
        "http://devpi/user/dev/+api", result=dict(
            pypisubmit="http://devpi/user/dev/",
            simpleindex="http://devpi/user/dev/+simple/",
            index="http://devpi/user/dev",
            login="http://devpi/+login",
            authstatus=["noauth", "", []]))
    mock_http_api.set(
        "http://devpi/src/dev/+api", result=dict(
            pypisubmit="http://devpi/src/dev/",
            simpleindex="http://devpi/src/dev/+simple/",
            index="http://devpi/src/dev",
            login="http://devpi/+login",
            authstatus=["noauth", "", []]))
    pusher = parse_target(loghub, loghub.args)
    mock_http_api.set("http://devpi/src/dev", result={})
    pusher.execute(loghub, "pytest", "2.3.5", req_options={})
    dict(name="pytest", version="2.3.5", targetindex="user/name")
    assert len(mock_http_api.called) == 3


@pytest.mark.parametrize("spec", ("pkg==1.0", "pkg-1.0"))
def test_main_push_pypi(capsys, monkeypatch, tmpdir, spec):
    from devpi.push import main
    l = []

    def mypost(method, url, data, headers, auth=None, cert=None, verify=None):
        l.append((method, url, data))

        class r:
            status_code = 201
            reason = "created"
            content = json.dumps(dict(
                type="actionlog", status=201,
                result=[("200", "register", "pkg", "1.0"),
                        ("200", "upload", "pkg-1.3.tar.gz")]
            ))
            headers = {"content-type": "application/json"}
            _json = json.loads(content)
        r.url = url
        return r

    class args:
        clientdir = tmpdir.join("client")
        debug = False
        index = None
        no_docs = False
        only_docs = False
        register_project = False
    hub = Hub(args)
    monkeypatch.setattr(hub.http, "request", mypost)
    hub.current.reconfigure(dict(index="/some/index"))
    p = tmpdir.join("pypirc")
    p.write(textwrap.dedent("""
        [distutils]
        index-servers = whatever

        [whatever]
        repository: http://anotherserver
        username: test
        password: testp
    """))

    class args:
        pypirc = str(p)
        target = "pypi:whatever"
        pkgspec = spec
        index = None
        no_docs = False
        only_docs = False
        register_project = False

    main(hub, args)
    assert len(l) == 1
    method, url, data = l[0]
    assert url == hub.current.index
    req = json.loads(data)
    assert req["name"] == "pkg"
    assert req["version"] == "1.0"
    assert req["posturl"] == "http://anotherserver"
    assert req["username"] == "test"
    assert req["password"] == "testp"

    class args:
        pypirc = str(p)
        target = "pypi:notspecified"
        pkgspec = spec
        index = None
        no_docs = False
        only_docs = False
        register_project = False

    (out, err) = capsys.readouterr()
    with pytest.raises(SystemExit):
        main(hub, args)
    (out, err) = capsys.readouterr()
    assert "Error while trying to read section 'notspecified'" in out
    assert "KeyError" in out


def test_push_pypi(mock_http_api, tmpdir):
    p = tmpdir.join("pypirc")
    p.write(textwrap.dedent("""
        [distutils]
        index-servers = whatever

        [whatever]
        repository: http://anotherserver
        username: test
        password: testp
    """))
    out = StringIO()
    (hub, push) = initmain([
        "devpitest",
        "--clientdir", tmpdir.join("client").strpath,
        "push",
        "--pypirc", str(p),
        "pytest==2.3.5",
        "pypi:whatever"], file=out)
    mock_http_api.set(hub.current.index, result={})
    push(hub, hub.args)
    ((methodname, _url, kwargs),) = mock_http_api.called
    assert methodname == "push"
    kvdict = dict(kwargs["kvdict"])
    assert kvdict.pop("name") == "pytest"
    assert kvdict.pop("version") == "2.3.5"
    assert kvdict.pop("username") == "test"
    assert kvdict.pop("password") == "testp"
    assert kvdict.pop("posturl") == "http://anotherserver"
    assert not kvdict


def test_push_pypi_register_project(mock_http_api, tmpdir):
    p = tmpdir.join("pypirc")
    p.write(textwrap.dedent("""
        [distutils]
        index-servers = whatever

        [whatever]
        repository: http://anotherserver
        username: test
        password: testp
    """))
    out = StringIO()
    (hub, push) = initmain([
        "devpitest",
        "--clientdir", tmpdir.join("client").strpath,
        "push",
        "--pypirc", str(p),
        "--register-project",
        "pytest==2.3.5",
        "pypi:whatever"], file=out)
    mock_http_api.set(hub.current.index, result={})
    with pytest.raises(SystemExit):
        push(hub, hub.args)
    assert not mock_http_api.called
    assert "server doesn't support the '--register-project' option" in out.getvalue()
    hub.current.features = {'push-register-project'}
    push(hub, hub.args)
    ((methodname, _url, kwargs),) = mock_http_api.called
    assert methodname == "push"
    kvdict = dict(kwargs["kvdict"])
    assert kvdict.pop("name") == "pytest"
    assert kvdict.pop("version") == "2.3.5"
    assert kvdict.pop("username") == "test"
    assert kvdict.pop("password") == "testp"
    assert kvdict.pop("posturl") == "http://anotherserver"
    assert kvdict.pop("register_project") is True
    assert not kvdict


def test_fail_push(monkeypatch, tmpdir):
    from devpi.push import main
    l = []

    def mypost(method, url, data, headers, auth=None, cert=None, verify=None):
        l.append((method, url, data))

        class r:
            status_code = 500
            reason = "Internal Server Error"
            content = json.dumps(dict(
                type="actionlog", status=201,
                result=[("500", "Internal Server Error", "Internal Server Error")]
            ))
            headers = {"content-type": "application/json"}
            _json = json.loads(content)

            class request:
                method = ''

        r.url = url
        r.request.method = method

        return r

    class args:
        clientdir = tmpdir.join("client")
        debug = False
        index = None
        no_docs = False
        only_docs = False
        register_project = False
    hub = Hub(args)
    monkeypatch.setattr(hub.http, "request", mypost)
    hub.current.reconfigure(dict(index="/some/index"))
    p = tmpdir.join("pypirc")
    p.write(textwrap.dedent("""
        [distutils]
        index-servers = whatever

        [whatever]
        repository: http://anotherserver
        username: test
        password: testp
    """))

    class args:
        pypirc = str(p)
        target = "pypi:whatever"
        pkgspec = "pkg==1.0"
        index = None
        no_docs = False
        only_docs = False
        register_project = False

    try:
        main(hub, args)
    except SystemExit as e:
        assert e.code==1


def test_derive_token_non_token():
    class MockHub:
        pass
    hub = MockHub()
    hub.derive_token = Hub.derive_token.__get__(hub)
    assert hub.derive_token("foo", None) == "foo"


@pytest.mark.parametrize("prefix", ["devpi", "pypi"])
def test_derive_token_invalid_token(prefix):
    msgs = []

    class MockHub:
        def info(self, msg):
            msgs.append(msg)

        def warn(self, msg):
            msgs.append(msg)
    hub = MockHub()
    hub.derive_token = Hub.derive_token.__get__(hub)
    assert hub.derive_token("%s-foo" % prefix, None) == "%s-foo" % prefix
    (msg,) = msgs
    assert "can not parse it" in msg


def test_derive_token():
    import pypitoken
    token = pypitoken.Token.create(
        domain="example.com",
        identifier="devpi",
        key="secret")
    passwd = token.dump()
    msgs = []

    class MockHub:
        def debug(self, *msg):
            pass

        def info(self, msg):
            msgs.append(msg)
    hub = MockHub()
    hub.derive_token = Hub.derive_token.__get__(hub)
    derived_passwd = hub.derive_token(passwd, 'pkg', now=10)
    assert derived_passwd != passwd
    (msg,) = msgs
    assert "create a unique PyPI token" in msg
    derived_token = pypitoken.Token.load(derived_passwd)
    assert sorted(derived_token.restrictions, key=lambda x: x.__class__.__name__) == [
        pypitoken.DateRestriction(not_before=9, not_after=70),
        pypitoken.ProjectNamesRestriction(project_names=["pkg"])]


def test_derive_legacy_token():
    import pypitoken
    token = pypitoken.Token.create(
        domain="example.com",
        identifier="devpi",
        key="secret")
    token.restrict(legacy_noop=True)
    passwd = token.dump()
    msgs = []

    class MockHub:
        def debug(self, *msg):
            pass

        def info(self, msg):
            msgs.append(msg)
    hub = MockHub()
    hub.derive_token = Hub.derive_token.__get__(hub)
    derived_passwd = hub.derive_token(passwd, 'pkg', now=10)
    assert derived_passwd != passwd
    (msg,) = msgs
    assert "create a unique PyPI token" in msg
    derived_token = pypitoken.Token.load(derived_passwd)
    assert sorted(derived_token.restrictions, key=lambda x: x.__class__.__name__) == [
        pypitoken.LegacyDateRestriction(not_before=9, not_after=70),
        pypitoken.LegacyNoopRestriction(),
        pypitoken.LegacyProjectNamesRestriction(project_names=["pkg"])]


def test_derive_devpi_token():
    import pypitoken
    passwd = "devpi-AgEAAhFmc2NodWx6ZS1yTlk5a0RuYQAABiBcjsOFkn7_3fn6mFoeJve_cOv-thDRL-4fQzbf_sOGjQ"
    msgs = []

    class MockHub:
        def debug(self, *msg):
            pass

        def info(self, msg):
            msgs.append(msg)
    hub = MockHub()
    hub.derive_token = Hub.derive_token.__get__(hub)
    derived_passwd = hub.derive_token(passwd, 'pkg', now=10)
    assert derived_passwd != passwd
    (msg,) = msgs
    assert "create a unique Devpi token" in msg
    derived_token = pypitoken.Token.load(derived_passwd)
    assert derived_token.restrictions == sorted(
        [
            pypitoken.DateRestriction(not_before=9, not_after=70),
            pypitoken.ProjectNamesRestriction(project_names=["pkg"])],
        key=lambda x: x.__class__.__name__)


class TestPush:
    def test_help(self, ext_devpi):
        result = ext_devpi("push", "-h")
        assert result.ret == 0
        result.stdout.fnmatch_lines("""
            *TARGET*
        """)
