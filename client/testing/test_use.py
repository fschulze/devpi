from devpi.use import BuildoutCfg
from devpi.use import DistutilsCfg
from devpi.use import PipCfg
from devpi.use import UvConf
from devpi.use import Current, PersistentCurrent
from devpi.use import get_keyvalues, out_index_list
import pytest
import re
import requests.exceptions
import urllib3.exceptions


def test_ask_confirm(makehub, monkeypatch):
    import devpi.main
    hub = makehub(["remove", "something"])
    monkeypatch.setattr(devpi.main, "raw_input", lambda msg: "yes",
                        raising=False)
    assert hub.ask_confirm("hello")
    monkeypatch.setattr(devpi.main, "raw_input", lambda msg: "no")
    assert not hub.ask_confirm("hello")
    l = ["yes", "qwoeiu"]
    monkeypatch.setattr(devpi.main, "raw_input", lambda msg: l.pop())
    assert hub.ask_confirm("hello")


def test_ask_confirm_delete_args_yes(makehub):
    hub = makehub(["remove", "-y", "whatever"])
    assert hub.ask_confirm("hello")


class TestUnit:
    def test_write_and_read(self, tmpdir):
        auth_path = tmpdir.join("auth")
        current_path = tmpdir.join("current")
        current = PersistentCurrent(auth_path, current_path)
        assert not current.simpleindex
        current.reconfigure(dict(
            pypisubmit="/post",
            simpleindex="/index",
            login="/login",
        ))
        assert current.simpleindex
        newcurrent = PersistentCurrent(auth_path, current_path)
        assert newcurrent.pypisubmit == current.pypisubmit
        assert newcurrent.simpleindex == current.simpleindex
        assert newcurrent.venvdir == current.venvdir
        assert newcurrent.login == current.login

    def test_read_empty(self, tmpdir):
        auth_path = tmpdir.join("auth").ensure()
        current_path = tmpdir.join("current").ensure()
        current = PersistentCurrent(auth_path, current_path)
        assert current._authdict == {}
        assert current._currentdict == {}

    def test_write_and_read_always_setcfg(self, tmpdir):
        auth_path = tmpdir.join("auth")
        current_path = tmpdir.join("current")
        current = PersistentCurrent(auth_path, current_path)
        assert not current.simpleindex
        current.reconfigure(dict(
            pypisubmit="/post",
            simpleindex="/index",
            login="/login",
        ))
        assert current.simpleindex
        current.reconfigure(dict(always_setcfg=True))
        newcurrent = PersistentCurrent(auth_path, current_path)
        assert newcurrent.always_setcfg is True
        newcurrent.reconfigure(data=dict(simpleindex="/index2"))
        current = PersistentCurrent(auth_path, current_path)
        assert current.always_setcfg
        assert current.simpleindex == "/index2"

    def test_local_config(self, capfd, cmd_devpi, create_venv, mock_http_api, monkeypatch):
        import json
        venvdir = create_venv()
        (out, err) = capfd.readouterr()
        monkeypatch.setenv("VIRTUAL_ENV", venvdir.strpath)
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                index="/foo/bar",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://devpi/foo/bar?no_projects=", result=dict())
        hub = cmd_devpi("use", "http://devpi/foo/bar")
        (out, err) = capfd.readouterr()
        current_path = hub.current_path
        assert current_path.name == 'current.json'
        assert current_path.parent.name == 'client'
        local_current_path = hub.local_current_path
        assert venvdir.strpath in str(local_current_path)
        assert local_current_path.name == 'devpi.json'
        assert not local_current_path.exists()
        assert venvdir.strpath in out
        hub = cmd_devpi("use", "--local")
        (out, err) = capfd.readouterr()
        assert str(hub.current_path) == str(local_current_path)
        assert local_current_path.exists()
        mock_http_api.set(
            "http://devpi/foo/ham/+api", result=dict(
                index="/foo/ham",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://devpi/foo/ham?no_projects=", result=dict())
        hub = cmd_devpi("use", "http://devpi/foo/ham")
        (out, err) = capfd.readouterr()
        with current_path.open("r") as f:
            current_config = json.load(f)
        with local_current_path.open("r") as f:
            local_current_config = json.load(f)
        assert current_config['index'] == 'http://devpi/foo/bar'
        assert local_current_config['index'] == 'http://devpi/foo/ham'
        hub = cmd_devpi("use")
        (out, err) = capfd.readouterr()
        assert 'current devpi index: http://devpi/foo/ham' in out
        local_current_path.unlink()
        hub = cmd_devpi("use")
        (out, err) = capfd.readouterr()
        assert 'current devpi index: http://devpi/foo/bar' in out

    def test_local_config_no_auth_key(self, cmd_devpi, create_venv, monkeypatch):
        # test that the legacy ``auth`` key is removed
        import json
        venvdir = create_venv()
        monkeypatch.setenv("VIRTUAL_ENV", venvdir.strpath)
        hub = cmd_devpi("use")
        current_path = hub.current_path
        assert current_path.name == 'current.json'
        assert current_path.parent.name == 'client'
        local_current_path = hub.local_current_path
        assert not local_current_path.exists()
        with current_path.open("w") as f:
            json.dump(dict(auth=[]), f)
        hub = cmd_devpi("use", "--local")
        assert local_current_path.exists()
        with local_current_path.open("r") as f:
            local_current_config = json.load(f)
        assert 'auth' not in local_current_config

    def test_rewritten_url_scheme(self, capfd, cmd_devpi, mock_http_api):
        from devpi_common.url import URL
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                index="https://devpi/foo/bar",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "https://devpi/foo/bar?no_projects=", result=dict())
        hub = cmd_devpi("use", "http://devpi/foo/bar")
        (out, err) = capfd.readouterr()
        assert 'The server has rewritten the url to: https://devpi/foo/bar\n' in out
        assert hub.current.index_url == URL('https://devpi/foo/bar')

    def test_rewritten_url_host(self, capfd, cmd_devpi, mock_http_api):
        from devpi_common.url import URL
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                index="http://127.0.0.1/foo/bar",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://127.0.0.1/foo/bar?no_projects=", result=dict())
        hub = cmd_devpi("use", "http://devpi/foo/bar")
        (out, err) = capfd.readouterr()
        assert 'The server has rewritten the url to: http://127.0.0.1/foo/bar\n' in out
        assert hub.current.index_url == URL('http://127.0.0.1/foo/bar')

    def test_rewritten_url_port(self, capfd, cmd_devpi, mock_http_api):
        from devpi_common.url import URL
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                index="http://devpi:3141/foo/bar",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://devpi:3141/foo/bar?no_projects=", result=dict())
        hub = cmd_devpi("use", "http://devpi/foo/bar")
        (out, err) = capfd.readouterr()
        assert 'The server has rewritten the url to: http://devpi:3141/foo/bar\n' in out
        assert hub.current.index_url == URL('http://devpi:3141/foo/bar')

    def test_rewritten_url_path(self, capfd, cmd_devpi, mock_http_api):
        from devpi_common.url import URL
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                index="http://devpi/foo/bar/",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://devpi/foo/bar/?no_projects=", result=dict())
        hub = cmd_devpi("use", "http://devpi/foo/bar")
        (out, err) = capfd.readouterr()
        assert 'The server has rewritten the url to:' not in out
        assert hub.current.index_url == URL('http://devpi/foo/bar/')

    def test_rewritten_url_root(self, capfd, cmd_devpi, mock_http_api):
        from devpi_common.url import URL
        mock_http_api.set(
            "http://devpi/+api", result=dict(
                login="https://devpi/+login",
                authstatus=["noauth", ""]))
        hub = cmd_devpi("use", "http://devpi/")
        (out, err) = capfd.readouterr()
        assert 'The server has rewritten the url to: https://devpi/\n' in out
        assert hub.current.index_url == URL('')
        assert hub.current.root_url == URL('https://devpi/')

    @pytest.mark.skipif("config.option.fast")
    def test_use_list_doesnt_write(self, tmpdir, cmd_devpi, mock_http_api):
        import time
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                pypisubmit="/post",
                simpleindex="/index/",
                index="foo/bar",
                bases="root/pypi",
                login="/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set(
            "http://devpi/foo/bar?no_projects=",
            result=dict())
        mock_http_api.set(
            "http://devpi/", result=dict(
                foo=dict(username="foo", indexes=dict(
                    bar=dict(bases=("root/pypi",), volatile=False)))))
        cmd_devpi("use", "http://devpi/foo/bar")
        path = tmpdir.join("client", "current.json")
        mtime = path.mtime()
        time.sleep(1.5)
        cmd_devpi("use", "-l")
        assert mtime == path.mtime()

    def test_use_list_with_url_doesnt_write(self, tmpdir, cmd_devpi, mock_http_api):
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                pypisubmit="/post",
                simpleindex="/index/",
                index="foo/bar",
                bases="root/pypi",
                login="/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set(
            "http://devpi/", result=dict(
                foo=dict(username="foo", indexes=dict(
                    bar=dict(bases=("root/pypi",), volatile=False)))))
        path = tmpdir.join("client", "current.json")
        cmd_devpi("use", "-l", "http://devpi/foo/bar")
        assert not path.exists()

    def test_normalize_url(self):
        current = Current()
        current.reconfigure(dict(simpleindex="http://my.serv/index1"))
        url = current._normalize_url("index2")
        assert url == "http://my.serv/index2"

    def test_auth_multisite(self):
        current = Current()
        login1 = "http://site.com/+login"
        login2 = "http://site2.com/+login"
        current.login = login1
        current.set_auth("hello", "pass1")
        current.login = login2
        current.set_auth("hello", "pass2")
        assert current.get_auth(login1) == ("hello", "pass1")
        assert current.get_auth(login2) == ("hello", "pass2")
        current.login = login1
        current.del_auth()
        assert not current.get_auth(login1)
        assert current.get_auth(login2) == ("hello", "pass2")
        current.login = login2
        current.del_auth()
        assert not current.get_auth(login2)

    def test_invalid_reply(self, loghub, mock_http_api):
        current = Current()
        mock_http_api.add(
            'http://example.com/qwe/+api', reason="Not found", status=404)
        with pytest.raises(SystemExit):
            current.configure_fromurl(loghub, "http://example.com/qwe")
        loghub._getmatcher().fnmatch_lines("*404 Not found*")

    def test_invalid_url(self, loghub):
        current = Current()
        with pytest.raises(SystemExit):
            current.configure_fromurl(loghub, "http://heise.de:1802:31/qwe")
        loghub._getmatcher().fnmatch_lines("*invalid URL*")

    def test_auth_handling(self):
        current = Current()
        d = {
            "index": "http://l/some/index",
            "login": "http://l/login",
        }
        current.reconfigure(data=d)
        assert current.root_url
        current.set_auth("user", "password")
        assert current.get_auth() == ("user", "password")

        # ok response
        d["authstatus"] = ["ok", "user"]
        current._configure_from_server_api(d, current.root_url)
        assert current.get_auth() == ("user", "password")

        # invalidation response
        d["authstatus"] = ["nouser", "user"]
        current._configure_from_server_api(d, current.root_url)
        assert not current.get_auth()

    def test_rooturl_on_outside_url(self):
        current = Current()
        d = {
            "index": "http://l/subpath/some/index",
            "login": "http://l/subpath/login",
        }
        current.reconfigure(data=d)
        assert current.root_url == "http://l/subpath/"

    def test_use_with_no_rooturl(self, capfd, cmd_devpi, monkeypatch):
        from devpi import main
        monkeypatch.setattr(main.Hub, "http_api", None)
        cmd_devpi("use", "some/index", code=-2)
        out, err = capfd.readouterr()
        assert "invalid" in out

    @pytest.mark.parametrize("Exc", [
        requests.exceptions.ConnectionError,
        urllib3.exceptions.HTTPError,
    ])
    def test_use_with_nonexistent_domain(self, capfd, cmd_devpi, Exc,
                                         monkeypatch):
        from requests.sessions import Session

        def raise_(*args, **kwargs):
            raise Exc("qwe")

        monkeypatch.setattr(Session, "request", raise_)
        cmd_devpi("use", "http://qlwkejqlwke", code=-1)
        out, err = capfd.readouterr()
        assert "could not connect" in out

    def test_use_with_basic_auth(self, cmd_devpi, mock_http_api):
        mock_http_api.set(
            "http://devpi/foo/bar/+api", result=dict(
                pypisubmit="/post",
                simpleindex="/index/",
                index="foo/bar",
                bases="root/pypi",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://devpi/foo/bar?no_projects=",
            result=dict())
        mock_http_api.set(
            "http://devpi/foo/ham/+api", result=dict(
                pypisubmit="/post",
                simpleindex="/index/",
                index="foo/ham",
                bases="root/pypi",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "http://devpi/foo/ham?no_projects=",
            result=dict())
        mock_http_api.set(
            "http://devpi/", result=dict(
                foo=dict(username="foo", indexes=dict(
                    bar=dict(bases=("root/pypi",), volatile=False),
                    ham=dict(bases=("root/pypi",), volatile=False)))))
        # use with basic authentication
        hub = cmd_devpi("use", "http://user:password@devpi/foo/bar")
        # should work with and without explicit port if it's the default port
        assert hub.current.get_basic_auth(url="http://devpi/foo/bar") == ('user', 'password')
        assert hub.current.get_basic_auth(url="http://devpi:80/foo/bar") == ('user', 'password')
        assert len(mock_http_api.called) == 2
        assert mock_http_api.called[0][1].path == '/foo/bar/+api'
        assert mock_http_api.called[0][2]['basic_auth'] == ('user', 'password')
        assert mock_http_api.called[1][1].path == '/foo/bar'
        assert mock_http_api.called[1][2]['basic_auth'] is None  # http_api should do the lookup
        mock_http_api.called[:] = []  # clear call list
        # now we switch only the index, basic auth info should be kept
        hub = cmd_devpi("use", "/foo/ham")
        assert hub.current.get_basic_auth(url="http://devpi/foo/ham") == ('user', 'password')
        assert hub.current.get_basic_auth(url="http://devpi:80/foo/ham") == ('user', 'password')
        assert len(mock_http_api.called) == 2
        assert mock_http_api.called[0][1].path == '/foo/ham/+api'
        assert mock_http_api.called[0][2]['basic_auth'] == ('user', 'password')
        assert mock_http_api.called[1][1].path == '/foo/ham'
        assert mock_http_api.called[1][2]['basic_auth'] is None  # http_api should do the lookup
        mock_http_api.called[:] = []  # clear call list
        # just listing the index shouldn't change anything
        hub = cmd_devpi("use", "-l")
        assert hub.current.get_basic_auth(url="http://devpi/") == ('user', 'password')
        assert hub.current.get_basic_auth(url="http://devpi:80/") == ('user', 'password')
        assert len(mock_http_api.called) == 2
        assert mock_http_api.called[0][1].path == '/foo/ham/+api'
        assert mock_http_api.called[0][2]['basic_auth'] == ('user', 'password')
        assert mock_http_api.called[1][1] == 'http://devpi/'
        assert mock_http_api.called[1][2]['basic_auth'] is None  # http_api should do the lookup
        mock_http_api.called[:] = []  # clear call list
        # now without basic authentication the user and password should still be used
        hub = cmd_devpi("use", "http://devpi/foo/bar")
        assert hub.current.get_basic_auth(url="http://devpi/foo/bar") == ('user', 'password')
        assert hub.current.get_basic_auth(url="http://devpi/foo/ham") == ('user', 'password')
        assert hub.current.get_basic_auth(url="http://devpi/") == ('user', 'password')
        assert len(mock_http_api.called) == 2
        assert mock_http_api.called[0][1].path == '/foo/bar/+api'
        assert mock_http_api.called[0][2]['basic_auth'] == ('user', 'password')
        assert mock_http_api.called[1][1].path == '/foo/bar'
        assert mock_http_api.called[1][2]['basic_auth'] is None  # http_api should do the lookup

    def test_use_with_basic_auth_https(self, cmd_devpi, mock_http_api):
        mock_http_api.set(
            "https://devpi/foo/bar/+api", result=dict(
                pypisubmit="/post",
                simpleindex="/index/",
                index="foo/bar",
                bases="root/pypi",
                login="/+login",
                authstatus=["noauth", ""]))
        mock_http_api.set(
            "https://devpi/foo/bar?no_projects=",
            result=dict())
        # use with basic authentication
        hub = cmd_devpi("use", "https://user:password@devpi/foo/bar")
        # should work with and without explicit port if it's the default port
        assert hub.current.get_basic_auth(url="https://devpi/foo/bar") == ('user', 'password')
        assert hub.current.get_basic_auth(url="https://devpi:443/foo/bar") == ('user', 'password')
        assert len(mock_http_api.called) == 2
        assert mock_http_api.called[0][1].path == '/foo/bar/+api'
        assert mock_http_api.called[0][2]['basic_auth'] == ('user', 'password')
        assert mock_http_api.called[1][1].path == '/foo/bar'
        assert mock_http_api.called[1][2]['basic_auth'] is None  # http_api should do the lookup
        mock_http_api.called[:] = []  # clear call list
        # now without basic authentication the user and password should still be used
        hub = cmd_devpi("use", "https://devpi/foo/bar")
        assert hub.current.get_basic_auth(url="https://devpi/foo/bar") == ('user', 'password')
        assert len(mock_http_api.called) == 2
        assert mock_http_api.called[0][1].path == '/foo/bar/+api'
        assert mock_http_api.called[0][2]['basic_auth'] == ('user', 'password')
        assert mock_http_api.called[1][1].path == '/foo/bar'
        assert mock_http_api.called[1][2]['basic_auth'] is None  # http_api should do the lookup
        # previously, requesting a URL with query would yield no auth
        assert hub.current.get_basic_auth(url="https://devpi/foo/bar?no_projects=") == ('user', 'password')

    def test_change_index(self, cmd_devpi, mock_http_api):
        mock_http_api.set("http://world.com/+api", result=dict(
            index="/index",
            login="/+login",
            authstatus=["noauth", ""],
        ))
        mock_http_api.set(
            "http://world.com/index?no_projects=",
            result=dict())
        mock_http_api.set("http://world2.com/+api", result=dict(
            login="/+login",
            authstatus=["noauth", ""],
        ))

        hub = cmd_devpi("use", "http://world.com")
        assert hub.current.index == "http://world.com/index"
        assert hub.current.root_url == "http://world.com/"

        hub = cmd_devpi("use", "http://world2.com")
        assert not hub.current.index
        assert hub.current.root_url == "http://world2.com/"

    def test_switch_to_temporary(self, makehub, mock_http_api):
        hub = makehub(['use'])
        mock_http_api.set(
            "http://foo/+api", result=dict(
                pypisubmit="/post",
                simpleindex="/index/",
                index="root/some",
                bases="root/dev",
                login="/+login",
                authstatus=["noauth", ""]))
        current = Current()
        d = {
            "index": "http://l/some/index",
            "login": "http://l/login",
        }
        current.reconfigure(data=d)
        current.set_auth("user", "password")
        assert current.get_auth() == ("user", "password")
        temp = current.switch_to_temporary(hub, "http://foo")
        assert temp.get_auth("http://l") == ("user", "password")
        temp.set_auth("user1", "password1")
        assert temp.get_auth() == ("user1", "password1")
        # original is unaffected
        assert current.get_auth() == ("user", "password")
        assert temp._currentdict is not current._currentdict
        assert temp._authdict is not current._authdict

    def test_main(self, cmd_devpi, mock_http_api):
        mock_http_api.set("http://world/this/+api", result=dict(
            pypisubmit="/post",
            simpleindex="/index/",
            index="root/some",
            bases="root/dev",
            login="/+login",
            authstatus=["noauth", ""],
        ))
        mock_http_api.set(
            "http://world/root/some?no_projects=",
            result=dict())

        hub = cmd_devpi("use", "--venv", "-", "http://world/this")
        newapi = hub.current
        assert newapi.pypisubmit == "http://world/post"
        assert newapi.simpleindex == "http://world/index/"
        assert not newapi.venvdir

        # some url helpers
        current = hub.current
        assert current.get_index_url(slash=False) == "http://world/root/some"
        assert current.get_index_url() == "http://world/root/some/"
        assert current.get_project_url("pytest") == "http://world/root/some/pytest/"

        #hub = cmd_devpi("use", "--delete")
        #assert not hub.current.exists()

    def test_main_list(self, out_devpi, cmd_devpi, mock_http_api):
        mock_http_api.set("http://world/+api", result=dict(
            pypisubmit="",
            simpleindex="",
            index="",
            bases="",
            login="/+login",
            authstatus=["noauth", ""],
        ))
        mock_http_api.set(
            "http://world/?no_projects=",
            result=dict())

        cmd_devpi("use", "http://world/")
        mock_http_api.set(
            "http://world/", result=dict(
                user1=dict(indexes={
                    "dev": {"bases": ["x"], "volatile": False}}),
                user2=dict(indexes={
                    "foo": {"bases": ["x"], "volatile": False}})))
        out = out_devpi("use", "-l")
        out.stdout.fnmatch_lines("""
            user1/dev*x*False*
            user2/foo*x*False*
        """)
        mock_http_api.set(
            "http://world/user2", result=dict(
                indexes={
                    "foo": {"bases": ["x"], "volatile": False}}))
        out = out_devpi("use", "-l", "-u", "user2")
        out.stdout.fnmatch_lines("""
            user2/foo*x*False*
        """)

    def test_main_venvsetting(self, create_venv, out_devpi, cmd_devpi, tmpdir, monkeypatch):
        venvdir = create_venv()
        monkeypatch.chdir(tmpdir)
        hub = cmd_devpi("use", "--venv=%s" % venvdir)
        current = PersistentCurrent(hub.current.auth_path, hub.current.current_path)
        assert current.venvdir == str(venvdir)
        cmd_devpi("use", "--venv=%s" % venvdir)
        res = out_devpi("use")
        res.stdout.fnmatch_lines("*venv*%s" % venvdir)

        # clean venv setting
        cmd_devpi("use", "--venv=-")

        # test via env for virtualenvwrapper
        monkeypatch.setenv("WORKON_HOME", venvdir.dirpath().strpath)
        hub = cmd_devpi("use", "--venv=%s" % venvdir.basename)
        assert hub.current.venvdir == venvdir

        # clean venv setting
        cmd_devpi("use", "--venv=-")

        # test via env for activated venv
        monkeypatch.setenv("VIRTUAL_ENV", venvdir.strpath)
        hub = cmd_devpi("use")
        assert hub.current.venvdir is None, \
            "When --venv is not given, hub.current shouldn't be set"
        res = out_devpi("use")
        res.stdout.fnmatch_lines("*venv*%s" % venvdir)

    def test_new_venvsetting(self, capfd, cmd_devpi, tmpdir, monkeypatch):
        venvdir = tmpdir.join('.venv')
        assert not venvdir.exists()
        monkeypatch.chdir(tmpdir)
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--venv=%s" % venvdir)
        (out, err) = capfd.readouterr()
        assert "No virtualenv found at:" in out

    def test_venv_setcfg(self, mock_http_api, cmd_devpi, tmpdir, monkeypatch):
        from devpi.use import vbin
        monkeypatch.setenv("HOME", tmpdir.join('home').strpath)
        monkeypatch.setattr(PipCfg, "pip_conf_name", "pip.cfg")
        monkeypatch.setattr(DistutilsCfg, "default_location",
                            tmpdir.join("dist.cfg"))
        monkeypatch.setattr(BuildoutCfg, "default_location",
                            tmpdir.join("buildout.cfg"))
        mock_http_api.set("http://world/simple/+api", result=dict(
            pypisubmit="",
            simpleindex="/simple",
            index="/",
            bases="",
            login="/+login",
            authstatus=["noauth", ""],
        ))
        mock_http_api.set(
            "http://world/?no_projects=",
            result=dict())
        venvdir = tmpdir
        venvdir.ensure(vbin, dir=1)
        monkeypatch.chdir(tmpdir)
        index = "http://world/simple"
        cmd_devpi("use", "--venv=%s" % venvdir, "--set-cfg", index)

        assert not UvConf().path.exists()
        assert not PipCfg().path.exists()
        assert not DistutilsCfg.default_location.exists()
        assert not BuildoutCfg.default_location.exists()

        venv_pip_config = venvdir.join("pip.cfg")
        assert venv_pip_config.exists()
        content = venv_pip_config.read()
        assert len(re.findall(r"index_url\s*=\s*%s" % index, content)) == 1
        result = re.findall(
            r"\[search\].*index\s*=\s*%s" % index.replace('simple', ''), content, flags=re.DOTALL)
        assert len(result) == 1
        result = result[0].splitlines()
        assert len(result) == 2

        venv_uv_conf = venvdir.join("uv.toml")
        assert venv_uv_conf.exists()
        content = venv_uv_conf.read()
        assert len(re.findall(rf'index-url\s*=\s*"{index}"', content)) == 1

    def test_active_venv_setcfg(self, mock_http_api, cmd_devpi, tmpdir, monkeypatch):
        from devpi.use import vbin
        monkeypatch.setenv("HOME", tmpdir.join('home').strpath)
        monkeypatch.setattr(PipCfg, "pip_conf_name", "pip.cfg")
        monkeypatch.setattr(DistutilsCfg, "default_location",
                            tmpdir.join("dist.cfg"))
        monkeypatch.setattr(BuildoutCfg, "default_location",
                            tmpdir.join("buildout.cfg"))
        mock_http_api.set("http://world/simple/+api", result=dict(
            pypisubmit="",
            simpleindex="/simple",
            index="/",
            bases="",
            login="/+login",
            authstatus=["noauth", ""],
        ))
        mock_http_api.set(
            "http://world/?no_projects=",
            result=dict())
        venvdir = tmpdir
        venvdir.ensure(vbin, dir=1)
        monkeypatch.chdir(tmpdir)
        monkeypatch.setenv("VIRTUAL_ENV", venvdir.strpath)
        index = "http://world/simple"
        cmd_devpi("use", "--set-cfg", index)

        assert PipCfg(venv=venvdir).path.exists()
        assert UvConf(venv=venvdir).path.exists()

        assert not PipCfg().path.exists()
        assert not UvConf().path.exists()
        assert not DistutilsCfg.default_location.exists()
        assert not BuildoutCfg.default_location.exists()

    @pytest.mark.parametrize(['scheme', 'basic_auth'], [
        ('http', ''),
        ('https', ''),
        ('http', 'foo:bar@'),
        ('https', 'foo:bar@')])
    def test_main_setcfg(self, scheme, basic_auth, capfd, mock_http_api, cmd_devpi, tmpdir, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("HOME", tmpdir.join('home').strpath)
        monkeypatch.setattr(PipCfg, "pip_conf_name", "pip.cfg")
        monkeypatch.setattr(DistutilsCfg, "default_location",
                            tmpdir.join("dist.cfg"))
        monkeypatch.setattr(BuildoutCfg, "default_location",
                            tmpdir.join("buildout.cfg"))
        mock_http_api.set("%s://world/+api" % scheme, result=dict(
            pypisubmit="",
            simpleindex="/simple",
            index="/",
            bases="",
            login="/+login",
            authstatus=["noauth", ""],
        ))
        mock_http_api.set(
            "%s://world/?no_projects=" % scheme,
            result=dict())

        cmd_devpi("use", "--set-cfg", "%s://%sworld" % (scheme, basic_auth))
        # run twice to find any issues where lines are added more than once
        cmd_devpi("use", "--set-cfg", "%s://%sworld" % (scheme, basic_auth))
        if '@' in basic_auth:
            (out, err) = capfd.readouterr()
            assert basic_auth not in out
            assert ':****@' in out
        assert PipCfg().default_location.exists()
        content = PipCfg().default_location.read_text()
        assert len(
            re.findall(r"index_url\s*=\s*%s://%sworld/simple" % (
                scheme, basic_auth), content)) == 1
        result = re.findall(
            r"\[search\].*index\s*=\s*%s://%sworld/" % (
                scheme, basic_auth), content, flags=re.DOTALL)
        assert len(result) == 1
        result = result[0].splitlines()
        assert len(result) == 2
        assert UvConf().default_location.exists()
        content = UvConf().default_location.read_text()
        assert len(
            re.findall(
                rf'index-url\s*=\s*"{scheme}://{basic_auth}world/simple"',
                content)) == 1
        assert DistutilsCfg.default_location.exists()
        content = DistutilsCfg.default_location.read()
        assert len(
            re.findall(r"index_url\s*=\s*%s://%sworld/simple" % (
                scheme, basic_auth), content)) == 1
        assert BuildoutCfg.default_location.exists()
        content = BuildoutCfg.default_location.read()
        assert len(
            re.findall(r"index\s*=\s*%s://%sworld/simple" % (
                scheme, basic_auth), content)) == 1
        hub = cmd_devpi("use", "--always-set-cfg=yes")
        assert hub.current.always_setcfg
        hub = cmd_devpi("use", "--always-set-cfg=no")
        assert not hub.current.always_setcfg
        # Now set the trusted-host
        cmd_devpi(
            "use", "--set-cfg", "--pip-set-trusted=yes", "%s://%sworld" % (
                scheme, basic_auth))
        content = PipCfg().default_location.read_text()
        assert len(
            re.findall(r"trusted-host\s*=\s*world", content)) == 1
        hub = cmd_devpi("use", "--always-set-cfg=yes", "--pip-set-trusted=yes")
        assert hub.current.settrusted
        hub = cmd_devpi("use", "--always-set-cfg=no", "--pip-set-trusted=no")
        assert not hub.current.settrusted

    def test_environment_without_current(self, capfd, cmd_devpi, mock_http_api, monkeypatch):
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "no server:" in out
        monkeypatch.setenv("DEVPI_INDEX", "http://devpi/user/dev")
        mock_http_api.set(
            "http://devpi/user/dev/+api", result=dict(
                pypisubmit="http://devpi/user/dev/",
                simpleindex="http://devpi/user/dev/+simple/",
                index="http://devpi/user/dev",
                login="http://devpi/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://devpi/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: http://devpi/user/dev\n" in out
        assert "index: http://devpi/user/dev" in out
        assert "simpleindex: http://devpi/user/dev/+simple/" in out
        assert "pypisubmit: http://devpi/user/dev/" in out
        assert "login: http://devpi/+login" in out

    @pytest.mark.parametrize("devpi_index", ["user/dev", "/user/dev"])
    def test_environment_relative_without_current(
            self, capfd, cmd_devpi, devpi_index, mock_http_api, monkeypatch):
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "no server:" in out
        monkeypatch.setenv("DEVPI_INDEX", devpi_index)
        mock_http_api.set(
            "http://devpi/user/dev/+api", result=dict(
                pypisubmit="http://devpi/user/dev/",
                simpleindex="http://devpi/user/dev/+simple/",
                index="http://devpi/user/dev",
                login="http://devpi/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://devpi/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "No server set and DEVPI_INDEX from environment is not a full valid URL: %s\n" % devpi_index in out

    def test_environment_with_current(self, capfd, cmd_devpi, mock_http_api, monkeypatch):
        mock_http_api.set(
            "http://world/user/dev/+api", result=dict(
                pypisubmit="http://world/user/dev/",
                simpleindex="http://world/user/dev/+simple/",
                index="http://world/user/dev",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "http://world/user/dev")
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "index: http://world/user/dev" in out
        assert "simpleindex: http://world/user/dev/+simple/" in out
        assert "pypisubmit: http://world/user/dev/" in out
        assert "login: http://world/+login" in out
        monkeypatch.setenv("DEVPI_INDEX", "http://devpi/user/dev")
        mock_http_api.set(
            "http://devpi/user/dev/+api", result=dict(
                pypisubmit="http://devpi/user/dev/",
                simpleindex="http://devpi/user/dev/+simple/",
                index="http://devpi/user/dev",
                login="http://devpi/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://devpi/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: http://devpi/user/dev\n" in out
        assert "index: http://devpi/user/dev" in out
        assert "simpleindex: http://devpi/user/dev/+simple/" in out
        assert "pypisubmit: http://devpi/user/dev/" in out
        assert "login: http://devpi/+login" in out

    @pytest.mark.parametrize("devpi_index", ["user/dev", "/user/dev"])
    def test_environment_relative_with_current(
            self, capfd, cmd_devpi, devpi_index, mock_http_api, monkeypatch):
        mock_http_api.set(
            "http://world/user/dev/+api", result=dict(
                pypisubmit="http://world/user/dev/",
                simpleindex="http://world/user/dev/+simple/",
                index="http://world/user/dev",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "http://world/user/dev")
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "index: http://world/user/dev" in out
        assert "simpleindex: http://world/user/dev/+simple/" in out
        assert "pypisubmit: http://world/user/dev/" in out
        assert "login: http://world/+login" in out
        monkeypatch.setenv("DEVPI_INDEX", devpi_index)
        mock_http_api.set(
            "http://world/user/dev/+api", result=dict(
                pypisubmit="http://world/user/dev/",
                simpleindex="http://world/user/dev/+simple/",
                index="http://world/user/dev",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: %s\n" % devpi_index in out
        assert "index: http://world/user/dev" in out
        assert "simpleindex: http://world/user/dev/+simple/" in out
        assert "pypisubmit: http://world/user/dev/" in out
        assert "login: http://world/+login" in out

    def test_environment_with_root_current(self, capfd, cmd_devpi, mock_http_api, monkeypatch):
        mock_http_api.set(
            "http://world/+api", result=dict(
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        cmd_devpi("use", "http://world/")
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "using server: http://world/ (not logged in)" in out
        assert "no current index" in out
        monkeypatch.setenv("DEVPI_INDEX", "http://devpi/user/dev")
        mock_http_api.set(
            "http://devpi/user/dev/+api", result=dict(
                pypisubmit="http://devpi/user/dev/",
                simpleindex="http://devpi/user/dev/+simple/",
                index="http://devpi/user/dev",
                login="http://devpi/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://devpi/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: http://devpi/user/dev\n" in out
        assert "index: http://devpi/user/dev" in out
        assert "simpleindex: http://devpi/user/dev/+simple/" in out
        assert "pypisubmit: http://devpi/user/dev/" in out
        assert "login: http://devpi/+login" in out

    @pytest.mark.parametrize("devpi_index", ["user/dev", "/user/dev"])
    def test_environment_relative_with_root_current(
            self, capfd, cmd_devpi, devpi_index, mock_http_api, monkeypatch):
        mock_http_api.set(
            "http://world/+api", result=dict(
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        cmd_devpi("use", "http://world/")
        (out, err) = capfd.readouterr()
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "using server: http://world/ (not logged in)" in out
        assert "no current index" in out
        monkeypatch.setenv("DEVPI_INDEX", devpi_index)
        mock_http_api.set(
            "http://world/user/dev/+api", result=dict(
                pypisubmit="http://world/user/dev/",
                simpleindex="http://world/user/dev/+simple/",
                index="http://world/user/dev",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: %s\n" % devpi_index in out
        assert "index: http://world/user/dev" in out
        assert "simpleindex: http://world/user/dev/+simple/" in out
        assert "pypisubmit: http://world/user/dev/" in out
        assert "login: http://world/+login" in out

    def test_environment_with_url_from_commandline(
            self, capfd, cmd_devpi, mock_http_api, monkeypatch):
        monkeypatch.setenv("DEVPI_INDEX", "http://devpi/user/dev")
        mock_http_api.set(
            "http://world/user/foo/+api", result=dict(
                pypisubmit="http://world/user/foo/",
                simpleindex="http://world/user/foo/+simple/",
                index="http://world/user/foo",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/foo?no_projects=", result=dict())
        cmd_devpi("use", "http://world/user/foo")
        (out, err) = capfd.readouterr()
        assert "Using index URL from command line" in out
        mock_http_api.set(
            "http://devpi/user/dev/+api", result=dict(
                pypisubmit="http://devpi/user/dev/",
                simpleindex="http://devpi/user/dev/+simple/",
                index="http://devpi/user/dev",
                login="http://devpi/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://devpi/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: http://devpi/user/dev\n" in out
        assert "index: http://devpi/user/dev" in out
        assert "simpleindex: http://devpi/user/dev/+simple/" in out
        assert "pypisubmit: http://devpi/user/dev/" in out
        assert "login: http://devpi/+login" in out
        cmd_devpi("use", "http://world/user/foo", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using index URL from command line" in out
        assert "index: http://world/user/foo" in out
        assert "simpleindex: http://world/user/foo/+simple/" in out
        assert "pypisubmit: http://world/user/foo/" in out
        assert "login: http://world/+login" in out

    @pytest.mark.parametrize("devpi_index", ["user/dev", "/user/dev"])
    def test_environment_relative_with_url_from_commandline(
            self, capfd, cmd_devpi, devpi_index, mock_http_api, monkeypatch):
        monkeypatch.setenv("DEVPI_INDEX", devpi_index)
        mock_http_api.set(
            "http://world/user/foo/+api", result=dict(
                pypisubmit="http://world/user/foo/",
                simpleindex="http://world/user/foo/+simple/",
                index="http://world/user/foo",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/foo?no_projects=", result=dict())
        cmd_devpi("use", "http://world/user/foo")
        (out, err) = capfd.readouterr()
        assert "Using index URL from command line" in out
        mock_http_api.set(
            "http://world/user/dev/+api", result=dict(
                pypisubmit="http://world/user/dev/",
                simpleindex="http://world/user/dev/+simple/",
                index="http://world/user/dev",
                login="http://world/+login",
                authstatus=["noauth", "", []]))
        mock_http_api.set("http://world/user/dev?no_projects=", result=dict())
        cmd_devpi("use", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using DEVPI_INDEX from environment: %s\n" % devpi_index in out
        assert "index: http://world/user/dev" in out
        assert "simpleindex: http://world/user/dev/+simple/" in out
        assert "pypisubmit: http://world/user/dev/" in out
        assert "login: http://world/+login" in out
        cmd_devpi("use", "http://world/user/foo", "--urls")
        (out, err) = capfd.readouterr()
        assert "Using index URL from command line" in out
        assert "index: http://world/user/foo" in out
        assert "simpleindex: http://world/user/foo/+simple/" in out
        assert "pypisubmit: http://world/user/foo/" in out
        assert "login: http://world/+login" in out


def test_getparse_keyvalues_invalid():
    with pytest.raises(ValueError):
        get_keyvalues(["hello123"])


@pytest.mark.parametrize(("input", "expected"), [
    (["hello=123", "world=42"], dict(hello="123", world="42")),
    (["hello=123=1"], dict(hello="123=1")),
    (["hello=1", "hello=2"], dict(hello="2")),
    (["hello+=1"], {"hello+": "1"}),
    (["hello-=1"], {"hello-": "1"})])
def test_getparse_keyvalues_kvdict(input, expected):
    result = get_keyvalues(input)
    assert result.kvdict == expected


def test_user_no_index(loghub):
    out_index_list(loghub, {"user": {"username": "user"}})


def test_pipcfg_default_location(tmpdir, monkeypatch):
    path = PipCfg().path
    monkeypatch.setenv('PIP_CONFIG_FILE', tmpdir.join("cfg").strpath)
    assert path != PipCfg().path


def test_uvconf_default_location(tmpdir, monkeypatch):
    path = UvConf().path
    monkeypatch.setenv('UV_CONFIG_FILE', tmpdir.join("cfg").strpath)
    assert path != UvConf().path


class TestCfgParsing:
    @pytest.fixture(scope="class", params=[DistutilsCfg, PipCfg, BuildoutCfg, UvConf])
    def cfgclass(self, request):
        return request.param

    def test_empty(self, cfgclass, tmpdir):
        p = tmpdir.join("cfg")
        assert not cfgclass(p).exists()
        assert cfgclass(p).indexserver is None
        assert cfgclass(p).screen_name == str(p)
        assert cfgclass.default_location

    def test_read(self, cfgclass, tmpdir):
        p = tmpdir.join("cfg")
        cfg = cfgclass(p)
        cfg.write_default("http://some.com/something")
        with pytest.raises(ValueError):
            cfg.write_default("http://some.com/something")
        cfg = cfgclass(p)
        assert cfg.exists()
        assert cfg.indexserver == "http://some.com/something"
        assert cfg.screen_name == str(p)

    def test_read_config_but_no_index(self, tmpdir, cfgclass):
        path = tmpdir.join("cfg")
        if cfgclass.section_name:
            path.write(cfgclass.section_name + "\n")
        cfg = cfgclass(path)
        cfg.write_indexserver("http://qwe")
        cfg = cfgclass(path)
        assert cfg.indexserver == "http://qwe"

    def test_read_config_but_no_section(self, tmpdir, cfgclass):
        path = tmpdir.join("cfg")
        path.write("")
        cfg = cfgclass(path)
        cfg.write_indexserver("http://qwe")
        cfg = cfgclass(path)
        assert cfg.indexserver == "http://qwe"

    def test_write_fresh(self, cfgclass, tmpdir):
        p = tmpdir.join("cfg")
        cfg = cfgclass(p)
        cfg.write_indexserver("http://hello.com")
        assert cfg.indexserver == "http://hello.com"

    def test_rewrite(self, cfgclass, tmpdir):
        p = tmpdir.join("cfg")
        cfgclass(p).write_default("http://some.com/something")
        cfg = cfgclass(p)
        cfg.write_indexserver("http://hello.com")
        assert cfg.indexserver == "http://hello.com"
        assert cfgclass(cfg.backup_path).indexserver == \
               "http://some.com/something"
        cfg.write_indexserver("http://hello.com")
        assert cfgclass(cfg.backup_path).indexserver == \
               "http://some.com/something"
