from textwrap import dedent
import pytest


pytest_plugins = ["pytest_devpi_server", "test_devpi_server.plugin"]


def pytest_addoption(parser):
    parser.addoption("--fast", help="skip functional/slow tests", default=False,
                     action="store_true")
    parser.addoption(
        "--html-validation",
        help="run HTML validation using vnu",
        default=False,
        action="store_true",
    )


@pytest.fixture
def maketestapp(maketestapp, request):
    import subprocess
    import tempfile

    def do_request(self, *args, **kw):
        resp = self._original_do_request(*args, **kw)
        if resp.content_type in {
            "application/json",
            "text/css",
            "text/plain",
            "text/xml",
        }:
            return resp
        if resp.content_type is None and not resp.body:
            return resp
        if (ct := resp.content_type) in {"text/html"}:
            if not resp.body or resp.status_code in {302}:
                return resp
            with tempfile.NamedTemporaryFile(suffix=f".{ct.split('/')[-1]}") as f:
                f.write(resp.body)
                f.flush()
                try:
                    subprocess.check_call(  # noqa: S603 - testing
                        ["vnu", "--Werror", "--format", "text", f.name],  # noqa: S607 - testing
                    )
                except subprocess.CalledProcessError:
                    print(  # noqa: T201 - testing
                        "\n".join(
                            f"{i}:{l}"
                            for i, l in enumerate(resp.text.splitlines(), start=1)
                        )
                    )
                    raise
            return resp
        raise ValueError(resp.content_type)

    def _maketestapp(xom):
        testapp = maketestapp(xom)
        if request.config.getoption("--html-validation"):
            testapp._original_do_request = testapp.do_request
            testapp.do_request = do_request.__get__(testapp)
        return testapp

    return _maketestapp


@pytest.fixture
def xom(request, makexom):
    import devpi_web.main
    xom = makexom(plugins=[(devpi_web.main, None)])
    return xom


@pytest.fixture
def theme_path(request, tmp_path):
    marker = request.node.get_closest_marker("theme_files")
    files = {} if marker is None else marker.args[0]
    path = tmp_path / "theme"
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("static").mkdir(parents=True, exist_ok=True)
    path.joinpath("templates").mkdir(parents=True, exist_ok=True)
    for filepath, content in files.items():
        path.joinpath(*filepath).write_text(dedent(content))
    return path


@pytest.fixture(params=[None, "tox38"])
def tox_result_data(request):
    from test_devpi_server.example import tox_result_data
    import copy
    tox_result_data = copy.deepcopy(tox_result_data)
    if request.param == "tox38":
        retcode = int(tox_result_data['testenvs']['py27']['test'][0]['retcode'])
        tox_result_data['testenvs']['py27']['test'][0]['retcode'] = retcode
    return tox_result_data


@pytest.fixture(params=[True, False])
def keep_docs_packed(monkeypatch, request):
    value = request.param

    def func(config):
        return value

    monkeypatch.setattr("devpi_web.doczip.keep_docs_packed", func)
    return value


@pytest.fixture
def bs_text():
    def bs_text(resultset):
        return ' '.join(''.join(x.text for x in resultset).split())

    return bs_text
