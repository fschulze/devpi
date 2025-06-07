from devpi_common.metadata import parse_version
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
        if (ct := resp.content_type) == "text/html":
            if not resp.body or resp.status_code == 302:
                return resp
            with tempfile.NamedTemporaryFile(suffix=f".{ct.split('/')[-1]}") as f:
                filterpatterns = [
                    "Article lacks heading.*",
                    "The heading.+follows the heading.+skipping.+heading level.*",
                    "This document has heading elements but none of them has a computed heading level of.*",
                ]
                f.write(resp.body)
                f.flush()
                try:
                    subprocess.check_call(  # noqa: S603 - testing
                        ["vnu", "--Werror", "--filterpattern", "|".join(filterpatterns), "--format", "text", f.name],  # noqa: S607 - testing
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


@pytest.fixture
def pypistage(pypistage, remote_index_info):
    if remote_index_info.remote_search_option:
        if isinstance(pypistage.ixconfig, dict):
            pypistage.ixconfig[remote_index_info.remote_search_option] = True
        else:
            pypistage.ixconfig._data[remote_index_info.remote_search_option] = True
    return pypistage


@pytest.fixture
def remote_index_info(server_version):
    if server_version < parse_version("7.0.0.dev2"):

        class MirrorInfo:
            merge_all_option = "mirror_whitelist"
            merge_all_value = "*"
            refresh_option = "mirror_cache_expiry"
            remote_search_option = None
            type = "mirror"
            url_fmt_option = "mirror_web_url_fmt"
            url_option = "mirror_url"

        return MirrorInfo()

    class RemoteInfo:
        merge_all_option = "project_inheritance_rules"
        merge_all_value = ("allow all",)
        refresh_option = "remote_refresh_delay"
        remote_search_option = "remote_include_in_search"
        type = "remote"
        url_fmt_option = "remote_web_url_fmt"
        url_option = "remote_url"

    return RemoteInfo()
