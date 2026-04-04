from .functional import TestIndexPushThings as BaseTestIndexPushThings
from .functional import TestIndexThings as BaseTestIndexThings
from .functional import TestProjectThings as BaseTestProjectThings
from .functional import TestRemoteIndexThings as BaseTestRemoteIndexThings
from .functional import TestUserThings as BaseTestUserThings
import pytest


@pytest.fixture
def mapp(makemapp, monkeypatch, nginx_host_port, secretfile):
    app = makemapp(options=[
        '--primary-url', 'http://%s:%s' % nginx_host_port,
        '--secretfile', secretfile])
    monkeypatch.setattr(app.xom.replica_thread.connection, "REPLICA_REQUEST_TIMEOUT", 5)
    app.xom.thread_pool.start_one(app.xom.replica_thread)
    try:
        yield app
    finally:
        app.xom.thread_pool.kill()


@pytest.fixture
def remote_index_info(server_version):
    from devpi_common.metadata import parse_version

    if server_version < parse_version("7.0.0.dev2"):

        class MirrorInfo:
            merge_all_option = "mirror_whitelist"
            merge_all_value = "*"
            refresh_option = "mirror_cache_expiry"
            type = "mirror"
            url_fmt_option = "mirror_web_url_fmt"
            url_option = "mirror_url"

        return MirrorInfo()

    class RemoteInfo:
        merge_all_option = "project_inheritance_rules"
        merge_all_value = ("allow all",)
        refresh_option = "remote_refresh_delay"
        type = "remote"
        url_fmt_option = "remote_web_url_fmt"
        url_option = "remote_url"

    return RemoteInfo()


@pytest.mark.slow
class TestProjectThings(BaseTestProjectThings):
    pass


@pytest.mark.slow
class TestUserThings(BaseTestUserThings):
    pass


@pytest.mark.slow
class TestIndexThings(BaseTestIndexThings):
    pass


@pytest.mark.slow
class TestIndexPushThings(BaseTestIndexPushThings):
    pass


@pytest.mark.slow
class TestRemoteIndexThings(BaseTestRemoteIndexThings):
    pass
