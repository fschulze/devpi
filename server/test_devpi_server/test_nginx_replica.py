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
            refresh_option = "mirror_cache_expiry"
            type = "mirror"

        return MirrorInfo()

    class RemoteInfo:
        refresh_option = "remote_refresh_delay"
        type = "remote"

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
