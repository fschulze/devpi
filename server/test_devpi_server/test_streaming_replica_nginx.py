from . import test_streaming
import pytest
import sys


pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="issues with process management on Windows"),
    pytest.mark.slow]


@pytest.fixture
def host_port(request, storage_info):
    if "storage_with_filesystem" not in storage_info.get("_test_markers", []):
        pytest.skip("The storage doesn't have marker 'storage_with_filesystem'.")
    return request.getfixturevalue("nginx_replica_host_port")


@pytest.fixture
def files_path(replica_server_path):
    return replica_server_path / '+files'


server_url_session = test_streaming.server_url_session
content_digest = test_streaming.content_digest


for attr in dir(test_streaming):
    if attr.startswith(('test_', 'Test')):
        globals()[attr] = getattr(test_streaming, attr)
