import json
import pytest
import re


def test_yank_unyank_file(initproj, devpi, out_devpi):
    initproj("hello-1.0")
    hub = devpi("upload", "--no-isolation")
    if "yank" not in hub.current.features:
        pytest.skip("Server doesn't support 'yank'")
    initproj("hello-1.1")
    devpi("upload", "--no-isolation")
    out = out_devpi("getjson", "hello")
    result = json.loads("\n".join(out.outlines))
    version_links = {v: vd["+links"] for v, vd in result["result"].items()}
    assert all("yanked" not in l for links in version_links.values() for l in links)
    (url,) = (l["href"] for l in version_links["1.0"] if l["href"].endswith(".whl"))
    assert "hello-1.0" in url
    out = out_devpi("yank", url)
    out.stdout.fnmatch_lines_random("*release yanked")
    out = out_devpi("unyank", url)
    out.stdout.fnmatch_lines_random("*release unyanked")


@pytest.mark.parametrize("other_index", ["root/pypi", None])
def test_yank_unyank_version(initproj, devpi, out_devpi, other_index):
    initproj("hello-1.0")
    hub = devpi("upload", "--no-isolation")
    if "yank" not in hub.current.features:
        pytest.skip("Server doesn't support 'yank'")
    initproj("hello-1.1")
    devpi("upload", "--no-isolation")
    # remember username
    out = out_devpi("use")
    user = re.search(r"\(logged in as (.+?)\)", out.stdout.str()).group(1)
    if other_index is not None:
        # go to other index
        devpi("use", other_index)
    out = out_devpi("getjson", "hello")
    result = json.loads("\n".join(out.outlines))
    version_links = {v: vd["+links"] for v, vd in result["result"].items()}
    assert all("yanked" not in l for links in version_links.values() for l in links)
    if other_index is None:
        out = out_devpi("yank", "hello==1.0")
    else:
        out = out_devpi("yank", "--index", f"{user}/dev", "hello==1.0")
    out.stdout.fnmatch_lines_random("*version yanked")
    if other_index is None:
        out = out_devpi("unyank", "hello==1.0")
    else:
        out = out_devpi("unyank", "--index", f"{user}/dev", "hello==1.0")
    out.stdout.fnmatch_lines_random("*version unyanked")
