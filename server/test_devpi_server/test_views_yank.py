from __future__ import annotations

from typing import TYPE_CHECKING
import pytest


if TYPE_CHECKING:
    from .plugin import Mapp
    from .plugin import MyTestApp


def getlinks(text):
    from bs4 import BeautifulSoup

    return BeautifulSoup(text, "html.parser").find_all("a")


@pytest.mark.notransaction
def test_yank_release(mapp: Mapp, testapp: MyTestApp) -> None:
    api = mapp.create_and_use()
    assert "yank" in api.features
    mapp.upload_file_pypi("pkg-2.6.tgz", b"123", "pkg", "2.6")
    (result,) = mapp.getjson(f"{api.index}/pkg")["result"]["2.6"]["+links"]
    assert "yanked" not in result
    assert "brownbag" not in mapp.get_simple("pkg").text
    assert "brownbag" not in mapp.get_simple("pkg", use_json=True).text
    (link,) = mapp.getreleaseslist("pkg")
    r = testapp.post(link, expect_errors=True)
    assert r.status_code == 400
    assert "Bad request: could not decode json" in r.text
    r = testapp.post_json(link, expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: could not decode json"
    r = testapp.post_json(link, [], expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: json is not a dict"
    r = testapp.post_json(link, {"type": "yank"}, expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: yank 'reason' must be a string or False"
    r = testapp.post_json(link, {"type": "yank", "reason": "brownbag"})
    assert r.json["message"] == "release yanked"
    assert {
        a.text: a.attrs.get("data-yanked")
        for a in getlinks(mapp.get_simple("pkg").text)
    } == {"pkg-2.6.tgz": "brownbag"}
    assert {
        x["filename"]: x.get("yanked")
        for x in mapp.get_simple("pkg", use_json=True).json["files"]
    } == {"pkg-2.6.tgz": "brownbag"}
    (result,) = mapp.getjson(f"{api.index}/pkg")["result"]["2.6"]["+links"]
    assert result["yanked"] == "brownbag"
    r = testapp.post_json(link, {"type": "yank", "reason": False})
    assert r.json["message"] == "release unyanked"
    assert "brownbag" not in mapp.get_simple("pkg").text
    assert "brownbag" not in mapp.get_simple("pkg", use_json=True).text
    (result,) = mapp.getjson(f"{api.index}/pkg")["result"]["2.6"]["+links"]
    assert "yanked" not in result
    r = testapp.post_json(link, {}, expect_errors=True)
    assert r.status_code == 200
    assert r.json["type"] == "toxresultpath"


@pytest.mark.notransaction
def test_yank_version(mapp: Mapp, testapp: MyTestApp) -> None:
    api = mapp.create_and_use()
    assert "yank" in api.features
    mapp.upload_file_pypi("pkg-2.6.tgz", b"123", "pkg", "2.6")
    mapp.upload_file_pypi("pkg-2.7.tgz", b"456", "pkg", "2.7")
    result = {
        k: v["+links"] for k, v in mapp.getjson(f"{api.index}/pkg")["result"].items()
    }
    assert set(result.keys()) == {"2.6", "2.7"}
    assert len(result["2.6"]) == 1
    assert "yanked" not in result["2.6"][0]
    assert len(result["2.7"]) == 1
    assert "yanked" not in result["2.7"][0]
    assert "brownbag" not in mapp.get_simple("pkg").text
    assert "brownbag" not in mapp.get_simple("pkg", use_json=True).text
    version_path = f"{api.index}/pkg/2.7"
    r = testapp.post(version_path, expect_errors=True)
    assert r.status_code == 406
    assert r.json["message"] == "Not Acceptable"
    r = testapp.post_json(version_path, expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: could not decode json"
    r = testapp.post_json(version_path, [], expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: json is not a dict"
    r = testapp.post_json(version_path, {}, expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: unrecognized 'type' in json: None"
    r = testapp.post_json(version_path, {"type": "yank"}, expect_errors=True)
    assert r.status_code == 400
    assert r.json["message"] == "Bad request: yank 'reason' must be a string or False"
    r = testapp.post_json(version_path, {"type": "yank", "reason": "brownbag"})
    assert r.json["message"] == "version yanked"
    assert {
        a.text: a.attrs.get("data-yanked")
        for a in getlinks(mapp.get_simple("pkg").text)
    } == {"pkg-2.6.tgz": None, "pkg-2.7.tgz": "brownbag"}
    assert {
        x["filename"]: x.get("yanked")
        for x in mapp.get_simple("pkg", use_json=True).json["files"]
    } == {"pkg-2.6.tgz": None, "pkg-2.7.tgz": "brownbag"}
    result = {
        k: v["+links"] for k, v in mapp.getjson(f"{api.index}/pkg")["result"].items()
    }
    assert set(result.keys()) == {"2.6", "2.7"}
    assert len(result["2.6"]) == 1
    assert "yanked" not in result["2.6"][0]
    assert len(result["2.7"]) == 1
    assert "yanked" in result["2.7"][0]
    r = testapp.post_json(version_path, {"type": "yank", "reason": False})
    assert r.json["message"] == "version unyanked"
    assert "brownbag" not in mapp.get_simple("pkg").text
    assert "brownbag" not in mapp.get_simple("pkg", use_json=True).text
    result = {
        k: v["+links"] for k, v in mapp.getjson(f"{api.index}/pkg")["result"].items()
    }
    assert set(result.keys()) == {"2.6", "2.7"}
    assert len(result["2.6"]) == 1
    assert "yanked" not in result["2.6"][0]
    assert len(result["2.7"]) == 1
    assert "yanked" not in result["2.7"][0]
