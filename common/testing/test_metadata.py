from devpi_common.metadata import BasenameMeta
from devpi_common.metadata import Version
from devpi_common.metadata import encode_int
from devpi_common.metadata import get_latest_version
from devpi_common.metadata import get_pyversion_filetype
from devpi_common.metadata import get_sorted_versions
from devpi_common.metadata import parse_requirement
from devpi_common.metadata import splitbasename
from devpi_common.metadata import splitext_archive
from devpi_common.metadata import version_sort_string
from hypothesis import given
from hypothesis import strategies as st
import pytest
import string


@pytest.mark.parametrize(("releasename", "expected"), [
    ("pytest-2.3.4.zip", ("pytest", "2.3.4", ".zip")),
    ("pytest-2.3.4-py27.egg", ("pytest", "2.3.4", "-py27.egg")),
    ("dddttt-0.1.dev38-py2.7.egg", ("dddttt", "0.1.dev38", "-py2.7.egg")),
    ("devpi-0.9.5.dev1-cp26-none-linux_x86_64.whl",
        ("devpi", "0.9.5.dev1", "-cp26-none-linux_x86_64.whl")),
    ("wheel-0.21.0-py2.py3-none-any.whl", ("wheel", "0.21.0", "-py2.py3-none-any.whl")),
    ("green-0.4.0-py2.5-win32.egg", ("green", "0.4.0", "-py2.5-win32.egg")),
    ("Candela-0.2.1.macosx-10.4-x86_64.exe",
        ("Candela", "0.2.1", ".macosx-10.4-x86_64.exe")),
    ("Cambiatuscromos-0.1.1alpha.linux-x86_64.exe",
        ("Cambiatuscromos", "0.1.1alpha", ".linux-x86_64.exe")),
    ("Aesthete-0.4.2.win32.exe", ("Aesthete", "0.4.2", ".win32.exe")),
    ("DTL-1.0.5.win-amd64.exe", ("DTL", "1.0.5", ".win-amd64.exe")),
    ("Cheetah-2.2.2-1.x86_64.rpm", ("Cheetah", "2.2.2", "-1.x86_64.rpm")),
    ("Cheetah-2.2.2-1.src.rpm", ("Cheetah", "2.2.2", "-1.src.rpm")),
    ("Cheetah-2.2.2-1.x85.rpm", ("Cheetah", "2.2.2", "-1.x85.rpm")),
    ("Cheetah-2.2.2.dev1-1.x85.rpm", ("Cheetah", "2.2.2.dev1", "-1.x85.rpm")),
    ("Cheetah-2.2.2.dev1-1.noarch.rpm", ("Cheetah", "2.2.2.dev1", "-1.noarch.rpm")),
    ("deferargs.tar.gz", ("deferargs", "", ".tar.gz")),
    ("hello-1.0.doc.zip", ("hello", "1.0", ".doc.zip")),
    ("Twisted-12.0.0.win32-py2.7.msi",
        ("Twisted", "12.0.0", ".win32-py2.7.msi")),
    ("django_ipware-0.0.8-py3-none-any.whl", ("django_ipware", "0.0.8", "-py3-none-any.whl")),
    ("my-binary-package-name-1-4-3-yip-0.9.tar.gz", ("my-binary-package-name-1-4-3-yip", "0.9", ".tar.gz")),
    ("my-binary-package-name-1-4-3-yip-0.9+deadbeef.tar.gz", ("my-binary-package-name-1-4-3-yip", "0.9+deadbeef", ".tar.gz")),
    ("cffi-1.6.0-pp251-pypy_41-macosx_10_11_x86_64.whl", ("cffi", "1.6.0", "-pp251-pypy_41-macosx_10_11_x86_64.whl")),
    ("argon2_cffi-18.2.0.dev0.0-pp2510-pypy_41-macosx_10_13_x86_64.whl", ("argon2_cffi", "18.2.0.dev0.0", "-pp2510-pypy_41-macosx_10_13_x86_64.whl")),
])
def test_splitbasename(releasename, expected):
    result = splitbasename(releasename)
    assert result == expected


@pytest.mark.parametrize(("releasename", "expected"), [
    ("x-2.3.zip", ("source", "sdist")),
    ("x-2.3-0.4.0.win32-py3.1.exe", ("3.1", "bdist_wininst")),
    ("x-2.3-py27.egg", ("2.7", "bdist_egg")),
    ("wheel-0.21.0-py2.py3-none-any.whl", ("2.7", "bdist_wheel")),
    ("devpi-0.9.5.dev1-cp26-none-linux_x86_64.whl", ("2.6", "bdist_wheel")),
    ("greenlet-0.4.0-py3.3-win-amd64.egg", ("3.3", "bdist_egg")),
    ("greenlet-0.4.0.linux-x86_64.tar.gz", ("any", "bdist_dumb")),
    ("cffi-1.6.0-pp251-pypy_41-macosx_10_11_x86_64.whl", ("2.5.1", "bdist_wheel")),
    ("cryptography-1.4-pp253-pypy_41-linux_x86_64.whl", ("2.5.3", "bdist_wheel")),
    ("cryptography-39.0.0-pp38-pypy38_pp73-macosx_10_12_x86_64.whl", ("3.8", "bdist_wheel")),
    ("argon2_cffi-18.2.0.dev0.0-pp2510-pypy_41-macosx_10_13_x86_64.whl", ("2.5.1.0", "bdist_wheel")),
    ("h5py-3.6.0-cp310-cp310-win_amd64.whl", ("3.10", "bdist_wheel")),
    ("h5py-3.6.0-cp319-cp319-win_amd64.whl", ("3.19", "bdist_wheel")),
    ("h5py-3.6.0-cp399-cp399-win_amd64.whl", ("3.99", "bdist_wheel")),
])
def test_get_pyversion_filetype(releasename, expected):
    result = get_pyversion_filetype(releasename)
    assert result == expected


@pytest.mark.parametrize(("releasename", "expected"), [
    ("pytest-2.3.4.zip", ("pytest-2.3.4", ".zip")),
    ("green-0.4.0-py2.5-win32.egg", ("green-0.4.0-py2.5-win32", ".egg")),
    ("green-1.0.tar.gz", ("green-1.0", ".tar.gz")),
    ("green-1.0.doc.zip", ("green-1.0", ".doc.zip")),
])
def test_splitext_archive(releasename, expected):
    assert splitext_archive(releasename) == expected


@pytest.mark.parametrize(("expected", "versions"), [
    (None, []),
    ("1.0", ["1.0"]),
    ("1.0", ["1.0", "0.9"]),
    ("1.0.1.dev0", ["1.0", "1.0.1.dev0"]),
    ("2.0-alpha1", ["1.0", "2.0a0", "2.0.a0", "2.0-alpha1"]),
    ("2.0-beta1", ["1.0", "2.0b0", "2.0.b0", "2.0-beta1"]),
    ("2.0-rc1", ["1.0", "2.0rc0", "2.0.rc0", "2.0-rc1"]),
    ("2.0-pre1", ["1.0", "2.0pre0", "2.0.pre0", "2.0-pre1"]),
])
def test_get_latest_version(expected, versions):
    assert get_latest_version(versions) == expected


@pytest.mark.parametrize(("expected", "versions"), [
    (None, ["1.0rc1"]),
    ("1.0", ["1.0"]),
    ("1.0", ["1.0", "0.9"]),
    ("1.0", ["1.0", "1.0.1.dev0"]),
    ("1.0", ["1.0", "2.0a0", "2.0.a0", "2.0-alpha1"]),
    ("1.0", ["1.0", "2.0b0", "2.0.b0", "2.0-beta1"]),
    ("1.0", ["1.0", "2.0rc0", "2.0.rc0", "2.0-rc1"]),
    ("1.0", ["1.0", "2.0pre0", "2.0.pre0", "2.0-pre1"]),
])
def test_get_latest_stable_version(expected, versions):
    assert get_latest_version(versions, stable=True) == expected


@pytest.mark.parametrize(("versions", "expected"), [
    (
        ["2022.7.1", "2022.7", "2005i", "2004d"],
        ["2004d", "2005i", "2022.7", "2022.7.1"]
    ),
    (
        ["1.0alpha1", "1.0beta5prerelease2", "3.1.4-ec6"],
        ["1.0beta5prerelease2", "3.1.4-ec6", "1.0alpha1"],
    ),
    (
        ["1.0", "2.0", "2005i", "2004d"],
        ["2004d", "2005i", "1.0", "2.0"]
    ),
])
def test_get_sorted_versions_legacy(versions, expected):
    assert get_sorted_versions(versions, reverse=False) == expected


@pytest.mark.parametrize(("versions", "expected"), [
    (
        ["2022.7.1", "2022.7", "2005i", "2004d"],
        ["2004d", "2005i", "2022.7", "2022.7.1"],
    ),
    (
        ["1.0alpha1", "1.0beta5prerelease2", "3.1.4-ec6"],
        ["1.0beta5prerelease2", "3.1.4-ec6"],
    ),
    (
        ["1.0", "2.0", "2005i", "2004d"],
        ["2004d", "2005i", "1.0", "2.0"],
    ),
])
def test_get_sorted_versions_legacy_stable(versions, expected):
    assert get_sorted_versions(versions, stable=True, reverse=False) == expected


@given(st.lists(st.integers(min_value=-(10**100), max_value=10**100)))
def test_encode_int(integers):
    si = sorted((i, encode_int(i)) for i in integers)
    se = sorted((encode_int(i), i) for i in integers)
    assert [i for i, _e in si] == [i for _e, i in se]
    assert [e for _i, e in si] == [e for e, _i in se]
    assert not any("/" in e for e, _i in se)


@st.composite
def pep440versions(draw):
    version = "v" if draw(st.booleans()) else ""
    if draw(st.booleans()):
        # epoch
        version = f"{version}{draw(st.integers(min_value=0))}!"
    version = f"{version}{draw(st.integers(min_value=0))}"
    for _n in range(draw(st.integers(min_value=0, max_value=10))):
        version = f"{version}.{draw(st.integers(min_value=0))}"
    seps = ["", ".", "-", "_"]
    if draw(st.booleans()):
        # pre-release
        kind = draw(
            st.sampled_from(["a", "alpha", "b", "beta", "c", "pre", "preview", "rc"])
        )
        pre_sep = draw(st.sampled_from(seps))
        sep = draw(st.sampled_from(seps))
        _num = (
            draw(st.integers(min_value=0))
            if sep
            else draw(st.just("") | st.integers(min_value=0))
        )
        version = f"{version}{pre_sep}{kind}{sep}{_num}"
    if draw(st.booleans()):
        # post-release
        kind = draw(st.sampled_from(["post", "r", "rev"]))
        post_sep = draw(st.sampled_from(seps))
        sep = draw(st.sampled_from(seps))
        _num = (
            draw(st.integers(min_value=0))
            if sep
            else draw(st.just("") | st.integers(min_value=0))
        )
        version = f"{version}{post_sep}{kind}{sep}{_num}"
    if draw(st.booleans()):
        # post-release
        dev_sep = draw(st.sampled_from(seps))
        sep = draw(st.sampled_from(seps))
        _num = (
            draw(st.integers(min_value=0))
            if sep
            else draw(st.just("") | st.integers(min_value=0))
        )
        version = f"{version}{dev_sep}dev{sep}{_num}"
    if draw(st.booleans()):
        # local version
        version = f"{version}+{''.join(draw(st.lists(st.integers().map(str) | st.text(string.ascii_letters + string.digits) | st.sampled_from(['.', '-', '_']), max_size=5)))}"
    return version


_versions = [
    "3.1.4-ec6",
    "1.0beta5prerelease2",
    "1.0alpha1",
    "2.0.0.post1",
    "2.0.0",
    "2.0.post1",
    "2.0",
    "1!2.0",
    "2!2.0",
    "2.0-pre1",
    "2.0pre0",
    "2.0dev1",
    "2.0.dev0",
    "2.0rc2",
    "1.0",
    "1.0.dev0",
    "1.0dev0",
    "1.0.0.dev2",
    "1.0.0dev2",
    "1.0+foo.10",
    "1.0+foo.1",
    "1.0+foo.3",
    "1.0+foo.4.ham",
    "1.0+foo.4.bar",
    "1.0.4.3.1.2.4.5.6.2.3.4.5.1.2.3.5.6.3.2.4.2.3.4.5.3.4.1.3.4.23.5.5.32.42.43.2.34.44",
    "0.1.40404040404040404040404040404040404040404040404040404040404040404040404040404040",
    "99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999111111111111111111111111111111111100000000.6.0",
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593",
    "2004d",
    "2005i",
    "2022.7",
    "2022.7.1",
]


@given(st.lists(pep440versions() | st.sampled_from(_versions), min_size=2, max_size=20))
def test_version_sort_string(versions):
    from operator import itemgetter

    expected = get_sorted_versions(versions, reverse=False)
    result = sorted(
        (
            (
                version_sort_string(x.cmpval),
                x.cmpval,
                getattr(x.cmpval, "_key", None),
                x.string,
            )
            for x in map(Version, versions)
        ),
        key=itemgetter(0),
    )
    assert not any("/" in x[0] for x in result)
    result_strings = [x[-1] for x in result]
    assert expected == result_strings


def test_version():
    ver1 = Version("1.0")
    ver2 = Version("1.1")
    assert max([ver1, ver2]) == ver2


class TestBasenameMeta:
    def test_doczip(self):
        meta1 = BasenameMeta("x-1.0.doc.zip")
        assert meta1.name == "x"
        assert meta1.version == "1.0"
        assert meta1.ext == ".doc.zip"

    def test_two_comparison(self):
        meta1 = BasenameMeta("x-1.0.tar.gz")
        meta2 = BasenameMeta("x-1.1.tar.gz")
        assert meta1 != meta2
        assert meta1 < meta2
        assert meta1.name == "x"
        assert meta1.version == "1.0"
        assert meta1.ext == ".tar.gz"
        assert meta1.obj == "x-1.0.tar.gz"

    def test_normalize_equal(self):
        meta1 = BasenameMeta("x-1.0.tar.gz")
        meta2 = BasenameMeta("X-1.0.tar.gz")
        assert meta1 == meta2
        meta3 = BasenameMeta("X-1.0.zip")
        assert meta3 != meta1
        assert meta3 > meta1

    def test_basename_attribute(self):
        class B:
            basename = "x-1.0.tar.gz"
        meta1 = BasenameMeta(B)
        meta2 = BasenameMeta("x-1.0.tar.gz")
        assert meta1 == meta2

    def test_noversion_sameproject(self):
        meta1 = BasenameMeta("py-1.0.zip", sameproject=True)
        meta2 = BasenameMeta("master", sameproject=True)
        meta3 = BasenameMeta("zer", sameproject=True)
        assert meta1 > meta2
        assert meta2 < meta3
        assert meta1 > meta3

    def test_notsameproject(self):
        meta1 = BasenameMeta("py-1.0.zip")
        meta2 = BasenameMeta("abc-1.0.zip")
        meta3 = BasenameMeta("zbc-1.0.zip")
        assert meta1 > meta2
        assert meta1 < meta3


def test_parse_requirement():
    req = parse_requirement("hello>=1.0")
    assert req.project_name == "hello"
    assert "1.0" in req
    assert "1.1.dev0" in req
    assert "1.0.dev0" not in req
    assert "0.9" not in req


def test_parse_requirement_without_version():
    req = parse_requirement("hello")
    assert req.project_name == "hello"
    assert "1.0" in req
    assert "1.0.dev0" in req
