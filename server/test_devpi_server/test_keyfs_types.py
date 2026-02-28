from __future__ import annotations

from devpi_server import keyfs_types
from devpi_server.keyfs_types import RelPath
import pytest
import sys


@pytest.mark.parametrize(("relpath", "match_str"), [
    ("../foo", "must not contain"),
    ("./foo", "must not contain"),
    ("/absolute", "must be relative"),
    ("foo/.", "must not contain"),
    ("foo/..", "must not contain"),
])
def test_invalid_relpath(match_str: str, relpath: str) -> None:
    with pytest.raises(ValueError, match=match_str):
        RelPath(relpath)


def test_ulid_type():
    import time

    time_ns = time.time_ns()
    ulid = keyfs_types.ULID.new()
    assert issubclass(type(ulid), keyfs_types.ULID)
    assert isinstance(ulid, keyfs_types.ULID)
    assert issubclass(type(ulid), int)
    assert isinstance(ulid, int)
    assert ulid > 0
    assert ulid.bit_length() < 64
    # check that we start with the timestamp
    assert (ulid >> keyfs_types.ULID_RAND_BITS) == (
        time_ns // keyfs_types.ULID_NS_DIVISOR
    )
    ulid2 = keyfs_types.ULID(2)
    assert issubclass(type(ulid2), keyfs_types.ULID)
    assert isinstance(ulid2, keyfs_types.ULID)
    assert issubclass(type(ulid2), int)
    assert isinstance(ulid2, int)
    assert ulid2 == 2
    assert ulid2.bit_length() < 64
    # there should be no duplicates
    assert keyfs_types.ULID.new() != keyfs_types.ULID.new()
    ulids = [int(keyfs_types.ULID.new()) for x in range(100)]
    assert sorted(ulids) == sorted(set(ulids))


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Maximum year error on Windows"
)
def test_ulid_type_max_year():
    import datetime as dt

    # 64-bit signed integer maximum
    max_int = 0x7FFFFFFFFFFFFFFF
    ulid_max = keyfs_types.ULID(max_int)
    assert ulid_max.bit_length() < 64
    utc_max = dt.datetime.fromtimestamp(
        int(ulid_max) >> keyfs_types.ULID_RAND_BITS, tz=dt.UTC
    )
    # we should be good until the year 3000
    assert utc_max.year > 3000
