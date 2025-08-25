from devpi_server import keyfs_types
import pytest
import sys


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
    # check that we start with the timestamp (rounded to 16 seconds)
    assert ((ulid >> keyfs_types.ULID_SHIFT) // 16) == (
        time_ns // (16 * keyfs_types.ULID_NS_DIVISOR)
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
    import datetime

    # 64-bit signed integer maximum
    max_int = 0x7FFFFFFFFFFFFFFF
    ulid_max = keyfs_types.ULID(max_int)
    assert ulid_max.bit_length() < 64
    utc_max = datetime.datetime.fromtimestamp(
        int(ulid_max) >> keyfs_types.ULID_SHIFT, tz=datetime.timezone.utc
    )
    # we should be good until the year 3000
    assert utc_max.year > 3000
