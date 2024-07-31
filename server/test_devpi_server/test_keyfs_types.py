from devpi_server import keyfs_types


def test_ulid_type():
    import time
    time_ns = time.time_ns()
    ulid = keyfs_types.ULID()
    assert issubclass(type(ulid), keyfs_types.ULID)
    assert isinstance(ulid, keyfs_types.ULID)
    assert issubclass(type(ulid), int)
    assert isinstance(ulid, int)
    assert ulid > 0
    assert ulid.bit_length() < 64
    # check that we start with the timestamp (rounded to 10 seconds)
    assert ((ulid >> 16) // 10) == (time_ns // 10_000_000)
    ulid2 = keyfs_types.ULID(2)
    assert issubclass(type(ulid2), keyfs_types.ULID)
    assert isinstance(ulid2, keyfs_types.ULID)
    assert issubclass(type(ulid2), int)
    assert isinstance(ulid2, int)
    assert ulid2 == 2
    assert ulid2.bit_length() < 64
    assert keyfs_types.ULID() != keyfs_types.ULID()
