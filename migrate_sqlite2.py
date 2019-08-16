from devpi_server.fileutil import loads
from devpi_server.main import get_pluginmanager
from devpi_server.main import xom_from_argv
import time


def migrate(conn_old, conn_new):
    old_serial = conn_old.last_changelog_serial
    new_serial = conn_new.last_changelog_serial
    old_changes = {}
    last_time = time.time()
    for serial in range(new_serial + 1, old_serial + 1):
        if time.time() - last_time > 1:
            print("\r%10i/%i" % (serial, old_serial), end="")
            last_time = time.time()
        raw = conn_old.get_raw_changelog_entry(serial)
        (old_changes, rel_renames) = loads(raw)
        conn_old._changelog_cache.put(serial, old_changes)
        new_changes = {}

        for relpath, (keyname, back_serial, value) in old_changes.items():
            new_changes[relpath] = (keyname, back_serial, value)

        for relpath, (keyname, back_serial, value) in new_changes.items():
            conn_new.db_write_typedkey(relpath, keyname, serial)
        conn_new.write_changelog_entry(serial, (new_changes, rel_renames))
        conn_new._sqlconn.commit()
        conn_new._changelog_cache.put(serial, new_changes)
        conn_new._sqlconn.execute("begin immediate")


def main():
    pm = get_pluginmanager()
    xom_old = xom_from_argv(pm, [
        'devpi-server',
        '--serverdir', 'tmp/devpi.net'])
    xom_new = xom_from_argv(pm, [
        'devpi-server',
        '--serverdir', 'tmp/devpi.net',
        '--storage', 'sqlite2'])
    with xom_old.keyfs._storage.get_connection(write=False) as conn_old:
        with xom_new.keyfs._storage.get_connection(write=True) as conn_new:
            migrate(conn_old, conn_new)


if __name__ == '__main__':
    main()
