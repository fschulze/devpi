from devpi_server.fileutil import loads
from devpi_server.main import get_pluginmanager
from devpi_server.main import xom_from_argv
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
import time


def join_requires(links, requires_python):
    # build list of (key, href, require_python) tuples
    result = []
    for link, require_python in zip_longest(links, requires_python, fillvalue=None):
        key, href = link
        result.append((key, href, require_python))
    return result


# 281857, 281860, 281872
def migrate(conn_old, conn_new):
    old_serial = conn_old.last_changelog_serial
    new_serial = conn_new.last_changelog_serial
    old_changes = {}
    elinks_in_projversion = set()
    indexes_in_user = dict()
    serial_in_simplelinks = set()
    last_time = time.time()
    for serial in range(new_serial + 1, old_serial + 1):
        if time.time() - last_time > 1:
            print("\r%10i/%i" % (serial, old_serial), end="")
            last_time = time.time()
        raw = conn_old.get_raw_changelog_entry(serial)
        (old_changes, rel_renames) = loads(raw)
        conn_old._changelog_cache.put(serial, old_changes)
        new_changes = {}

        def write_serial(relpath, back_serial, value):
            if value is None:
                if relpath not in serial_in_simplelinks:
                    return
                import pdb; pdb.set_trace()
                serial_in_simplelinks.remove(relpath)
            else:
                if 'serial' not in value:
                    if relpath not in serial_in_simplelinks:
                        return
                    value = None
                    serial_in_simplelinks.remove(relpath)
                else:
                    value = value.pop('serial')
                    serial_in_simplelinks.add(relpath)
            new_relpath = relpath.replace('/.simple', '/.serial')
            new_changes[new_relpath] = ('PROJSERIAL', back_serial, value)

        def write_index(relpath, back_serial, key, value):
            new_relpath = relpath.replace('/.config', '/%s/.config' % key)
            new_changes[new_relpath] = ('INDEX', back_serial, value)

        def write_indexes(relpath, back_serial, value):
            new_relpath = relpath.replace('/.config', '/.indexes')
            if value is None:
                if relpath not in indexes_in_user:
                    return
                new_changes[new_relpath] = ('INDEXLIST', back_serial, None)
                for index in indexes_in_user[relpath]:
                    write_index(relpath, back_serial, None)
                indexes_in_user.pop(relpath)
                return
            else:
                if 'indexes' not in value:
                    if relpath not in indexes_in_user:
                        return
                    import pdb; pdb.set_trace()
                    new_changes[new_relpath] = ('INDEXLIST', back_serial, None)
                    for index in indexes_in_user[relpath]:
                        write_index(relpath, back_serial, None)
                    indexes_in_user.pop(relpath)
                    return
                else:
                    indexes = value.pop('indexes')
                    indexes_in_user.setdefault(relpath, set())
            old_indexes = indexes_in_user[relpath]
            new_indexes = set(indexes.keys()) - old_indexes
            deleted_indexes = set(indexes.keys()) - new_indexes
            updated_indexes = set(indexes.keys()) - old_indexes - new_indexes
            if updated_indexes:
                import pdb; pdb.set_trace()
            new_changes[new_relpath] = ('INDEXLIST', back_serial, set(indexes.keys()))
            for key in updated_indexes:
                write_index(relpath, back_serial, key, indexes[key])
            for key in new_indexes:
                write_index(relpath, -1, key, indexes[key])
            for key in deleted_indexes:
                import pdb; pdb.set_trace()
                write_index(relpath, back_serial, key, None)

        def write_elinks(relpath, back_serial, value):
            new_relpath = relpath.replace('/.config', '/.files')
            if value is None:
                if relpath not in elinks_in_projversion:
                    return
                elinks_in_projversion.remove(relpath)
                new_changes[new_relpath] = ('VERSIONFILES', back_serial, None)
                return
            else:
                if '+elinks' not in value:
                    if relpath not in elinks_in_projversion:
                        return
                    import pdb; pdb.set_trace()
                    pass
                else:
                    links = value.pop('+elinks')
                    if relpath not in elinks_in_projversion:
                        back_serial = -1
                    elinks_in_projversion.add(relpath)
            new_value = {}
            for item in links:
                key = item.pop('entrypath')
                if key in new_value:
                    print("\nOverwriting duplicate version file:\nold -> %r\nnew -> %r" % (new_value[key], item))
                new_value[key] = item
            new_changes[new_relpath] = ('VERSIONFILES', back_serial, new_value)

        for relpath, (keyname, back_serial, value) in old_changes.items():
            if keyname == 'PROJNAMES':
                keyname = 'PROJECTS'
                if value is not None:
                    value = {x: None for x in value}
            elif keyname == 'PROJSIMPLELINKS':
                write_serial(relpath, back_serial, value)
                if value is not None:
                    assert set(['links', 'requires_python']).union(value.keys()) == set(['links', 'requires_python']), value.keys()
                    value = set(join_requires(
                        value.get('links', ()),
                        value.get('requires_python', ())))
            elif keyname == 'USER':
                write_indexes(relpath, back_serial, value)
            elif keyname == 'PROJVERSION':
                write_elinks(relpath, back_serial, value)
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
            # import cProfile
            # cp = cProfile.Profile()
            # cp.enable()
            migrate(conn_old, conn_new)
            # cp.disable()
            # cp.dump_stats('migrate.pstat')


if __name__ == '__main__':
    main()
