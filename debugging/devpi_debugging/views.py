from devpi_server.markers import deleted
from difflib import SequenceMatcher
from functools import partial
from hashlib import sha256
from itertools import chain
from operator import itemgetter
from pyramid.httpexceptions import HTTPForbidden
from pyramid.view import view_config
import json
import sqlalchemy as sa


@view_config(
    route_name="keyfs",
    request_method="GET",
    renderer="templates/keyfs.pt")
def keyfs_view(request):
    xom = request.registry['xom']
    if not xom.config.args.debug_keyfs:
        raise HTTPForbidden("+keyfs views disabled")
    storage = xom.keyfs._storage
    query = request.params.get('query')
    serials = []
    if query:
        with storage.get_connection() as conn:
            if conn._sqlaconn.dialect.name == "sqlite":
                q = (
                    sa.select(storage.ulid_changelog_table.c.serial)
                    .where(
                        sa.func.hex(storage.ulid_changelog_table.c.value).contains(
                            sa.func.hex(query.encode())
                        )
                    )
                    .order_by(storage.ulid_changelog_table.c.serial)
                    .distinct()
                )
            elif conn._sqlaconn.dialect.name == "postgresql":
                q = (
                    sa.select(storage.ulid_changelog_table.c.serial)
                    .where(
                        sa.func.position(
                            sa.literal(query.encode()).op("IN")(
                                storage.ulid_changelog_table.c.value
                            )
                        )
                        > 0
                    )
                    .order_by(storage.ulid_changelog_table.c.serial)
                    .distinct()
                )
            else:
                q = (
                    sa.select(storage.ulid_changelog_table.c.serial)
                    .where(
                        storage.ulid_changelog_table.c.value.contains(query.encode())
                    )
                    .order_by(storage.ulid_changelog_table.c.serial)
                    .distinct()
                )
            serials.extend(x[0] for x in conn._sqlaconn.execute(q))
    else:
        with storage.get_connection() as conn:
            start = range(min(5, conn.last_changelog_serial + 1))
            end = range(max(0, conn.last_changelog_serial - 4), conn.last_changelog_serial + 1)
            serials.extend(sorted(set(chain(start, end))))
    return dict(
        query=query,
        serials=serials)


def diff(prev, current):
    prev_lines = prev.splitlines()
    lines = current.splitlines()
    cruncher = SequenceMatcher(None, prev_lines, lines)
    result = []
    for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
        if tag == 'equal':
            for i in range(alo, ahi):
                result.append(('equal', prev_lines[i]))
        elif tag == 'delete':
            for i in range(alo, ahi):
                result.append(('remove', prev_lines[i]))
        elif tag == 'insert':
            for i in range(blo, bhi):
                result.append(('insert', lines[i]))
        elif tag == 'replace':
            for i in range(alo, ahi):
                result.append(('remove', prev_lines[i]))
            for i in range(blo, bhi):
                result.append(('insert', lines[i]))
    return result


@view_config(
    route_name="keyfs_changelog",
    request_method="GET",
    renderer="templates/keyfs_changelog.pt")
def keyfs_changelog_view(request):
    xom = request.registry['xom']
    if not xom.config.args.debug_keyfs:
        raise HTTPForbidden("+keyfs views disabled")
    pformat = partial(json.dumps, indent=4, default=sorted, sort_keys=True)
    storage = xom.keyfs._storage
    serial = int(request.matchdict["serial"])
    query = request.params.get('query')
    changes = []
    with storage.get_connection() as conn:
        last_changelog_serial = conn.last_changelog_serial
        rel_renames = sorted(conn.iter_rel_renames(serial))
        for keydata in conn.iter_changes_at(serial):
            key = xom.keyfs.get_key_instance(keydata.keyname, keydata.relpath)
            prev_formatted = ''
            if keydata.back_serial >= 0:
                prev_formatted = pformat(
                    None
                    if (
                        v := conn.get_key_at_serial(
                            key, keydata.back_serial
                        ).mutable_value
                    )
                    is deleted
                    else v
                )
            formatted = pformat(None if (v := keydata.value) is deleted else v)
            diffed = diff(prev_formatted, formatted)
            latest_serial = conn.last_key_serial(key)
            changes.append(
                dict(
                    fragment=sha256(
                        f"{keydata.keyname}-{keydata.relpath}".encode()
                    ).hexdigest(),
                    name=keydata.keyname,
                    type=key.type,
                    relpath=keydata.relpath,
                    previous_serial=keydata.back_serial,
                    latest_serial=latest_serial,
                    diffed=diffed,
                )
            )
    changes.sort(key=itemgetter("name", "relpath"))
    return dict(
        changes=changes,
        rel_renames=rel_renames,
        pformat=pformat,
        last_changelog_serial=last_changelog_serial,
        serial=int(serial),
        query=query)
