from __future__ import annotations

from devpi_server.markers import Deleted
from difflib import SequenceMatcher
from functools import singledispatch
from itertools import chain
from itertools import groupby
from operator import attrgetter
from operator import itemgetter
from pyramid.httpexceptions import HTTPForbidden
from pyramid.view import view_config
from typing import TYPE_CHECKING
import json
import sqlalchemy as sa


if TYPE_CHECKING:
    from devpi_server.main import XOM
    from pyramid.request import Request
    from typing import Any


@singledispatch
def pformat(val: Any) -> str:
    msg = f"don't know how to handle type {type(val)!r}"
    raise TypeError(msg)


@pformat.register
def _(val: Deleted) -> str:
    return str(val)


@pformat.register
def _(val: object) -> str:
    return json.dumps(val, indent=4, default=sorted, sort_keys=True)


@view_config(
    route_name="keyfs",
    request_method="GET",
    renderer="templates/keyfs.pt")
def keyfs_view(request: Request) -> dict:
    xom: XOM = request.registry["xom"]
    if not xom.config.args.debug_keyfs:
        raise HTTPForbidden("+keyfs views disabled")
    storage = xom.keyfs._storage
    query = request.params.get('query')
    serials: list[int] = []
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
def keyfs_changelog_view(request: Request) -> dict:
    xom: XOM = request.registry["xom"]
    if not xom.config.args.debug_keyfs:
        raise HTTPForbidden("+keyfs views disabled")
    html_key_types_map = {
        k.key_name: str(k.key_type.__name__).replace(" ", "\xa0")
        for k in xom.keyfs.schema
    }
    storage = xom.keyfs._storage
    serial = int(request.matchdict["serial"])
    query = request.params.get('query')
    changes = []
    with storage.get_connection() as conn:
        last_changelog_serial = conn.last_changelog_serial
        rel_renames = sorted(conn.iter_rel_renames(serial))
        raw_changes = list(conn.iter_changes_at(serial))
        keys = {keydata.key for keydata in raw_changes}
        raw_parent_keys = {
            parent_key
            for key in keys
            if (parent_key := key.parent_key) is not None and parent_key not in keys
        }
        parent_keys = sorted(
            (
                dict(
                    fragment=f"{int(keydata.key.ulid)}",
                    name=keydata.key.key_name,
                    type=html_key_types_map[keydata.key.key_name],
                    relpath=keydata.key.relpath,
                    serial=keydata.serial,
                )
                for keydata in conn.iter_keys_at_serial(
                    raw_parent_keys, serial, fill_cache=True, with_deleted=False
                )
            ),
            key=itemgetter("name", "relpath"),
        )
        back_serial_keys_keydata = {
            serial: {
                keydata.key: keydata
                for keydata in conn.iter_keys_at_serial(
                    (kd.key for kd in keysdata),
                    serial,
                    fill_cache=True,
                    with_deleted=False,
                )
            }
            for serial, keysdata in groupby(
                sorted(raw_changes, key=attrgetter("back_serial")),
                attrgetter("back_serial"),
            )
            if serial >= 0
        }
        for keydata in raw_changes:
            prev_formatted = ""
            if keydata.back_serial >= 0:
                prev_formatted = pformat(
                    back_serial_keys_keydata[keydata.back_serial][
                        keydata.key
                    ].mutable_value
                )
            formatted = pformat(keydata.mutable_value)
            diffed = diff(prev_formatted, formatted)
            latest_serial = conn.last_key_serial(keydata.key)
            parent_key = keydata.key.parent_key
            changes.append(
                dict(
                    fragment=f"{int(keydata.key.ulid)}",
                    name=keydata.key.key_name,
                    type=html_key_types_map[keydata.key.key_name],
                    relpath=keydata.key.relpath,
                    previous_serial=keydata.back_serial,
                    latest_serial=latest_serial,
                    diffed=diffed,
                    parent_key=dict(
                        fragment=f"{int(parent_key.ulid)}",
                        name=parent_key.key_name,
                        type=html_key_types_map[parent_key.key_name],
                        relpath=parent_key.relpath,
                    )
                    if parent_key is not None
                    else None,
                )
            )
    changes.sort(key=itemgetter("name", "relpath"))
    return dict(
        changes=changes,
        rel_renames=rel_renames,
        parent_keys=parent_keys,
        pformat=pformat,
        last_changelog_serial=last_changelog_serial,
        serial=int(serial),
        query=query,
    )
