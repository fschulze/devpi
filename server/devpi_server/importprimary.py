from __future__ import annotations

from .filestore import Digests
from .keyfs_types import PatternedKey
from .keyfs_types import Record
from .keyfs_types import ULIDKey
from .log import thread_push_log
from .log import threadlog
from .main import CommandRunner
from .main import DATABASE_VERSION
from .main import init_default_indexes
from .main import set_state_version
from .main import xom_from_config
from .markers import absent
from .markers import deleted
from .model import join_links_data
from .readonly import get_mutable_deepcopy
from .replica import ReplicaConnection
from collections import Counter
from collections import defaultdict
from pathlib import Path
from repoze.lru import LRUCache
from typing import TYPE_CHECKING
import sys


if TYPE_CHECKING:
    from .keyfs_types import KeyData
    from .keyfs_types import KeyFSTypesRO
    from .keyfs_types import LocatedKey
    from .markers import Absent
    from .markers import Deleted
    from collections.abs import Iterable
    from collections.abs import Sequence
    from typing import Any


class Nameslist:
    def __init__(self, keyname, location, keys):
        self.existing = set()
        self.keyname = keyname
        self.location = location
        self.value = absent
        for key in keys.values():
            if key.key_name != keyname:
                continue
            if key.location != location:
                continue
            if key.exists():
                self.existing.add(key.name)

    def check(self, records):
        if self.value is absent:
            return
        to_add = set()
        to_keep = set(self.existing)
        to_remove = set()
        keyname = self.keyname
        location = self.location
        records = [
            r
            for r in records
            if (key := r.key).key_name == keyname and key.location == location
        ]
        for record in records:
            name = record.key.name
            if name not in to_keep:
                to_add.add(name)
            elif record.value is deleted:
                to_remove.add(name)
                to_keep.remove(name)
        assert self.value == to_add | to_keep, (
            self.keyname,
            self.value,
            self.existing,
            to_add,
            to_keep,
            to_remove,
        )
        assert to_add == set(self.value).difference(self.existing), (
            self.keyname,
            self.value,
            self.existing,
            to_add,
            to_keep,
            to_remove,
        )
        assert to_keep == set(self.existing).intersection(self.value), (
            self.keyname,
            self.value,
            self.existing,
            to_add,
            to_keep,
            to_remove,
        )
        assert to_remove == set(self.existing).difference(self.value), (
            self.keyname,
            self.value,
            self.existing,
            to_add,
            to_keep,
            to_remove,
        )

    @property
    def key(self):
        return (self.keyname, self.location)

    def set_new(self, value):
        assert self.value is absent
        self.value = value


class Nameslists:
    def __init__(self):
        self.lists = {}

    def __getitem__(self, key):
        return self.lists[key]

    def add(self, nameslist):
        assert nameslist.key not in self.lists
        self.lists[nameslist.key] = nameslist

    def check(self, records):
        for l in self.lists.values():
            l.check(records)


class ReplicaImport:
    def __init__(self, xom):
        self.xom = xom
        xom.keyfs._import_subscriber = None
        self.connection = ReplicaConnection(
            xom,
            self.continue_on_missing_remote_primary_uuid,
            self.set_primary_uuid,
            self.update_primary_serial,
        )
        self.connection.log = thread_push_log("[IPRIM]")
        self.primary_serial = None
        self.recent_existing = LRUCache(1000)

    def continue_on_missing_remote_primary_uuid(self, _headers):
        return True

    def fetch(self):
        while True:
            serial = self.xom.keyfs.get_current_serial() + 1
            (result, exc) = self.connection.fetch(
                self.connection.handler_multi,
                self.connection.get_changelog_url(serial),
                self.import_changes,
            )
            if exc is not None and not isinstance(exc, self.connection.http.Errors):
                raise exc

    def import_changes(self, serial: int, changes: Any) -> None:
        with self.xom.keyfs.read_transaction():
            records = self.make_records(serial, changes[0])
        self.xom.keyfs.import_records(serial, records)

    def iter_migrated_changes(
        self, changes: Any
    ) -> Iterable[tuple[str, str, int, Any]]:  # noqa: PLR0912
        sort_order = dict(
            USERLIST=1,
            USER=2,
            MIRRORNAMESINIT=3,
            PROJNAMES=4,
            PROJSIMPLELINKS=5,
            PROJVERSIONS=6,
            PROJVERSION=7,
            STAGEFILE=8,
        )
        renames = dict(
            PROJVERSION="VERSION",
            PYPIFILE_NOMD5="FILE_NOHASH",
            STAGEFILE="FILE",
        )
        relpath_suffix = dict(
            MIRRORNAMESINIT="/.mirrornameschange",
            PROJNAMES="/.projects",
            PROJSIMPLELINKS="/.simple",
            PROJVERSION="/.config",
            PROJVERSIONS="/.versions",
            USER="/.config",
            USERLIST="/.config",
        )
        unchanged_values = {
            "MIRRORNAMESINIT",
            "PROJNAMES",
            "PYPIFILE_NOMD5",
            "USERLIST",
        }
        changes = sorted(changes.items(), key=lambda x: sort_order[x[1][0]])
        for relpath, (keyname, back_serial, val) in changes:
            new_relpath = relpath.removesuffix(relpath_suffix.get(keyname, ""))
            new_keyname = renames.get(keyname, keyname)
            if keyname in (unchanged_values):
                yield (new_relpath, new_keyname, back_serial, val)
            elif keyname == "PROJSIMPLELINKS":
                yield (new_relpath, "PROJECTNAME", back_serial, Path(new_relpath).name)
                yield (new_relpath, new_keyname, back_serial, val)
                value = {} if val is None else val
                links = join_links_data(
                    value.get("links", ()),
                    value.get("requires_python", ()),
                    value.get("yanked", ()),
                )
                for fn, ep, rp, y in links:
                    link_data = dict(entrypath=ep)
                    if rp is not None:
                        link_data["requires_python"] = rp
                    if y is not None:
                        link_data["yanked"] = y
                    yield (f"{new_relpath}/{fn}", "MIRRORFILE", back_serial, link_data)
            elif keyname == "PROJVERSION":
                value = get_mutable_deepcopy(val)
                if value is None:
                    elinks = {}
                else:
                    elinks = value.pop("+elinks", {})
                    assert not any(x.startswith("+") for x in value)
                yield (new_relpath, new_keyname, back_serial, value)
                for elink in elinks:
                    path = Path(elink["entrypath"])
                    name = path.name
                    rel = elink["rel"]
                    elink["hashes"] = hashes = Digests(_hashes) if (_hashes := elink.get("hashes")) else Digests()
                    if hash_spec := elink.pop("hash_spec", None):
                        hashes.add_spec(hash_spec)
                    if rel == "doczip":
                        yield (new_relpath, "DOCZIP", back_serial, elink)
                    elif rel == "releasefile":
                        yield (
                            f"{new_relpath}/{name}",
                            "VERSIONFILE",
                            back_serial,
                            elink,
                        )
                    elif rel == "toxresult":
                        yield (f"{new_relpath}/{name}", "TOXRESULT", back_serial, elink)
                    else:
                        raise ValueError(f"{name=} {rel=} {elink=}")
            elif keyname == "PROJVERSIONS":
                yield (new_relpath, "PROJECTNAME", back_serial, Path(new_relpath).name)
                yield (new_relpath, new_keyname, back_serial, val)
            elif keyname == "STAGEFILE":
                if val is not None:
                    val["hashes"] = hashes = Digests(_hashes) if (_hashes := val.get("hashes")) else Digests()
                    if hash_spec := val.pop("hash_spec", None):
                        hashes.add_spec(hash_spec)
                yield (new_relpath, new_keyname, back_serial, val)
            elif keyname == "USER":
                value = get_mutable_deepcopy(val)
                indexes = {} if value is None else value.pop("indexes", {})
                yield (new_relpath, new_keyname, back_serial, value)
                yield (new_relpath, "INDEXLIST", back_serial, set(indexes))
                for name, config in indexes.items():
                    yield (f"{new_relpath}/{name}", "INDEX", back_serial, config)
            else:
                raise ValueError(f"{keyname=} {relpath=} {val=}")

    def iter_parents(self, key, params):
        parent_key = key.parent_key
        while parent_key is not None:
            yield (
                parent_key(**params)
                if isinstance(parent_key, PatternedKey)
                else parent_key
            )
            parent_key = parent_key.parent_key

    def iter_keys_to_fetch(self, changes: Any) -> Iterable[LocatedKey]:  # noqa: PLR0912
        keyfs = self.xom.keyfs
        for relpath, keyname, _back_serial, val in changes:
            if keyname in keyfs._keys:
                key = keyfs._keys[keyname]
                params = key.extract_params(relpath)
                yield key(**params)
                yield from self.iter_parents(key, params)
            elif keyname == "INDEXLIST":
                params = keyfs.INDEX.parent_key.extract_params(relpath)
                for index in val:
                    yield keyfs.INDEX(**params, index=index)
                    yield from self.iter_parents(key, params)
            elif keyname == "PROJNAMES":
                missing = set() if val is None else val
                if (
                    existing := self.recent_existing.get(("PROJECTNAME", relpath), None)
                ) is not None:
                    missing = missing.difference(existing)
                params = keyfs.PROJECTNAME.parent_key.extract_params(relpath)
                for project in missing:
                    key = keyfs.PROJECTNAME(**params, project=project)
                    yield key
                    yield from self.iter_parents(key, params)
            elif keyname == "PROJSIMPLELINKS":
                missing = {x[0] for x in (() if val is None else val.get("links", ()))}
                if (
                    existing := self.recent_existing.get(("MIRRORFILE", relpath), None)
                ) is not None:
                    missing = missing.difference(existing)
                params = keyfs.MIRRORFILE.parent_key.extract_params(relpath)
                for filename in missing:
                    key = keyfs.MIRRORFILE(**params, filename=filename)
                    yield key
                    yield from self.iter_parents(key, params)
            elif keyname == "PROJVERSIONS":
                params = keyfs.VERSION.parent_key.extract_params(relpath)
                for version in () if val is None else val:
                    key = keyfs.VERSION(**params, version=version)
                    yield key
                    yield from self.iter_parents(key, params)
            elif keyname == "USERLIST":
                for user in () if val is None else val:
                    yield keyfs.USER(user=user)
            else:
                raise ValueError(f"{keyname=} {relpath=} {val=}")

    def get_keys(self, keys_to_fetch: Iterable[LocatedKey]) -> dict[ULIDKey, KeyData]:
        keys = {}
        keynames_by_parent = self.xom.keyfs.keynames_by_parent
        tx = self.xom.keyfs.tx
        keyname_keys = defaultdict(set)
        for key in keys_to_fetch:
            keyname_keys[key.key_name].add(key)
        for key_names in keynames_by_parent:
            _keys = []
            for key_name in key_names:
                for _key in keyname_keys[key_name]:
                    parent_deleted = False
                    parent_key = _key.parent_key
                    while parent_key is not None:
                        if isinstance(parent_key, ULIDKey):
                            break
                        if keys.get(parent_key, deleted) is deleted:
                            parent_deleted = True
                            break
                        parent_key = parent_key.parent_key
                    if parent_deleted:
                        continue
                    if _key.parent_key is not None:
                        _key.parent_key = keys[_key.parent_key]
                    _keys.append(_key)
            if not _keys:
                continue
            keys.update(
                dict(
                    tx.resolve_keys(
                        _keys, fetch=True, fill_cache=True, new_for_missing=True
                    )
                )
            )
        return keys

    def make_records(self, serial: int, changes: Any) -> Sequence[Record]:
        changes = list(self.iter_migrated_changes(changes))
        keys_to_fetch = set(self.iter_keys_to_fetch(changes))
        threadlog.info("Fetching keys: %s", Counter(x.key_name for x in keys_to_fetch))
        keys = self.get_keys(keys_to_fetch)
        # threadlog.info("Fetched keys")
        get_key_instance = self.xom.keyfs.get_key_instance
        _new_ulidkeys = set()
        new_keys = dict()

        def _new_ulidkey(key):
            while True:
                new_ulid_key = key.new_ulid()
                # when rapidly generating new ulid keys, we can get
                # duplicates due to the birthday paradox
                if new_ulid_key not in _new_ulidkeys:
                    _new_ulidkeys.add(new_ulid_key)
                    return new_ulid_key

        tx = self.xom.keyfs.tx
        for key, ulid_key in list(keys.items()):
            if ulid_key is deleted:
                del keys[key]
                continue
            (back_serial, old_key, old_val) = tx.get_original(ulid_key)
            if old_val is deleted:
                del keys[key]

        records: list[Record] = []
        userlist = Nameslist("USER", "", keys)
        nameslists = Nameslists()
        nameslists.add(userlist)
        threadlog.info("Processing %s changes", len(changes))
        for relpath, keyname, _back_serial, _val in changes:
            # if keyname == "PROJECTNAME":
            #     import pdb; pdb.set_trace()
            if keyname == "INDEXLIST":
                nameslist = Nameslist("INDEX", relpath, keys)
                nameslists.add(nameslist)
                new_names = set() if _val is None else _val
                nameslist.set_new(new_names)
                for name in nameslist.existing.difference(new_names):
                    changes.append((f"{relpath}/{name}", "INDEX", -1, None))
                continue
            if keyname == "PROJNAMES":
                nameslist = Nameslist("PROJECTNAME", relpath, keys)
                if (
                    existing := self.recent_existing.get(nameslist.key, None)
                ) is not None:
                    nameslist.existing.update(existing)
                nameslists.add(nameslist)
                new_names = set() if _val is None else _val
                nameslist.set_new(new_names)
                for name in nameslist.existing.difference(new_names):
                    changes.append((f"{relpath}/{name}", "PROJECTNAME", -1, None))
                continue
            if keyname == "PROJSIMPLELINKS":
                nameslist = Nameslist("MIRRORFILE", relpath, keys)
                if (
                    existing := self.recent_existing.get(nameslist.key, None)
                ) is not None:
                    nameslist.existing.update(existing)
                nameslists.add(nameslist)
                val = dict(links=()) if _val is None else _val
                nameslists[("MIRRORFILE", relpath)].set_new(
                    {x[0] for x in val.get("links", ())}
                )
                continue
            if keyname == "PROJVERSIONS":
                nameslist = Nameslist("VERSION", relpath, keys)
                nameslists.add(nameslist)
                new_names = set() if _val is None else _val
                nameslist.set_new(new_names)
                continue
            if keyname == "USERLIST":
                userlist.set_new({} if _val is None else _val)
                continue
            try:
                key = get_key_instance(keyname, relpath)
            except AssertionError as e:
                raise ValueError(f"{keyname=} {relpath=} {_val=}") from e
            if key is None:
                raise ValueError(f"{keyname=} {relpath=} {_val=}")
            if (parent_key := key.parent_key) is not None:
                assert parent_key in keys
                if not isinstance(parent_key, ULIDKey):
                    key.parent_key = keys[parent_key]
            old_key: ULIDKey | Absent
            old_val: KeyFSTypesRO | Absent | Deleted
            if key not in keys:
                keys[key] = _new_ulidkey(key)
            assert key in keys
            ulid_key = keys[key]
            if ulid_key.exists():
                (back_serial, old_key, old_val) = tx.get_original(ulid_key)
            else:
                if key in new_keys:
                    continue
                new_keys[key] = ulid_key
                back_serial = ulid_key.back_serial
                old_key = absent
                old_val = absent
            val = deleted if _val is None else _val
            if old_val == val and ulid_key == old_key:
                continue
            if val is deleted and old_val in (absent, deleted):
                continue
            records.append(Record(ulid_key, val, back_serial, old_key, old_val))
        # threadlog.info("Checking %s records", len(records))
        nameslists.check(records)
        for nameslist in nameslists.lists.values():
            if nameslist.keyname == "PROJECTNAME":
                self.recent_existing.put(nameslist.key, nameslist.value)
        return records

    def set_primary_uuid(self, primary_uuid):
        pass

    def update_primary_serial(self, serial, *, update_sync=True, ignore_lower=False):
        self.primary_serial = serial


def importprimary(pluginmanager=None, argv=None):
    """devpi-import command line entry point."""
    argv = (
        sys.argv
        if argv is None
        # for tests
        else [str(x) for x in argv]
    )
    with CommandRunner(pluginmanager=pluginmanager) as runner:
        parser = runner.create_parser(
            description="Import previously exported data into a new devpi-server instance.",
            add_help=False,
        )
        parser.add_help_option()
        parser.add_secretfile_option()
        parser.add_configfile_option()
        parser.add_logging_options()
        parser.add_storage_options()
        parser.add_init_options()
        parser.add_import_options()
        parser.add_hard_links_option()
        parser.add_primary_url_option()
        config = runner.get_config(argv, parser=parser)
        runner.configure_logging(config.args)
        if not config.nodeinfo_path.exists():
            sdir = config.server_path
            if not (sdir.exists() and len(list(sdir.iterdir())) >= 2):
                set_state_version(config, DATABASE_VERSION)
            xom = xom_from_config(config, init=True)
            init_default_indexes(xom)
        else:
            xom = xom_from_config(config)
        replica_import = ReplicaImport(xom)
        replica_import.fetch()
    return runner.return_code or 0


if __name__ == "__main__":
    importprimary()
