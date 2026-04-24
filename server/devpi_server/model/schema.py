from __future__ import annotations

from devpi_server.filestore import FileEntry
from devpi_server.keyfs_schema import KeyFSSchema
from devpi_server.keyfs_types import is_dict_key
from devpi_server.markers import deleted
from devpi_server.normalized import normalize_name
from devpi_server.readonly import DictViewReadonly
from devpi_server.readonly import SetViewReadonly
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from devpi_server.keyfs import KeyChangeEvent
    from devpi_server.keyfs import KeyFS
    from devpi_server.main import XOM


class EventSubscribers:
    """the 'on_' functions are called within in the notifier thread."""

    def __init__(self, xom):
        self.xom = xom

    def on_changed_version_config(self, ev: KeyChangeEvent) -> None:
        """when version config is changed for a project in a stage"""
        params = ev.data.key.params
        keyfs = self.xom.keyfs
        hook = self.xom.config.hook
        with keyfs.read_transaction(at_serial=ev.at_serial) as tx:
            # find out if metadata changed
            if ev.data.back_serial == -1:
                old = {}
            else:
                assert ev.data.back_serial < ev.at_serial
                try:
                    old = tx.get_value_at(ev.data.key, ev.data.back_serial)
                except KeyError:
                    old = {}

            metadata = ev.data.value
            if metadata != old:
                source = old if metadata is deleted else metadata
                if not isinstance(source, (dict, DictViewReadonly)):
                    return
                stage = self.xom.model.getstage(params["user"], params["index"])
                hook.devpiserver_on_changed_versiondata(
                    stage=stage,
                    project=source["name"],
                    version=source["version"],
                    metadata=None if metadata is deleted else metadata,
                )

    def on_changed_file_entry(self, ev: KeyChangeEvent) -> None:
        """when a file entry is modified."""
        params = ev.data.key.params
        user = params.get("user")
        index = params.get("index")
        keyfs = self.xom.keyfs
        with keyfs.read_transaction(at_serial=ev.at_serial):
            stage = self.xom.model.getstage(user, index)
            if stage is not None and stage.index_type == "mirror":
                return  # we don't trigger on file changes of pypi mirror
            assert is_dict_key(ev.data.key)
            entry = FileEntry(ev.data.key, meta=ev.data.value)
            if not entry.project or not entry.version:
                # the entry was deleted
                self.xom.config.hook.devpiserver_on_remove_file(
                    stage=stage,
                    relpath=ev.data.key.relpath,
                )
                return
            name = entry.project
            assert name == normalize_name(name)
            linkstore = stage.get_linkstore_perstage(name, entry.version)
            links = linkstore.get_links(basename=entry.basename)
            if len(links) == 1:
                self.xom.config.hook.devpiserver_on_upload(
                    stage=stage, project=name, version=entry.version, link=links[0]
                )

    def on_mirror_initialnames(self, ev: KeyChangeEvent) -> None:
        """when projectnames are first loaded into a mirror."""
        params = ev.data.key.params
        user = params.get("user")
        index = params.get("index")
        keyfs = self.xom.keyfs
        with keyfs.read_transaction(at_serial=ev.at_serial):
            stage = self.xom.model.getstage(user, index)
            if stage is not None and stage.index_type == "mirror":
                self.xom.config.hook.devpiserver_mirror_initialnames(
                    stage=stage, projectnames=stage.list_projects_perstage()
                )

    def on_changed_index(self, ev: KeyChangeEvent) -> None:
        """when index data changes."""
        params = ev.data.key.params
        username = params.get("user")
        indexname = params.get("index")
        keyfs = self.xom.keyfs
        with keyfs.read_transaction(at_serial=ev.at_serial) as tx:
            if ev.data.back_serial > -1:
                try:
                    old = tx.get_value_at(ev.data.key, ev.data.back_serial)
                except KeyError:
                    # the user was previously deleted
                    old = None
            else:
                old = None
            if old is not None:
                # we only care about new stages
                return
            stage = self.xom.model.getstage(username, indexname)
            if stage is None:
                # deleted
                return
            self.xom.config.hook.devpiserver_stage_created(stage=stage)


class Schema(KeyFSSchema):
    # users, index and project configuration
    USER = KeyFSSchema.decl_patterned_key(
        "USER",
        "{user}",
        None,
        dict,
        DictViewReadonly,
    )
    INDEX = KeyFSSchema.decl_patterned_key(
        "INDEX",
        "{index}",
        USER,
        dict,
        DictViewReadonly,
    )
    PROJECT = KeyFSSchema.decl_patterned_key(
        "PROJECT",
        "{project}",
        INDEX,
        dict,
        DictViewReadonly,
    )

    # type mirror related data
    FILE_NOHASH = KeyFSSchema.decl_patterned_key(
        "FILE_NOHASH",
        "+e/{dirname}/{basename}",
        INDEX,
        dict,
        DictViewReadonly,
    )
    PROJECTCACHEINFO = KeyFSSchema.decl_anonymous_key(
        "PROJECTCACHEINFO",
        PROJECT,
        dict,
        DictViewReadonly,
    )
    MIRRORFILE = KeyFSSchema.decl_patterned_key(
        "MIRRORFILE",
        "{filename}",
        PROJECT,
        dict,
        DictViewReadonly,
    )
    MIRRORNAMESINIT = KeyFSSchema.decl_anonymous_key(
        "MIRRORNAMESINIT",
        INDEX,
        int,
        int,
    )

    # project and version related
    SIMPLEDATA = KeyFSSchema.decl_patterned_key(
        "SIMPLEDATA",
        "{version}/{filename}",
        PROJECT,
        dict,
        DictViewReadonly,
    )
    VERSION = KeyFSSchema.decl_patterned_key(
        "VERSION",
        "{version}",
        PROJECT,
        dict,
        DictViewReadonly,
    )
    VERSIONMETADATA = KeyFSSchema.decl_patterned_key(
        "VERSIONMETADATA",
        "{version}",
        PROJECT,
        dict,
        DictViewReadonly,
    )
    DOCZIP = KeyFSSchema.decl_patterned_key(
        "DOCZIP",
        "{version}",
        PROJECT,
        dict,
        DictViewReadonly,
    )
    TOXRESULT = KeyFSSchema.decl_patterned_key(
        "TOXRESULT",
        "{filename}",
        VERSION,
        dict,
        DictViewReadonly,
    )
    VERSIONFILE = KeyFSSchema.decl_patterned_key(
        "VERSIONFILE",
        "{filename}",
        VERSION,
        dict,
        DictViewReadonly,
    )
    FILE = KeyFSSchema.decl_patterned_key(
        "FILE",
        "+f/{hashdir_a}/{hashdir_b}/{filename}",
        INDEX,
        dict,
        DictViewReadonly,
    )

    # files related
    DIGESTULIDS = KeyFSSchema.decl_patterned_key(
        "DIGESTULIDS",
        "{digest}",
        None,
        set,
        SetViewReadonly,
    )

    def register_key_subscribers(self, xom: XOM, keyfs: KeyFS) -> None:
        sub = EventSubscribers(xom)
        notifier = keyfs.notifier
        notifier.on_key_change(self.VERSIONMETADATA, sub.on_changed_version_config)
        notifier.on_key_change(self.FILE, sub.on_changed_file_entry)
        notifier.on_key_change(self.MIRRORNAMESINIT, sub.on_mirror_initialnames)
        notifier.on_key_change(self.INDEX, sub.on_changed_index)
