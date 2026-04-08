from __future__ import annotations

from .base import BaseIndex
from .config import ConfigField
from .config import ensure_acl_list
from .config import ensure_boolean
from .config import ensure_list
from .config import normalize_bases
from .config import normalize_trust_inheritance
from .config import normalize_whitelist_name
from .customizer import BaseIndexCustomizer
from .exceptions import MissesRegistration
from .exceptions import MissesVersion
from .exceptions import ReadonlyIndex
from .links import ELink
from .links import Rel
from .links import SimplelinkMeta
from contextlib import suppress
from devpi_common.types import cached_property
from devpi_common.types import ensure_unicode
from devpi_common.validation import validate_metadata
from devpi_server.config import hookimpl
from devpi_server.filestore import AbsPath
from devpi_server.filestore import Digests
from devpi_server.filestore import get_hashes
from devpi_server.log import threadlog
from devpi_server.markers import Absent
from devpi_server.markers import Deleted
from devpi_server.markers import absent
from devpi_server.markers import deleted
from devpi_server.normalized import normalize_name
from devpi_server.readonly import DictViewReadonly
from devpi_server.readonly import ensure_deeply_readonly
from devpi_server.readonly import get_mutable_deepcopy
from functools import partial
from lazy import lazy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import overload


if TYPE_CHECKING:
    from .links import SimpleLinks
    from collections.abc import Sequence
    from devpi_server.filestore import BaseFileEntry
    from devpi_server.interfaces import ContentOrFile
    from devpi_server.keyfs_types import LocatedKey
    from devpi_server.keyfs_types import SearchKey
    from devpi_server.main import XOM
    from devpi_server.markers import Unknown
    from devpi_server.normalized import NormalizedName
    from typing import Any
    from typing import Literal

    LocalIndexType = type["LocalIndex"]


VERSIONDATA_DESCRIPTION_SIZE_THRESHOLD = 8192


class LocalIndex(BaseIndex):
    metadata_keys = (
        "name",
        "version",
        # additional meta-data
        "metadata_version",
        "summary",
        "home_page",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "license",
        "description",
        "keywords",
        "platform",
        "classifiers",
        "download_url",
        "supported_platform",
        "comment",
        # PEP 314
        "provides",
        "requires",
        "obsoletes",
        # Metadata 1.2
        "project_urls",
        "provides_dist",
        "obsoletes_dist",
        "requires_dist",
        "requires_external",
        "requires_python",
        # Metadata 2.1
        "description_content_type",
        "provides_extras",
        # Metadata 2.2
        "dynamic",
        # Metadata 2.4
        "license_expression",
        "license_file",
    )
    metadata_list_fields = (
        "platform",
        "classifiers",
        "supported_platform",
        # PEP 314
        "provides",
        "requires",
        "obsoletes",
        # Metadata 1.2
        "project_urls",
        "provides_dist",
        "obsoletes_dist",
        "requires_dist",
        "requires_external",
        # Metadata 2.1
        "provides_extras",
        # Metadata 2.2
        "dynamic",
        # Metadata 2.4
        "license_file",
    )

    use_external_url = False

    def __init__(
        self,
        xom: XOM,
        username: str,
        index: str,
        ixconfig: DictViewReadonly[str, Any],
        customizer_cls: type,
    ) -> None:
        super().__init__(xom, username, index, ixconfig, customizer_cls)
        self.http = xom.http

    @property
    def no_project_list(self) -> Literal[False]:
        return False

    def get_indexconfig_fields(self) -> Sequence[ConfigField]:
        return [
            ConfigField(
                name="acl_toxresult_upload",
                default=[":ANONYMOUS:"],
                normalize=ensure_acl_list,
            ),
            ConfigField(
                name="acl_upload", default=[self.username], normalize=ensure_acl_list
            ),
            ConfigField(
                name="bases",
                default=(),
                normalize=partial(normalize_bases, self.xom.model),
            ),
            ConfigField(name="custom_data"),
            ConfigField(name="description", normalize=str),
            ConfigField(
                name="mirror_whitelist",
                default=[],
                normalize=lambda v: [
                    normalize_whitelist_name(x) for x in ensure_list(v)
                ],
            ),
            ConfigField(name="title", normalize=str),
            ConfigField(
                name="trust_inheritance_rules_from",
                normalize=normalize_trust_inheritance,
            ),
            ConfigField(name="volatile", default=True, normalize=ensure_boolean),
        ]

    def delete(self) -> None:
        # delete all projects on this index
        for name in self.list_projects_perstage():
            self.del_project(name)
        BaseIndex.delete(self)

    #
    # registering project and version metadata
    #

    def set_versiondata(self, metadata):
        """register metadata.  Raises ValueError in case of metadata
        errors."""
        if self.customizer.readonly:
            raise ReadonlyIndex("index is marked read only")
        metadata = {k: v for k, v in metadata.items() if not k.startswith("+")}
        # use a copy, as validate_metadata actually removes metadata_version
        validate_metadata(dict(metadata))
        self._set_versiondata(metadata)

    @overload
    def key_doczip(
        self, project: NormalizedName | str, version: str
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_doczip(
        self, project: NormalizedName | str, version: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_doczip(
        self, project: NormalizedName | str, version: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.DOCZIP
        (kw, meth) = (
            ({}, key.search) if version is None else (dict(version=version), key.locate)
        )
        return meth(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    @overload
    def key_toxresult(
        self, project: NormalizedName | str, version: str, filename: str
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_toxresult(
        self, project: NormalizedName | str, version: str, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_toxresult(
        self, project: NormalizedName | str, version: str, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.TOXRESULT
        (kw, meth) = (
            ({}, key.search)
            if filename is None
            else (dict(filename=filename), key.locate)
        )
        return meth(
            user=self.username,
            index=self.index,
            project=normalize_name(project),
            version=version,
            **kw,
        )

    @overload
    def key_versionfile(
        self, project: NormalizedName | str, version: str, filename: str
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_versionfile(
        self, project: NormalizedName | str, version: str, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_versionfile(
        self, project: NormalizedName | str, version: str, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.VERSIONFILE
        (kw, meth) = (
            ({}, key.search)
            if filename is None
            else (dict(filename=filename), key.locate)
        )
        return meth(
            user=self.username,
            index=self.index,
            project=normalize_name(project),
            version=version,
            **kw,
        )

    @overload
    def key_versionmetadata(
        self, project: NormalizedName | str, version: str
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_versionmetadata(
        self, project: NormalizedName | str, version: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_versionmetadata(
        self, project: NormalizedName | str, version: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.VERSIONMETADATA
        (kw, meth) = (
            ({}, key.search) if version is None else (dict(version=version), key.locate)
        )
        return meth(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    def _set_versiondata(self, metadata):
        assert "+elinks" not in metadata
        project = normalize_name(metadata["name"])
        version = metadata["version"]
        self.add_project_name(project)
        key_version = self.key_version(project, version).with_resolved_parent()
        with key_version.update() as versiondata:
            if (rp := metadata.get("requires_python")) is not None:
                versiondata["requires_python"] = rp
        if (
            len(content := metadata.get("description", "").encode())
            > VERSIONDATA_DESCRIPTION_SIZE_THRESHOLD
        ):
            entry = self.filestore.store(
                user=self.username,
                index=self.index,
                basename=f"{project}-{version}.readme",
                content_or_file=content,
                hashes=get_hashes(content),
            )
            metadata["description"] = {
                "relpath": entry.index_relpath,
                "hashes": entry.hashes,
            }
        key_versionmetadata = self.key_versionmetadata(
            project, version
        ).with_resolved_parent()
        with key_versionmetadata.update() as versionmetadata:
            versionmetadata.clear()
            versionmetadata.update(metadata)
        threadlog.info("set_metadata %s-%s", project, version)

    def add_project_name(self, project: NormalizedName | str) -> None:
        project = normalize_name(project)
        key_project = self.key_project(project).with_resolved_parent()
        if not key_project.exists() and self.customizer.readonly:
            raise ReadonlyIndex("index is marked read only")
        with key_project.update() as projectdata:
            projectdata["name"] = project.original
        lazy.invalidate(self, "index_bases")

    def del_project(self, project: NormalizedName | str) -> None:
        project = normalize_name(project)
        key_version = self.key_version(project).with_resolved_parent()
        versions = {x.name for x in key_version.iter_ulidkeys()}
        for version in versions:
            self.del_versiondata(project, version, cleanup=False)
        threadlog.info("deleting project %s", project)
        self.key_project(project).with_resolved_parent().delete()
        lazy.invalidate(self, "index_bases")

    def del_versiondata(
        self, project: NormalizedName | str, version: str, *, cleanup: bool = True
    ) -> None:
        project = normalize_name(project)
        if not self.has_project_perstage(project):
            raise self.NotFound(
                "project %r not found on stage %r" % (project, self.name)
            )
        if not self.key_versionmetadata(project, version).exists(resolve_parents=True):
            raise self.NotFound(
                "version %r of project %r not found on stage %r"
                % (version, project, self.name)
            )
        linkstore = self.get_mutable_linkstore_perstage(project, version)
        linkstore.remove_links()
        key_versionmetadata = self.key_versionmetadata(
            project, version
        ).with_resolved_parent()
        self.key_version(project, version).with_resolved_parent().delete()
        metadata = key_versionmetadata.get()
        if "description" in metadata and not isinstance(metadata["description"], str):
            entry = self.filestore.get_file_entry(
                AbsPath(
                    f"{self.username}/{self.index}/{metadata['description']['relpath']}"
                )
            )
            if entry is not None:
                entry.delete()
        key_versionmetadata.delete()
        if cleanup:
            key_version = self.key_version(project).with_resolved_parent()
            has_versions = next(key_version.iter_ulidkeys(), absent) is not absent
            if not has_versions:
                self.del_project(project)

    def del_entry(self, entry: BaseFileEntry, *, cleanup: bool = True) -> None:
        # we need to store project and version for use in cleanup part below
        project = entry.project
        version = entry.version
        linkstore = self.get_mutable_linkstore_perstage(project, version)
        linkstore.remove_links(basename=entry.basename)
        entry.delete()
        if cleanup and not linkstore.get_links():
            self.del_versiondata(project, version)

    def list_versions_perstage(self, project):
        project = normalize_name(project)
        if not self.has_project_perstage(project):
            return set()
        key_version = self.key_version(project).with_resolved_parent()
        return {x.name for x in key_version.iter_ulidkeys()}

    @cached_property
    def _key_name_rel_map(self) -> dict[str, str]:
        return {
            self.keyfs.schema.DOCZIP.key_name: str(Rel.DocZip),
            self.keyfs.schema.TOXRESULT.key_name: str(Rel.ToxResult),
            self.keyfs.schema.VERSIONFILE.key_name: str(Rel.ReleaseFile),
        }

    def _get_elink_from_entry(self, entry: BaseFileEntry) -> ELink | None:
        project = entry.project
        version = entry.version
        basename = entry.basename
        keys = [
            self.key_doczip(project, version).with_resolved_parent(),
            self.key_toxresult(project, version, basename).with_resolved_parent(),
            self.key_versionfile(project, version, basename).with_resolved_parent(),
        ]
        key_name_rel_map = self._key_name_rel_map
        username = self.username
        index = self.index
        result = []
        for k, v in self.keyfs.tx.iter_ulidkey_values_for(keys):
            assert isinstance(v, DictViewReadonly)
            if Path(v["relpath"]).name != basename:
                continue
            data = dict((*v.items(), ("rel", key_name_rel_map[k.key_name])))
            data["entrypath"] = f"{username}/{index}/{data['relpath']}"
            if "for_relpath" in data:
                data["for_entrypath"] = f"{username}/{index}/{data['for_relpath']}"
            result.append(data)
        if not result:
            return None
        (data,) = result
        return ELink(entry, data)

    def _get_elinks(
        self, project: str, version: str, *, rel: Rel | None = None
    ) -> list:
        if not self.key_versionmetadata(project, version).exists(resolve_parents=True):
            return []
        rels = set(Rel) if rel is None else {rel}
        keys: list[LocatedKey | SearchKey] = list()
        if Rel.DocZip in rels:
            keys.append(self.key_doczip(project, version).with_resolved_parent())
        if Rel.ToxResult in rels:
            keys.append(self.key_toxresult(project, version).with_resolved_parent())
        if Rel.ReleaseFile in rels:
            keys.append(self.key_versionfile(project, version).with_resolved_parent())
        username = self.username
        index = self.index
        key_name_rel_map = self._key_name_rel_map
        result = []
        for k, v in self.keyfs.tx.iter_ulidkey_values_for(keys):
            assert isinstance(v, DictViewReadonly)
            data = dict((*v.items(), ("rel", key_name_rel_map[k.key_name])))
            data["entrypath"] = f"{username}/{index}/{data['relpath']}"
            if "for_relpath" in data:
                data["for_entrypath"] = f"{username}/{index}/{data['for_relpath']}"
            result.append(data)
        return result

    def get_last_project_change_serial_perstage(self, project, at_serial=None):
        project = normalize_name(project)
        tx = self.keyfs.tx
        if at_serial is None:
            at_serial = tx.at_serial
        (last_serial, _projectname_ulid, projectdata) = tx.get_last_serial_and_value_at(
            self.key_project(project).with_resolved_parent(), at_serial
        )
        if isinstance(projectdata, (Absent, Deleted)):
            # the whole index never existed or was deleted
            return last_serial
        for version_keydata in tx.conn.iter_keys_at_serial(
            (self.key_version(project).with_resolved_parent(),),
            at_serial=at_serial,
            fill_cache=False,
            with_deleted=True,
        ):
            last_serial = max(last_serial, version_keydata.key.last_serial)
            if last_serial >= at_serial:
                return last_serial
            if version_keydata.value in (absent, deleted):
                continue
            version = version_keydata.key.name
            key_versionmetadata = self.key_versionmetadata(
                project, version
            ).with_resolved_parent()
            (
                versionmetadata_serial,
                _versionmetadata_info_ulid,
                _versionmetadata_info,
            ) = tx.get_last_serial_and_value_at(key_versionmetadata, at_serial)
            last_serial = max(last_serial, versionmetadata_serial)
            if last_serial >= at_serial:
                return last_serial
            for versionfile_key in tx.conn.iter_ulidkeys_at_serial(
                (self.key_versionfile(project, version).with_resolved_parent(),),
                at_serial=at_serial,
                fill_cache=False,
                with_deleted=True,
            ):
                last_serial = max(last_serial, versionfile_key.last_serial)
                if last_serial >= at_serial:
                    return last_serial
        return last_serial

    def get_versiondata_perstage(
        self,
        project: NormalizedName | str,
        version: str,
        *,
        with_elinks: bool = True,
    ) -> DictViewReadonly[str, Any]:
        project = normalize_name(project)
        key_versionmetadata = self.key_versionmetadata(project, version)
        if not key_versionmetadata.exists(resolve_parents=True):
            return ensure_deeply_readonly({})
        verdata = key_versionmetadata.with_resolved_parent().get()
        if "description" in verdata and not isinstance(verdata["description"], str):
            entry = self.filestore.get_file_entry(
                AbsPath(
                    f"{self.username}/{self.index}/{verdata['description']['relpath']}"
                )
            )
            verdata = get_mutable_deepcopy(verdata)
            # if the file doesn't exist we return the dict as fallback
            if entry is not None:
                with suppress(FileNotFoundError):
                    verdata["description"] = entry.file_get_content().decode()
        assert "+elinks" not in verdata
        if with_elinks:
            elinks = self._get_elinks(project, version)
            if elinks:
                verdata = get_mutable_deepcopy(verdata)
                verdata["+elinks"] = elinks
        return ensure_deeply_readonly(verdata)

    def get_simplelinks_perstage(self, project: NormalizedName | str) -> SimpleLinks:
        links = self.SimpleLinks([])
        username = self.username
        index = self.index
        key_simpledata = self.key_simpledata(project).with_resolved_parent()
        for k, v in key_simpledata.iter_ulidkey_values():
            href = f"{username}/{index}/{v['relpath']}"
            if hash_spec := Digests(v["hashes"]).best_available_spec:
                href += "#" + hash_spec
            links.append(
                SimplelinkMeta(
                    (k.params["filename"], href, v.get("requires_python"), None),
                    core_metadata=True,
                )
            )
        return links

    def list_projects_perstage(self) -> dict[str, NormalizedName | str]:
        key_project = self.key_project.with_resolved_parent()
        return {k.name: v["name"] for k, v in key_project.iter_ulidkey_values()}

    def has_project_perstage(self, project: NormalizedName | str) -> bool | Unknown:
        return self.key_project(project).exists(resolve_parents=True)

    def has_version_perstage(self, project: str, version: str) -> bool:
        return self.key_versionmetadata(project, version).exists(resolve_parents=True)

    def store_releasefile(
        self,
        project: NormalizedName | str,
        version: str,
        filename: str,
        content_or_file: ContentOrFile,
        *,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        if self.customizer.readonly:
            raise ReadonlyIndex("index is marked read only")
        project = normalize_name(project)
        filename = ensure_unicode(filename)
        if not self.has_version_perstage(project, version):
            # There's a chance the version was guessed from the
            # filename, which might have swapped dashes to underscores
            if "_" in version:
                version = version.replace("_", "-")
                if not self.has_version_perstage(project, version):
                    raise MissesRegistration("%s-%s", project, version)
            else:
                raise MissesRegistration("%s-%s", project, version)
        linkstore = self.get_mutable_linkstore_perstage(project, version)
        link = linkstore.create_linked_entry(
            rel=Rel.ReleaseFile,
            basename=filename,
            content_or_file=content_or_file,
            hashes=hashes,
            last_modified=last_modified,
        )
        versiondata = (
            {}
            if version is None
            else self.key_version(project, version).with_resolved_parent().get()
        )
        key_simpledata = self.key_simpledata(
            project, (version, filename)
        ).with_resolved_parent()
        with key_simpledata.update() as simpledata:
            simpledata["relpath"] = link.entry.index_relpath
            simpledata["hashes"] = link.entry.hashes
            if rp := versiondata.get("requires_python"):
                simpledata["requires_python"] = rp
        return link

    def store_doczip(
        self,
        project: NormalizedName | str,
        version: str,
        content_or_file: ContentOrFile,
        *,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        if self.customizer.readonly:
            raise ReadonlyIndex("index is marked read only")
        project = normalize_name(project)
        if not version:
            version = self.get_latest_version_perstage(project)
            if not version:
                raise MissesVersion(
                    "doczip has no version and '%s' has no releases to derive one from",
                    project,
                )
            threadlog.info(
                "store_doczip: derived version of %s is %s", project, version
            )
        basename = f"{project}-{version}.doc.zip"
        with self.key_project(project).with_resolved_parent().update() as projectdata:
            projectdata["name"] = project
        linkstore = self.get_mutable_linkstore_perstage(project, version)
        return linkstore.create_linked_entry(
            rel=Rel.DocZip,
            basename=basename,
            content_or_file=content_or_file,
            hashes=hashes,
            last_modified=last_modified,
        )

    def get_doczip_link(self, project, version):
        """get link of documentation zip or None if no docs exists."""
        doczip = self.key_doczip(project, version).with_resolved_parent().get()
        if not doczip:
            return None
        entrypath = AbsPath(f"{self.username}/{self.index}/{doczip['relpath']}")
        entry = self.filestore.get_file_entry(entrypath)
        return ELink(entry, dict(doczip, rel=Rel.DocZip))

    def get_doczip_entry(self, project, version):
        """get entry of documentation zip or None if no docs exists."""
        link = self.get_doczip_link(project, version)
        return link.entry if link else None

    def get_doczip(self, project, version):
        """get documentation zip content or None if no docs exists."""
        link = self.get_doczip_link(project, version)
        return link.entry.file_get_content() if link else None

    def get_last_change_serial_perstage(self, at_serial=None):  # noqa: PLR0911, PLR0912
        tx = self.keyfs.tx
        if at_serial is None:
            at_serial = tx.at_serial
        last_serial = -1
        for project_keydata in tx.conn.iter_keys_at_serial(
            (self.key_project.with_resolved_parent(),),
            at_serial=at_serial,
            fill_cache=False,
            with_deleted=True,
        ):
            last_serial = max(last_serial, project_keydata.key.last_serial)
            projectdata = project_keydata.value
            if last_serial >= at_serial:
                return last_serial
            if isinstance(projectdata, Deleted):
                continue
            project = projectdata["name"]
            for doczip_keydata in tx.conn.iter_keys_at_serial(
                (self.key_doczip(project).with_resolved_parent(),),
                at_serial=at_serial,
                fill_cache=False,
                with_deleted=True,
            ):
                last_serial = max(last_serial, doczip_keydata.key.last_serial)
                if last_serial >= at_serial:
                    return last_serial
            for version_keydata in tx.conn.iter_keys_at_serial(
                (self.key_version(project).with_resolved_parent(),),
                at_serial=at_serial,
                fill_cache=False,
                with_deleted=True,
            ):
                last_serial = max(last_serial, version_keydata.key.last_serial)
                if last_serial >= at_serial:
                    return last_serial
                if version_keydata.value in (absent, deleted):
                    continue
                version = version_keydata.key.name
                key_versionmetadata = self.key_versionmetadata(
                    project, version
                ).with_resolved_parent()
                try:
                    (versionmetadata_serial, _versionmetadata) = (
                        tx.last_serial_and_value_at(key_versionmetadata, at_serial)
                    )
                except KeyError:
                    pass
                else:
                    last_serial = max(last_serial, versionmetadata_serial)
                for toxresult_key in tx.conn.iter_ulidkeys_at_serial(
                    (self.key_toxresult(project, version).with_resolved_parent(),),
                    at_serial=at_serial,
                    fill_cache=False,
                    with_deleted=True,
                ):
                    last_serial = max(last_serial, toxresult_key.last_serial)
                    if last_serial >= at_serial:
                        return last_serial
                for versionfile_key in tx.conn.iter_ulidkeys_at_serial(
                    (self.key_versionfile(project, version).with_resolved_parent(),),
                    at_serial=at_serial,
                    fill_cache=False,
                    with_deleted=True,
                ):
                    last_serial = max(last_serial, versionfile_key.last_serial)
                    if last_serial >= at_serial:
                        return last_serial
        # no project uploaded yet
        key_index = self.key_index.with_resolved_parent()
        (index_serial, _index_config) = tx.last_serial_and_value_at(
            key_index, at_serial
        )
        if last_serial >= index_serial:
            return last_serial
        return index_serial


class LocalIndexCustomizer(BaseIndexCustomizer):
    pass


@hookimpl
def devpiserver_get_stage_customizer_classes():
    # prevent plugins from installing their own under the reserved names
    return [("local", LocalIndexCustomizer)]
