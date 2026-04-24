from __future__ import annotations

from .exceptions import InvalidIndex
from .exceptions import InvalidIndexconfig
from .exceptions import ReadonlyIndex
from devpi_server.log import threadlog
from operator import iconcat
from typing import TYPE_CHECKING
import functools


if TYPE_CHECKING:
    from .local import BaseStage


def get_stage_customizer_classes(xom):
    customizer_classes: list[tuple[str, type]] = functools.reduce(
        iconcat, xom.config.hook.devpiserver_get_stage_customizer_classes(), []
    )
    return dict(customizer_classes)


def get_stage_customizer_class(xom, index_type):
    index_customizers = get_stage_customizer_classes(xom)
    cls = index_customizers.get(index_type)
    if cls is None:
        threadlog.warn("unknown index type %r" % index_type)
        cls = UnknownCustomizer
    if not issubclass(cls, BaseStageCustomizer):
        # we add the BaseStageCustomizer here to keep plugins simpler
        cls = type(cls.__name__, (cls, BaseStageCustomizer), dict(cls.__dict__))
    cls.InvalidIndex = InvalidIndex
    cls.InvalidIndexconfig = InvalidIndexconfig
    cls.ReadonlyIndex = ReadonlyIndex
    return cls


class BaseStageCustomizer:
    readonly = False

    def __init__(self, stage: BaseStage) -> None:
        self.stage = stage
        self.hooks = self.stage.xom.config.hook

    # get_principals_for_* methods for each of the following permissions:
    # upload, toxresult_upload, index_delete, index_modify,
    # del_entry, del_project, del_verdata
    # also see __acl__ method of BaseStage

    def get_principals_for_pkg_read(self, restrict_modify=None):
        principals = self.hooks.devpiserver_stage_get_principals_for_pkg_read(
            ixconfig=self.stage.ixconfig
        )
        principals = {":ANONYMOUS:"} if principals is None else set(principals)
        # admins should always be able to read the packages
        if restrict_modify is None:
            principals.add("root")
        else:
            principals.update(restrict_modify)
        return principals

    def get_principals_for_upload(
        self,
        restrict_modify=None,  # noqa: ARG002 - API
    ):
        return self.stage.ixconfig.get("acl_upload", [])

    def get_principals_for_toxresult_upload(
        self,
        restrict_modify=None,  # noqa: ARG002 - API
    ):
        return self.stage.ixconfig.get("acl_toxresult_upload", [":ANONYMOUS:"])

    def get_principals_for_index_delete(self, restrict_modify=None):
        if restrict_modify is None:
            modify_principals = {"root", self.stage.username}
        else:
            modify_principals = restrict_modify
        return modify_principals

    get_principals_for_index_modify = get_principals_for_index_delete

    def get_principals_for_del_entry(self, restrict_modify=None):
        modify_principals = set(self.stage.ixconfig.get("acl_upload", []))
        if restrict_modify is None:
            modify_principals.update(["root", self.stage.username])
        else:
            modify_principals.update(restrict_modify)
        return modify_principals

    get_principals_for_del_project = get_principals_for_del_entry
    get_principals_for_del_verdata = get_principals_for_del_entry

    def get_possible_indexconfig_keys(self):
        """Returns all possible custom index config keys.

        These are in addition to the existing keys of a regular private index.
        """
        return ()

    def get_default_config_items(self):
        """Returns a list of defaults as key/value tuples.

        Only applies to new keys, not existing options of a private index.
        """
        return ()

    def normalize_indexconfig_value(self, key, value):
        """Returns value normalized to the type stored in the database.

        A return value of None is treated as an error.
        Can raise InvalidIndexconfig.
        Will only be called for custom options, not for existing options
        of a private index.
        """

    def validate_config(self, oldconfig, newconfig):
        """Validates the index config.

        Can raise InvalidIndexconfig."""

    def on_modified(self, request, oldconfig):
        """Called after index was created or modified via a request.

        Can do further changes in the current transaction.

        Must use request.apifatal method to indicate errors instead
        of raising HTTPException responses.

        Other exceptions will be handled."""

    def get_projects_filter_iter(
        self,
        projects,  # noqa: ARG002 - API
    ):
        """Called when a list of projects is returned.

        Returns None for no filtering, or an iterator returning
        True for items to keep and False for items to remove."""
        return

    def get_versions_filter_iter(
        self,
        project,  # noqa: ARG002 - API
        versions,  # noqa: ARG002 - API
    ):
        """Called when a list of versions is returned.

        Returns None for no filtering, or an iterator returning
        True for items to keep and False for items to remove."""
        return

    def get_simple_links_filter_iter(
        self,
        project,  # noqa: ARG002 - API
        links,  # noqa: ARG002 - API
    ):
        """Called when a list of simple links is returned.

        Returns None for no filtering, or an iterator returning
        True for items to keep and False for items to remove.
        The size of the tuples in links might grow, develop defensively."""
        return


class UnknownCustomizer(BaseStageCustomizer):
    readonly = True

    # prevent uploads and deletions besides complete index removal
    def get_principals_for_index_modify(
        self,
        restrict_modify=None,  # noqa: ARG002 - API
    ):
        return []

    get_principals_for_upload = get_principals_for_index_modify
    get_principals_for_toxresult_upload = get_principals_for_index_modify
    get_principals_for_del_entry = get_principals_for_index_modify
    get_principals_for_del_project = get_principals_for_index_modify
    get_principals_for_del_verdata = get_principals_for_index_modify
