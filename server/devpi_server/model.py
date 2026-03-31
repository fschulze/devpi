from __future__ import annotations

from .auth import hash_password
from .auth import verify_and_update_password_hash
from .compat import StrEnum
from .config import hookimpl
from .filestore import Digests
from .filestore import FileEntry
from .filestore import get_hashes
from .keyfs_schema import KeyFSSchema
from .keyfs_types import RelPath
from .keyfs_types import is_dict_key
from .log import threadlog
from .markers import Absent
from .markers import Deleted
from .markers import NotSet
from .markers import absent
from .markers import deleted
from .markers import notset
from .markers import unknown
from .normalized import normalize_name
from .readonly import DictViewReadonly
from .readonly import SetViewReadonly
from .readonly import ensure_deeply_readonly
from .readonly import get_mutable_deepcopy
from abc import ABC
from abc import abstractmethod
from attrs import define
from contextlib import suppress
from devpi_common.metadata import get_latest_version
from devpi_common.metadata import parse_version
from devpi_common.metadata import splitbasename
from devpi_common.types import cached_property
from devpi_common.types import ensure_unicode
from devpi_common.url import URL
from devpi_common.validation import validate_metadata
from functools import total_ordering
from lazy import lazy
from operator import iconcat
from pathlib import Path
from pyramid.authorization import Allow
from pyramid.authorization import Authenticated
from pyramid.authorization import Everyone
from time import gmtime
from time import strftime
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import overload
import enum
import functools
import getpass
import json
import re
import warnings


if TYPE_CHECKING:
    from .filestore import BaseFileEntry
    from .filestore import FileStore
    from .filestore import MutableFileEntry
    from .interfaces import ContentOrFile
    from .keyfs import KeyChangeEvent
    from .keyfs import KeyFS
    from .keyfs_types import LocatedKey
    from .keyfs_types import SearchKey
    from .keyfs_types import ULIDKey
    from .main import XOM
    from .markers import Unknown
    from .normalized import NormalizedName
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import MutableSequence
    from collections.abc import Sequence
    from devpi_common.metadata import Version
    from types import TracebackType
    from typing import Any
    from typing import Literal
    from typing import NotRequired
    from typing import Self

    RequiresPython = str | None
    Yanked = Literal[True] | str | None
    JoinedLink = tuple[str, str, RequiresPython, Yanked]


class FileLogEntry(TypedDict):
    what: str
    who: str | None
    when: str
    count: NotRequired[int]
    dst: NotRequired[str]
    src: NotRequired[str]


class Rel(StrEnum):
    DocZip = "doczip"
    ReleaseFile = "releasefile"
    ToxResult = "toxresult"


VERSIONDATA_DESCRIPTION_SIZE_THRESHOLD = 8192


def apply_filter_iter(items, filter_iter):
    for item in items:
        if next(filter_iter, True):
            yield item


def run_passwd(root, username):
    user = root.get_user(username)
    log = threadlog
    if user is None:
        log.error("user %r not found", username)
        return 1
    for i in range(3):
        pwd = getpass.getpass("enter password for %s: " % user.name)
        pwd2 = getpass.getpass("repeat password for %s: " % user.name)
        if pwd != pwd2:
            log.error("password don't match")
        else:
            break
    else:
        log.error("no password set")
        return 1
    user.modify(password=pwd)


class RemoveValue(object):
    """ Marker object for index configuration keys to remove. """


class ModelException(Exception):
    """ Base Exception. """
    def __init__(self, msg, *args):
        if args:
            msg = msg % args
        self.msg = msg
        Exception.__init__(self, msg)


class InvalidUser(ModelException):
    """ If a username is invalid or already in use. """


class InvalidIndex(ModelException):
    """ If a indexname is invalid or already in use. """


class ReadonlyIndex(ModelException):
    """ If a indexname is invalid or already in use. """


class NotFound(ModelException):
    """ If a project or version cannot be found. """


class UpstreamError(ModelException):
    """ If an upstream could not be reached or didn't respond correctly. """


class UpstreamNotFoundError(UpstreamError):
    """ If upstream returned a not found error. """


class UpstreamNotModified(ModelException):
    """ If upstream returned a not modified reply. """
    def __init__(self, msg, *args, etag=None):
        super().__init__(msg, *args)
        self.etag = etag


class MissesRegistration(ModelException):
    """ A prior registration of release metadata is required. """


class MissesVersion(ModelException):
    """ A version number is required. """


class NonVolatile(ModelException):
    """ A release is overwritten on a non volatile index. """

    link: ELink  # the conflicting link


class RootModel:
    """ per-process root model object. """

    def __init__(self, xom: XOM) -> None:
        self.xom = xom
        self.keyfs = xom.keyfs
        self.key_user = self.keyfs.schema.USER

    def create_user(self, username, password, **kwargs):
        if self.key_user.locate(username).exists():
            raise InvalidUser("username '%s' already exists" % username)
        if not is_valid_name(username):
            raise InvalidUser(
                "username '%s' contains characters that aren't allowed. "
                "Any ascii symbol besides -.@_ is blocked." % username)
        user = User(self, username)
        kwargs.update(created=strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))
        user._modify(password=password, **kwargs)
        if "email" in kwargs:
            threadlog.info("created user %r with email %r" % (username, kwargs["email"]))
        else:
            threadlog.info("created user %r" % username)
        # Call any user created hooks, passing along the newly created user object.
        self.xom.config.hook.devpiserver_user_created(user=user)
        return user

    def create_stage(self, user, index, type="stage", **kwargs):
        if user.key_index.with_resolved_parent()(index).exists():
            raise InvalidIndex("indexname '%s' already exists" % index)
        if not is_valid_name(index):
            raise InvalidIndex(
                "indexname '%s' contains characters that aren't allowed. "
                "Any ascii symbol besides -.@_ is blocked." % index)
        stage = user._getstage(index, type, {"type": type})
        if isinstance(stage.customizer, UnknownCustomizer):
            raise InvalidIndexconfig("unknown index type %r" % type)
        # apply default values for new indexes
        for key, value in stage.get_default_config_items():
            kwargs.setdefault(key, value)
        # apply default values from customizer class for new indexes
        for key, value in stage.customizer.get_default_config_items():
            kwargs.setdefault(key, value)
        stage._modify(**kwargs)
        threadlog.debug("created index %s: %s", stage.name, stage.ixconfig)
        return stage

    def delete_user(self, username: str) -> None:
        self.key_user.locate(username).delete()

    def delete_stage(self, username: str, index: str) -> None:
        stage = self.getstage(username, index)
        if stage is None:
            threadlog.info("index %s/%s does not exist", username, index)
            return
        stage.key_index.with_resolved_parent().delete()
        self.xom.del_singletons(f"{username}/{index}")

    def get_user(self, name: str) -> User | None:
        user = User(self, name)
        return user if user.key.exists() else None

    def get_userlist(self):
        # using iter_ulidkey_values pre-fetches values of users
        # which are then used in User
        return [
            User(self, key.name)
            for key, v in self.key_user.search().iter_ulidkey_values()
        ]

    def get_usernames(self):
        return {key.params["user"] for key in self.key_user.search().iter_ulidkeys()}

    def _get_user_and_index(self, user, index=None):
        assert isinstance(user, str)
        if index is None:
            assert not user.startswith('/')
            assert not user.endswith('/')
            user, index = user.split("/")
        else:
            assert isinstance(index, str)
        return user, index

    def getstage(self, user: str, index: str | None = None) -> BaseStage | None:
        (username, indexname) = self._get_user_and_index(user, index)
        _user = self.get_user(username)
        if _user is None:
            return None
        ixconfig = _user.key_index(indexname).with_resolved_parent().get_mutable()
        if not ixconfig:
            return None
        return _user._getstage(indexname, ixconfig["type"], ixconfig)


def ensure_boolean(value):
    if isinstance(value, bool):
        return value
    if not hasattr(value, "lower"):
        raise InvalidIndexconfig("Unknown boolean value %r." % value)
    if value.lower() in ["false", "no"]:
        return False
    if value.lower() in ["true", "yes"]:
        return True
    raise InvalidIndexconfig("Unknown boolean value '%s'." % value)


def ensure_list(data):
    if isinstance(data, (list, tuple, set)):
        return list(data)
    if not hasattr(data, "split"):
        raise InvalidIndexconfig("Unknown list value %r." % data)
    # split and remove empty
    return list(filter(None, (x.strip() for x in data.split(","))))


class ACLList(list):
    # marker class for devpiserver_indexconfig_defaults
    pass


def ensure_acl_list(data):
    data = ensure_list(data)
    for index, name in enumerate(data):
        if name.upper() in (':ANONYMOUS:', ':AUTHENTICATED:'):
            data[index] = name.upper()
    return data


def normalize_whitelist_name(name):
    if name == '*':
        return name
    return normalize_name(name)


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
        cls = type(
            cls.__name__,
            (cls, BaseStageCustomizer),
            dict(cls.__dict__))
    cls.InvalidIndex = InvalidIndex
    cls.InvalidIndexconfig = InvalidIndexconfig
    cls.ReadonlyIndex = ReadonlyIndex
    return cls


name_char_blocklist_regexp = re.compile(
    r'[\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    r'\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
    r' !"#$%&\'()*+,/:;<=>?\[\\\\\]^`{|}~]')


def is_valid_name(name: str) -> bool:
    return not name_char_blocklist_regexp.search(name)


class InvalidUserconfig(Exception):
    def __init__(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = messages
        Exception.__init__(self, messages)


class User:
    keyfs: KeyFS[Schema]

    # ignored_keys are skipped on create and modify
    ignored_keys = frozenset(('indexes', 'username'))
    # info keys are updated on create and modify and input is ignored
    info_keys = frozenset(('created', 'modified'))
    hidden_keys = frozenset((
        "password", "pwhash", "pwsalt"))
    public_keys = frozenset((
        "custom_data", "description", "email", "title"))
    # allowed_keys can be modified
    allowed_keys = hidden_keys.union(public_keys)
    known_keys = allowed_keys.union(ignored_keys, info_keys)
    # modification update keys control which changed keys will trigger
    # an update of the modified key
    modification_update_keys = known_keys - info_keys.union(ignored_keys)
    # visible_keys are returned via json
    visible_keys = ignored_keys.union(info_keys, public_keys)

    def __init__(self, parent: RootModel, name: str) -> None:
        self.parent = parent
        self.keyfs = parent.keyfs
        self.xom = parent.xom
        self.name = name
        self.key = self.keyfs.schema.USER.locate(name)
        self.key_index = self.keyfs.schema.INDEX.search(user=name)

    def get_cleaned_config(self, **kwargs):
        result = {}
        for key in self.allowed_keys:
            if key not in kwargs:
                continue
            result[key] = kwargs[key]
        return result

    def validate_config(self, **kwargs):
        unknown_keys = set(kwargs) - self.known_keys
        if unknown_keys:
            raise InvalidUserconfig(
                "Unknown keys in user config: %s" % ", ".join(unknown_keys))

    def _modify(self, password=None, pwhash=None, **kwargs):
        self.validate_config(**kwargs)
        modified: dict[str, object] = {}
        with self.key.update() as userconfig:
            if password is not None or pwhash:
                self._setpassword(userconfig, password, pwhash=pwhash)
                modified["password"] = "*******"  # noqa: S105
                kwargs['pwsalt'] = None
            for key, value in kwargs.items():
                key = ensure_unicode(key)
                if value:
                    if userconfig.get(key, notset) != value:
                        userconfig[key] = value
                        if key in self.hidden_keys:
                            value = "*******"
                        modified[key] = value
                elif key in userconfig:
                    del userconfig[key]
                    modified[key] = None
            if self.modification_update_keys.intersection(modified):
                if "created" not in userconfig:
                    # old data will be set to epoch
                    modified["created"] = userconfig["created"] = "1970-01-01T00:00:00Z"
                if "created" not in kwargs:
                    # only set modified if not created at the same time
                    modified_ts = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
                    if modified_ts != userconfig["created"]:
                        modified["modified"] = userconfig["modified"] = modified_ts
            assert "indexes" not in userconfig
        return ["%s=%s" % (k, v) for k, v in sorted(modified.items())]

    def modify(self, **kwargs):
        kwargs = self.get_cleaned_config(**kwargs)
        modified = self._modify(**kwargs)
        threadlog.info("modified user %r: %s", self.name,
                       ", ".join(modified))

    def _setpassword(self, userconfig, password, pwhash=None):
        if pwhash:
            userconfig["pwhash"] = ensure_unicode(pwhash)
        else:
            userconfig["pwhash"] = hash_password(password)
        threadlog.info("setting password for user %r", self.name)

    def delete(self) -> None:
        # delete all projects on the index
        key_index = self.key_index.with_resolved_parent()
        for key in list(key_index.iter_ulidkeys()):
            stage = self.getstage(key.params["index"])
            assert stage is not None
            stage.delete()
        # delete the user information itself
        self.key.delete()
        self.parent.delete_user(self.name)

    def validate(self, password: str) -> bool:
        userconfig = self.key.get()
        if not userconfig:
            return False
        pwhash = userconfig.get("pwhash")
        if pwhash is None:
            return False
        salt = userconfig.get("pwsalt")
        valid, newhash = verify_and_update_password_hash(password, pwhash, salt)
        if valid:
            if newhash and self.keyfs.tx.write:
                self.modify(pwsalt=None, pwhash=newhash)
            return True
        return False

    def get(self, credentials=False):
        if not self.key.exists():
            return {}
        d = self.key.get_mutable()
        if not credentials:
            for key in list(d):
                if key not in self.visible_keys:
                    del d[key]
        d["username"] = self.name
        d["indexes"] = self.get_indexes()
        return d

    def get_indexes(self):
        key_index = self.key_index.with_resolved_parent()
        return {
            key.params["index"]: get_mutable_deepcopy(value)
            for key, value in key_index.iter_ulidkey_values()
        }

    def create_stage(self, index, type="stage", **kwargs):
        return self.parent.create_stage(self, index, type=type, **kwargs)

    @cached_property
    def MirrorStage(self):
        from .mirror import MirrorStage
        return MirrorStage

    def _getstage(self, indexname, index_type, ixconfig):
        if index_type == "mirror":
            cls = self.MirrorStage
        else:
            cls = PrivateStage
        customizer_cls = get_stage_customizer_class(self.xom, index_type)
        return cls(
            self.xom,
            username=self.name, index=indexname,
            ixconfig=ixconfig,
            customizer_cls=customizer_cls)

    def getstage(self, indexname: str) -> BaseStage | None:
        return self.parent.getstage(self.name, indexname)

    def getstages(self) -> list[BaseStage]:
        stages = []
        for index in self.get()["indexes"]:
            stage = self.getstage(index)
            assert stage is not None
            stages.append(stage)
        return stages


class InvalidIndexconfig(Exception):
    def __init__(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = messages
        Exception.__init__(self, messages)


def get_principals(value):
    principals = set(value)
    if ':AUTHENTICATED:' in principals:
        principals.remove(':AUTHENTICATED:')
        principals.add(Authenticated)
    if ':ANONYMOUS:' in principals:
        principals.remove(':ANONYMOUS:')
        principals.add(Everyone)
    return principals


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
            ixconfig=self.stage.ixconfig)
        if principals is None:
            principals = {':ANONYMOUS:'}
        else:
            principals = set(principals)
        # admins should always be able to read the packages
        if restrict_modify is None:
            principals.add("root")
        else:
            principals.update(restrict_modify)
        return principals

    def get_principals_for_upload(self, restrict_modify=None):
        return self.stage.ixconfig.get("acl_upload", [])

    def get_principals_for_toxresult_upload(self, restrict_modify=None):
        return self.stage.ixconfig.get("acl_toxresult_upload", [':ANONYMOUS:'])

    def get_principals_for_index_delete(self, restrict_modify=None):
        if restrict_modify is None:
            modify_principals = set(['root', self.stage.username])
        else:
            modify_principals = restrict_modify
        return modify_principals

    get_principals_for_index_modify = get_principals_for_index_delete

    def get_principals_for_del_entry(self, restrict_modify=None):
        modify_principals = set(self.stage.ixconfig.get("acl_upload", []))
        if restrict_modify is None:
            modify_principals.update(['root', self.stage.username])
        else:
            modify_principals.update(restrict_modify)
        return modify_principals

    get_principals_for_del_project = get_principals_for_del_entry
    get_principals_for_del_verdata = get_principals_for_del_entry

    def get_possible_indexconfig_keys(self):
        """ Returns all possible custom index config keys.

        These are in addition to the existing keys of a regular private index.
        """
        return ()

    def get_default_config_items(self):
        """ Returns a list of defaults as key/value tuples.

        Only applies to new keys, not existing options of a private index.
        """
        return ()

    def normalize_indexconfig_value(self, key, value):
        """ Returns value normalized to the type stored in the database.

            A return value of None is treated as an error.
            Can raise InvalidIndexconfig.
            Will only be called for custom options, not for existing options
            of a private index.
            """

    def validate_config(self, oldconfig, newconfig):
        """ Validates the index config.

            Can raise InvalidIndexconfig."""

    def on_modified(self, request, oldconfig):
        """ Called after index was created or modified via a request.

            Can do further changes in the current transaction.

            Must use request.apifatal method to indicate errors instead
            of raising HTTPException responses.

            Other exceptions will be handled."""

    def get_projects_filter_iter(self, projects):
        """ Called when a list of projects is returned.

            Returns None for no filtering, or an iterator returning
            True for items to keep and False for items to remove."""
        return

    def get_versions_filter_iter(self, project, versions):
        """ Called when a list of versions is returned.

            Returns None for no filtering, or an iterator returning
            True for items to keep and False for items to remove."""
        return

    def get_simple_links_filter_iter(self, project, links):
        """ Called when a list of simple links is returned.

            Returns None for no filtering, or an iterator returning
            True for items to keep and False for items to remove.
            The size of the tuples in links might grow, develop defensively."""
        return


class UnknownCustomizer(BaseStageCustomizer):
    readonly = True

    # prevent uploads and deletions besides complete index removal
    def get_principals_for_index_modify(self, restrict_modify=None):
        return []

    get_principals_for_upload = get_principals_for_index_modify
    get_principals_for_toxresult_upload = get_principals_for_index_modify
    get_principals_for_del_entry = get_principals_for_index_modify
    get_principals_for_del_project = get_principals_for_index_modify
    get_principals_for_del_verdata = get_principals_for_index_modify


@total_ordering
class SimpleLinks:
    __slots__ = ('_links', 'stale')
    _links: list[SimplelinkMeta]
    stale: bool

    def __init__(
        self, links: Sequence[JoinedLink] | SimpleLinks, *, stale: bool = False
    ) -> None:
        assert links is not None
        if isinstance(links, SimpleLinks):
            self._links = links._links
            self.stale = links.stale or stale
        else:
            self._links = [SimplelinkMeta(x) for x in links]
            self.stale = stale

    def __hash__(self):
        return hash((self._links, self.stale))

    def __iter__(self):
        return self._links.__iter__()

    def __len__(self) -> int:
        return len(self._links)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            other = other._links
        return self._links == other

    def __lt__(self, other):
        if isinstance(other, type(self)):
            other = other._links
        return self._links < other

    def append(self, item):
        self._links.append(item)

    def sort(self, *args, **kw):
        self._links.sort(*args, **kw)

    def __repr__(self) -> str:
        clsname = f"{self.__class__.__module__}.{self.__class__.__name__}"
        content = ', '.join(repr(x) for x in self._links)
        return f"<{clsname} stale={self.stale!r} [{content}]>"


class check_upstream_error:
    def __init__(self, current: BaseStage, other: BaseStage) -> None:
        self.current = current
        self.other = other
        self.failed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        cls: type[BaseException] | None,
        val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if not isinstance(val, UpstreamError):
            return False
        if self.other is self.current:
            # If we are currently checking ourself raise the error, it is fatal
            return False
        threadlog.warn(
            "Failed to check mirror whitelist. Assume it does not exist (%s)", val
        )
        self.failed = True
        return True


class SkipReason(enum.StrEnum):
    InheritanceCycle = enum.auto()
    Missing = enum.auto()
    SROSkipHook = enum.auto()


@define(frozen=True, kw_only=True)
class TraversalInfo(ABC):
    name: str


@define(frozen=True, kw_only=True)
class PostponedTraversal(TraversalInfo):
    pass


@define(frozen=True, kw_only=True)
class TraversedStage(TraversalInfo):
    seen: bool
    stage: BaseStage


@define(frozen=True, kw_only=True)
class UntrustedTraversal(TraversedStage):
    pass


@define(frozen=True, kw_only=True)
class SkippedTraversal(TraversalInfo):
    reason: SkipReason
    src: str


class IndexBases:
    def __init__(
        self, stage: BaseStage, *, devpiserver_sro_skip: Callable, model: RootModel
    ) -> None:
        self.devpiserver_sro_skip = devpiserver_sro_skip
        self.model = model
        self.stage = stage

    def __iter__(self) -> Iterator[str]:
        return iter(self.bases)

    def __reversed__(self) -> Iterator[str]:
        return iter(reversed(self.bases))

    @cached_property
    def bases(self) -> tuple[str]:
        """Returns bases as tuple of strings."""
        return tuple(self.stage.ixconfig.get("bases", ()))

    def iter_stages(self) -> Iterator[BaseStage]:
        """Iterates stages in defined order without loops."""
        for traversal_info in self.traversal_infos:
            match traversal_info:
                case (
                    PostponedTraversal()
                    | SkippedTraversal(reason=SkipReason.InheritanceCycle)
                    | TraversedStage(seen=True)
                ):
                    continue
                case SkippedTraversal(name=name, reason=SkipReason.Missing, src=src):
                    threadlog.warn(
                        "Index %s refers to non-existing base %s.", src, name
                    )
                    continue
                case SkippedTraversal(
                    name=name, reason=SkipReason.SROSkipHook, src=src
                ):
                    threadlog.warn(
                        "Index %s base %s excluded via devpiserver_sro_skip.",
                        src,
                        name,
                    )
                    continue
                case TraversedStage(seen=False, stage=stage):
                    yield stage
                case _:
                    raise RuntimeError(traversal_info)

    def is_untrusted(self, stage: BaseStage) -> bool:
        # we have to postpone mirrors, as there
        # may be private releases in other paths
        return stage.index_type == "mirror"

    @cached_property
    def traversal_infos(self) -> list[TraversalInfo]:
        """Returns traversal information."""
        devpiserver_sro_skip = self.devpiserver_sro_skip
        getstage = self.model.getstage
        info: list[TraversalInfo] = []
        postponed: list[tuple[BaseStage, list[str]]] = []
        seen = set()
        is_untrusted = self.is_untrusted
        stage = self.stage
        todo = [(stage, list(reversed(stage.index_bases)))]
        while todo:
            (current_stage, bases) = todo[-1]
            current_name = current_stage.name
            if bases or current_name not in seen:
                info.append(
                    (
                        UntrustedTraversal
                        if is_untrusted(current_stage)
                        else TraversedStage
                    )(
                        name=current_name,
                        seen=current_name in seen,
                        stage=current_stage,
                    )
                )
            seen.add(current_name)
            if not bases:
                todo.pop()
                if not todo:
                    todo.extend(postponed)
                    postponed.clear()
                continue
            next_name = bases.pop()
            if next_name in seen:
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=SkipReason.InheritanceCycle,
                        src=current_name,
                    )
                )
                continue
            next_stage = getstage(next_name)
            if next_stage is None:
                info.append(
                    SkippedTraversal(
                        name=next_name, reason=SkipReason.Missing, src=current_name
                    )
                )
                seen.add(next_name)
                continue
            if devpiserver_sro_skip(stage=stage, base_stage=next_stage):
                info.append(
                    SkippedTraversal(
                        name=next_name, reason=SkipReason.SROSkipHook, src=current_name
                    )
                )
                seen.add(next_name)
                continue
            if is_untrusted(next_stage):
                info.append(PostponedTraversal(name=next_name))
                postponed.append((next_stage, list(reversed(next_stage.index_bases))))
            else:
                todo.append((next_stage, list(reversed(next_stage.index_bases))))
        return info

    @cached_property
    def whitelist_merger(self) -> Callable[[set[str], set[str]], set[str]]:
        def whitelist_intersection_merger(
            whitelist: set[str], stage_whitelist: set[str]
        ) -> set[str]:
            return whitelist.intersection(stage_whitelist)

        def whitelist_union_merger(
            whitelist: set[str], stage_whitelist: set[str]
        ) -> set[str]:
            return whitelist.union(stage_whitelist)

        whitelist_inheritance = self.stage.get_whitelist_inheritance()
        if whitelist_inheritance == "intersection":
            return whitelist_intersection_merger
        if whitelist_inheritance == "union":
            return whitelist_union_merger
        msg = f"Unknown whitelist_inheritance setting {whitelist_inheritance!r}"
        raise RuntimeError(msg)


class BaseStage:
    keyfs: KeyFS[Schema]
    offline: bool

    def __init__(
        self,
        xom: XOM,
        username: str,
        index: str,
        ixconfig: DictViewReadonly[str, Any],
        customizer_cls: type,
    ) -> None:
        self.xom = xom
        self.username = username
        self.index = index
        self.name = username + "/" + index
        self.ixconfig = ixconfig
        self.customizer = customizer_cls(self)
        # the following attributes are per-xom singletons
        self.keyfs = xom.keyfs
        self.filestore = xom.filestore
        self.key_index = self.keyfs.schema.INDEX.locate(user=username, index=index)
        self.key_project = self.keyfs.schema.PROJECT.search(user=username, index=index)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"

    @lazy
    def index_bases(self) -> IndexBases:
        return self.get_index_bases()

    @cached_property
    def index_type(self):
        return self.ixconfig["type"]

    @overload
    def key_simpledata(
        self, project: NormalizedName | str, version_filename: tuple[str, str]
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_simpledata(
        self, project: NormalizedName | str, version_filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_simpledata(
        self,
        project: NormalizedName | str,
        version_filename: tuple[str, str] | None = None,
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.SIMPLEDATA
        (version, filename) = (
            (None, None) if version_filename is None else version_filename
        )
        (kw, meth) = (
            ({}, key.search)
            if version is None or filename is None
            else (dict(version=version, filename=filename), key.locate)
        )
        return meth(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    @overload
    def key_version(
        self, project: NormalizedName | str, version: str
    ) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_version(
        self, project: NormalizedName | str, version: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_version(
        self, project: NormalizedName | str, version: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        key = self.keyfs.schema.VERSION
        (kw, meth) = (
            ({}, key.search) if version is None else (dict(version=version), key.locate)
        )
        return meth(
            user=self.username, index=self.index, project=normalize_name(project), **kw
        )

    @property
    def model(self) -> RootModel:
        return self.xom.model

    @abstractmethod
    def add_project_name(self, project: NormalizedName | str) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def no_project_list(self) -> bool:
        raise NotImplementedError

    def get_indexconfig_from_kwargs(self, **kwargs):
        """Normalizes values and validates keys.

        Returns the parts touched by kwargs as dict.
        This is not the complete index configuration."""
        index_type = self.index_type
        ixconfig = {}
        # get known keys and validate them
        stage_keys = set(self.get_possible_indexconfig_keys())
        customizer_keys = set(self.customizer.get_possible_indexconfig_keys())
        conflicting = stage_keys.intersection(customizer_keys)
        if conflicting:
            raise ValueError(
                "The stage customizer for '%s' defines keys which conflict "
                "with existing index configuration keys: %s"
                % (index_type, ", ".join(sorted(conflicting))))
        # prevent default values from being removed
        for key, value in self.get_default_config_items():
            if kwargs.get(key) is RemoveValue:
                raise InvalidIndexconfig("Default values can't be removed.")
        # now process any key known by the stage class
        for key in stage_keys:
            if key not in kwargs:
                continue
            value = kwargs.pop(key)
            if value is not RemoveValue:
                value = self.normalize_indexconfig_value(key, value)
                if value is None:
                    raise ValueError(
                        "The key '%s' wasn't processed."
                        % (key))
            ixconfig[key] = value
        # prevent removal of defaults from the customizer class
        for key, value in self.customizer.get_default_config_items():
            if kwargs.get(key) is RemoveValue:
                raise InvalidIndexconfig("Default values can't be removed.")
        # and process any key known by the customizer class
        for key in customizer_keys:
            if key not in kwargs:
                continue
            value = kwargs.pop(key)
            if value is not RemoveValue:
                value = self.customizer.normalize_indexconfig_value(key, value)
                if value is None:
                    raise ValueError(
                        "The key '%s' wasn't processed."
                        % (key))
            ixconfig[key] = value
        # lastly we get additional default from the hook
        hooks = self.xom.config.hook
        for defaults in hooks.devpiserver_indexconfig_defaults(index_type=index_type):
            conflicting = stage_keys.intersection(defaults)
            if conflicting:
                raise ValueError(
                    "A plugin returned the following keys which conflict with "
                    "existing index configuration keys for '%s': %s"
                    % (index_type, ", ".join(sorted(conflicting))))
            for key, value in defaults.items():
                new_value = kwargs.pop(key, value)
                if new_value is not RemoveValue:
                    if isinstance(value, bool):
                        new_value = ensure_boolean(new_value)
                    elif isinstance(value, ACLList):
                        new_value = ensure_acl_list(new_value)
                    elif isinstance(value, (list, tuple, set)):
                        new_value = ensure_list(new_value)
                ixconfig.setdefault(key, new_value)
        for key, value in list(kwargs.items()):
            if value is RemoveValue:
                ixconfig[key] = kwargs.pop(key)
        ixconfig["type"] = index_type
        return (ixconfig, kwargs)

    @abstractmethod
    def get_default_config_items(self) -> Sequence[tuple[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_possible_indexconfig_keys(self) -> Sequence:
        raise NotImplementedError

    @abstractmethod
    def get_versiondata_perstage(
        self,
        project: NormalizedName | str,
        version: str,
        *,
        with_elinks: bool = True,
    ) -> DictViewReadonly[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_projects_perstage(self) -> dict[str, NormalizedName | str]:
        raise NotImplementedError

    @abstractmethod
    def has_project_perstage(self, project: NormalizedName | str) -> bool | Unknown:
        raise NotImplementedError

    @abstractmethod
    def has_version_perstage(self, project: str, version: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_elink_from_entry(self, entry: BaseFileEntry) -> ELink | None:
        raise NotImplementedError

    @abstractmethod
    def normalize_indexconfig_value(self, key: str, value: Any) -> Any:
        raise NotImplementedError

    @cached_property
    def user(self):
        # only few methods need the user object.
        return self.model.get_user(self.username)

    @property
    def ixconfig_mutable(self) -> dict:
        return self.key_index.with_resolved_parent().get_mutable()

    def delete(self) -> None:
        self.model.delete_stage(self.username, self.index)

    def get_releaselinks(self, project):
        # compatibility access method used by devpi-web and tests
        project = normalize_name(project)
        try:
            return [self._make_elink(project, link_info)
                    for link_info in self.get_simplelinks(project)]
        except self.UpstreamNotFoundError:
            return []

    def get_releaselinks_perstage(self, project: NormalizedName | str) -> list[ELink]:
        # compatibility access method for devpi-findlinks and possibly other plugins
        project = normalize_name(project)
        return [self._make_elink(project, link_info)
                for link_info in self.get_simplelinks_perstage(project)]

    def _make_elink(self, project, link_meta):
        return ELink(
            self.filestore,
            dict(
                relpath=link_meta.relpath,
                hashes=link_meta.hashes,
                rel=Rel.ReleaseFile,
                require_python=link_meta.require_python,
                yanked=link_meta.yanked,
            ),
            link_meta.user,
            link_meta.index,
            project,
            link_meta.version,
        )

    def get_linkstore_perstage(self, name, version):
        return LinkStore(self, name, version)

    def get_mutable_linkstore_perstage(self, name, version):
        if self.customizer.readonly:
            threadlog.warn("index is marked read only")
        return MutableLinkStore(self, name, version)

    def get_keys_for_entrypaths(
        self, entrypaths: Iterable[str]
    ) -> list[LocatedKey | None]:
        return [
            self.keyfs.match_key(
                entrypath.rsplit("#", 1)[0],
                self.keyfs.schema.FILE_NOHASH,
                self.keyfs.schema.FILE,
            ).with_resolved_parent()
            for entrypath in entrypaths
        ]

    def get_entries_for_keys(
        self, keys: Iterable[LocatedKey | None]
    ) -> list[FileEntry | None]:
        key_to_ulidkey: dict[LocatedKey, ULIDKey | Absent | Deleted] = dict(
            self.keyfs.tx.resolve_keys(
                (k for k in keys if k is not None),
                fetch=True,
                fill_cache=True,
                new_for_missing=False,
            )
        )
        return [
            None
            if key is None
            or (ulid_key := key_to_ulidkey.get(key)) is None
            or isinstance(ulid_key, (Absent, Deleted))
            else FileEntry(ulid_key)
            for key in keys
        ]

    def get_entries_for_entrypaths(
        self, entrypaths: Iterable[str]
    ) -> list[FileEntry | None]:
        keys = self.get_keys_for_entrypaths(entrypaths)
        return self.get_entries_for_keys(keys)

    def get_link_from_entrypath(self, entrypath):
        relpath = entrypath.rsplit("#", 1)[0]
        entry = self.xom.filestore.get_file_entry(relpath)
        if entry is None or entry.project is None:
            return None
        linkstore = self.get_linkstore_perstage(entry.project,
                                                entry.version)
        links = linkstore.get_links(entrypath=entrypath)
        assert len(links) < 2
        return links[0] if links else None

    @abstractmethod
    def get_simplelinks_perstage(self, project: NormalizedName | str) -> SimpleLinks:
        raise NotImplementedError

    def store_toxresult(
        self,
        link: ELink,
        content_or_file: ContentOrFile,
        *,
        filename: str | None = None,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        if self.customizer.readonly:
            raise ReadonlyIndex("index is marked read only")
        assert not isinstance(content_or_file, dict)
        linkstore = self.get_mutable_linkstore_perstage(link.project, link.version)
        return linkstore.new_reflink(
            rel=Rel.ToxResult,
            content_or_file=content_or_file,
            for_link=link,
            filename=filename,
            hashes=hashes,
            last_modified=last_modified,
        )

    def get_toxresults(self, link):
        l = []
        linkstore = self.get_linkstore_perstage(link.project, link.version)
        for reflink in linkstore.get_links(rel=Rel.ToxResult, for_entrypath=link):
            with reflink.entry.file_open_read() as f:
                l.append(json.load(f))
        return l

    def filter_versions(
        self, project: NormalizedName | str, versions: set[str]
    ) -> set[str]:
        iterator = self.customizer.get_versions_filter_iter(project, versions)
        if iterator is None:
            return versions
        return set(apply_filter_iter(versions, iterator))

    def list_versions(self, project: NormalizedName | str) -> set[str]:
        project = normalize_name(project)
        versions = set()
        for stage in self.get_mergeable_stages(project, "list_versions"):
            with check_upstream_error(self, stage) as checker:
                res = stage.list_versions_perstage(project)
            if checker.failed:
                continue
            versions.update(res)
        return self.filter_versions(project, versions)

    @abstractmethod
    def list_versions_perstage(self, project: str) -> set:
        raise NotImplementedError

    def get_latest_version(self, name, stable=False):
        return get_latest_version(
            self.filter_versions(
                name, self.list_versions(name)),
            stable=stable)

    def get_latest_version_perstage(self, name, stable=False):
        return get_latest_version(
            self.filter_versions(
                name, self.list_versions_perstage(name)),
            stable=stable)

    def get_versiondata(
        self, project: NormalizedName | str, version: str
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if not self.filter_versions(project, {version}):
            return result
        for stage in self.get_mergeable_stages(
            normalize_name(project), "get_versiondata"
        ):
            with check_upstream_error(self, stage) as checker:
                res = stage.get_versiondata_perstage(project, version)
            if checker.failed:
                continue
            if res:
                if not result:
                    result.update(res)
                else:
                    l = result.setdefault("+shadowing", [])
                    l.append(res)
        return result

    def get_simplelinks(
        self, project: NormalizedName | str, *, sorted_links: bool = True
    ) -> SimpleLinks:
        """ Return list of (key, href) tuples where "href" is a path
        to a file entry with "#" appended hash-specs or egg-ids
        and "key" is usually the basename of the link or else
        the egg-ID if the link points to an egg.
        """
        project = normalize_name(project)
        all_links = self.SimpleLinks([])
        seen = set()

        try:
            for stage in self.get_mergeable_stages(project, "get_simplelinks"):
                with check_upstream_error(self, stage) as checker:
                    res = stage.get_simplelinks_perstage(project)
                if checker.failed:
                    continue
                if res is not None:
                    res = self.SimpleLinks(res)
                    all_links.stale = all_links.stale or res.stale
                iterator = self.customizer.get_simple_links_filter_iter(project, res)
                if iterator is not None:
                    res = apply_filter_iter(res, iterator)
                for link_info in res:
                    key = link_info.key
                    if key not in seen:
                        seen.add(key)
                        all_links.append(link_info)
        except self.UpstreamNotFoundError:
            return self.SimpleLinks([])

        if sorted_links:
            all_links.sort(reverse=True)
        return all_links

    def get_whitelist_inheritance(self) -> str:
        # the default value, if the setting is missing, is the old behavior,
        # so existing indexes work as before
        return self.ixconfig.get("mirror_whitelist_inheritance", "union")

    def get_mirror_whitelist_info(
        self, project: NormalizedName | str
    ) -> dict[str, Any]:
        bases = self.index_bases
        project = ensure_unicode(project)
        private_hit: bool | Unknown = False
        whitelisted = False
        whitelist_merger = bases.whitelist_merger
        whitelist = None
        for stage in bases.iter_stages():
            if stage.index_type == "mirror":
                if private_hit and not whitelisted:
                    # don't check the mirror for private packages
                    return dict(
                        has_mirror_base=unknown, blocked_by_mirror_whitelist=stage.name
                    )
                in_index = stage.has_project_perstage(project)
                if in_index is unknown:
                    return dict(
                        has_mirror_base=unknown, blocked_by_mirror_whitelist=None
                    )
                has_mirror_base = in_index and (not private_hit or whitelisted)
                blocked_by_mirror_whitelist = in_index and private_hit and not whitelisted
                return dict(
                    has_mirror_base=has_mirror_base,
                    blocked_by_mirror_whitelist=stage.name if blocked_by_mirror_whitelist else None)
            else:
                in_index = stage.has_project_perstage(project)
            private_hit = private_hit or in_index
            stage_whitelist = set(
                stage.ixconfig.get("mirror_whitelist", set()))
            whitelist = (
                stage_whitelist
                if whitelist is None
                else whitelist_merger(whitelist, stage_whitelist)
            )
            if whitelisted or whitelist.intersection(('*', project)):
                whitelisted = True
        return dict(
            has_mirror_base=False,
            blocked_by_mirror_whitelist=None)

    def has_mirror_base(self, project):
        return self.get_mirror_whitelist_info(project)['has_mirror_base']

    def filter_projects(self, projects):
        iterator = self.customizer.get_projects_filter_iter(projects)
        if iterator is None:
            return projects
        return frozenset(apply_filter_iter(projects, iterator))

    def has_project(self, project: NormalizedName | str) -> bool | Unknown:
        if not self.filter_projects([project]):
            return False
        for stage in self.sro():
            res = stage.has_project_perstage(project)
            if res is unknown:
                return res
            if res:
                return True
        return False

    def list_projects(self) -> list[tuple[BaseStage, dict[str, NormalizedName | str]]]:
        result = []
        for stage in self.sro():
            projects = stage.list_projects_perstage()
            result.append((
                stage,
                self.filter_projects(projects)))
        return result

    def _modify(self, **kw):
        if "type" in kw and self.index_type != kw["type"]:
            raise InvalidIndexconfig(
                ["the 'type' of an index can't be changed"])
        kw.pop('type', None)
        kw.pop("projects", None)  # we never modify this from the outside
        keep_unknown = kw.pop("_keep_unknown", False)
        (ixconfig, unknown) = self.get_indexconfig_from_kwargs(**kw)
        if unknown:
            if keep_unknown:
                # used to import data when plugins aren't installed anymore
                ixconfig.update(unknown)
            else:
                raise InvalidIndexconfig(
                    ["indexconfig got unexpected keyword arguments: %s"
                     % ", ".join("%s=%s" % x for x in unknown.items())])
        # modify indexconfig
        key_index = self.key_index.with_resolved_parent()
        with key_index.update() as newconfig:
            oldconfig = dict(self.ixconfig)
            for key, value in list(ixconfig.items()):
                if value is RemoveValue:
                    newconfig.pop(key, None)
                    ixconfig.pop(key)
            newconfig.update(ixconfig)
            self.customizer.validate_config(oldconfig, newconfig)
            self.ixconfig = ensure_deeply_readonly(newconfig)
            return newconfig

    def modify(self, index=None, **kw):
        lazy.invalidate(self, "index_bases")
        newconfig = self._modify(**kw)
        threadlog.info("modified index %s: %s", self.name, newconfig)
        return newconfig

    def get_index_bases(self) -> IndexBases:
        return IndexBases(
            self,
            devpiserver_sro_skip=self.xom.config.hook.devpiserver_sro_skip,
            model=self.model,
        )

    def get_mergeable_stages(
        self, project: NormalizedName, opname: str
    ) -> Iterable[BaseStage]:
        if not self.filter_projects([project]):
            return
        bases = self.index_bases
        whitelisted: BaseStage | Literal[False] = False
        private_hit = False
        whitelist_merger = bases.whitelist_merger
        whitelist = None
        for stage in bases.iter_stages():
            if stage.index_type == "mirror":
                if private_hit:
                    if not whitelisted:
                        threadlog.debug("%s: private package %r not whitelisted, "
                                        "ignoring %s", opname, project, stage.name)
                        continue
                    threadlog.debug("private package %r whitelisted at stage %s",
                                    project, whitelisted.name)
            else:
                stage_whitelist = set(
                    stage.ixconfig.get("mirror_whitelist", set()))
                whitelist = (
                    stage_whitelist
                    if whitelist is None
                    else whitelist_merger(whitelist, stage_whitelist)
                )
                if whitelist.intersection(('*', project)):
                    whitelisted = stage
                elif stage.has_project_perstage(project):
                    private_hit = True

            with check_upstream_error(self, stage):
                exists = stage.has_project_perstage(project)
                if not private_hit and exists is unknown and stage.no_project_list:
                    # direct fetching is allowed
                    pass
                elif not exists:
                    continue
                yield stage

    def op_sro(self, opname, **kw):
        warnings.warn(
            "The 'op_sro' method is deprecated, use 'index_bases.iter_stages()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "project" in kw:
            project = normalize_name(kw["project"])
            if not self.filter_projects([project]):
                return
        for stage in self.sro():
            yield stage, getattr(stage, opname)(**kw)

    def op_sro_check_mirror_whitelist(self, opname, **kw):
        warnings.warn(
            "The 'op_sro_check_mirror_whitelist' method is deprecated, "
            "use 'get_mergeable_stages' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        project = normalize_name(kw["project"])
        if not self.filter_projects([project]):
            return
        for stage in self.get_mergeable_stages(project, opname):
            with check_upstream_error(self, stage) as checker:
                res = getattr(stage, opname)(**kw)
            if checker.failed:
                continue
            yield stage, res

    def sro(self) -> Iterator[BaseStage]:
        """ return stage resolution order. """
        return self.index_bases.iter_stages()

    def __acl__(self):
        permissions = (
            'pkg_read',
            'toxresult_upload',
            'upload',
            'index_delete',
            'index_modify',
            'del_entry',
            'del_project',
            'del_verdata')
        restrict_modify = self.xom.config.restrict_modify
        acl = []
        for permission in permissions:
            method_name = 'get_principals_for_%s' % permission
            method = getattr(self.customizer, method_name, None)
            if not callable(method):
                msg = f"The attribute {method_name} with value {method!r} of {self.customizer!r} is not callable."
                raise AttributeError(msg)  # noqa: TRY004
            for principal in get_principals(method(restrict_modify=restrict_modify)):
                acl.append((Allow, principal, permission))
                if permission == 'upload':
                    # add pypi_submit alias for BBB
                    acl.append((Allow, principal, 'pypi_submit'))
        return acl

    InvalidIndex = InvalidIndex
    InvalidIndexconfig = InvalidIndexconfig
    InvalidUser = InvalidUser
    NotFound = NotFound
    UpstreamError = UpstreamError
    UpstreamNotFoundError = UpstreamNotFoundError
    UpstreamNotModified = UpstreamNotModified
    MissesRegistration = MissesRegistration
    MissesVersion = MissesVersion
    NonVolatile = NonVolatile
    SimpleLinks = SimpleLinks


class PrivateStage(BaseStage):
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

    def get_possible_indexconfig_keys(self):
        return tuple(dict(self.get_default_config_items())) + (
            "custom_data", "description", "title")

    def get_default_config_items(self):
        return [
            ("volatile", True),
            ("acl_upload", [self.username]),
            ("acl_toxresult_upload", [":ANONYMOUS:"]),
            ("bases", ()),
            ("mirror_whitelist", []),
            ("mirror_whitelist_inheritance", "intersection")]

    def normalize_indexconfig_value(self, key, value):
        if key == "volatile":
            return ensure_boolean(value)
        if key == "bases":
            return normalize_bases(
                self.xom.model, ensure_list(value))
        if key == "acl_upload":
            return ensure_acl_list(value)
        if key == "acl_toxresult_upload":
            return ensure_acl_list(value)
        if key == "mirror_whitelist":
            return [
                normalize_whitelist_name(x)
                for x in ensure_list(value)]
        if key == "mirror_whitelist_inheritance":
            value = value.lower()
            if value not in ("intersection", "union"):
                raise InvalidIndexconfig(
                    "Unknown value '%s' for mirror_whitelist_inheritance, "
                    "must be 'intersection' or 'union'." % value)
            return value
        if key in ("custom_data", "description", "title"):
            return value

    def delete(self) -> None:
        # delete all projects on this index
        for name in self.list_projects_perstage():
            self.del_project(name)
        BaseStage.delete(self)

    #
    # registering project and version metadata
    #

    def set_versiondata(self, metadata):
        """ register metadata.  Raises ValueError in case of metadata
        errors. """
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

    def del_project(self, project: NormalizedName | str) -> None:
        project = normalize_name(project)
        key_version = self.key_version(project).with_resolved_parent()
        versions = {x.name for x in key_version.iter_ulidkeys()}
        for version in versions:
            self.del_versiondata(project, version, cleanup=False)
        threadlog.info("deleting project %s", project)
        self.key_project(project).with_resolved_parent().delete()

    def del_versiondata(
        self, project: NormalizedName | str, version: str, *, cleanup: bool = True
    ) -> None:
        project = normalize_name(project)
        if not self.has_project_perstage(project):
            raise self.NotFound("project %r not found on stage %r" %
                                (project, self.name))
        if not self.key_versionmetadata(project, version).exists(resolve_parents=True):
            raise self.NotFound("version %r of project %r not found on stage %r" %
                                (version, project, self.name))
        linkstore = self.get_mutable_linkstore_perstage(project, version)
        linkstore.remove_links()
        key_versionmetadata = self.key_versionmetadata(
            project, version
        ).with_resolved_parent()
        self.key_version(project, version).with_resolved_parent().delete()
        metadata = key_versionmetadata.get()
        if "description" in metadata and not isinstance(metadata["description"], str):
            entry = self.filestore.get_file_entry(
                RelPath(
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
        return ELink.from_entry(self.filestore, entry, data)

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
                RelPath(
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
                    (k.params["filename"], href, v.get("requires_python"), None)
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
            if '_' in version:
                version = version.replace('_', '-')
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
                    "doczip has no version and '%s' has no releases to "
                    "derive one from", project)
            threadlog.info("store_doczip: derived version of %s is %s",
                           project, version)
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
        """ get link of documentation zip or None if no docs exists. """
        doczip = self.key_doczip(project, version).with_resolved_parent().get()
        if not doczip:
            return None
        return ELink(
            self.filestore,
            dict(doczip, rel=Rel.DocZip),
            self.username,
            self.index,
            project,
            version,
        )

    def get_doczip_entry(self, project, version):
        """ get entry of documentation zip or None if no docs exists. """
        link = self.get_doczip_link(project, version)
        return link.entry if link else None

    def get_doczip(self, project, version):
        """ get documentation zip content or None if no docs exists. """
        link = self.get_doczip_link(project, version)
        return link.entry.file_get_content() if link else None

    def get_last_change_serial_perstage(self, at_serial=None):
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


class StageCustomizer(BaseStageCustomizer):
    pass


@hookimpl
def devpiserver_get_stage_customizer_classes():
    # prevent plugins from installing their own under the reserved names
    return [
        ("stage", StageCustomizer)]


def linkdictprop(name, default=notset):
    def fget(self):
        try:
            return self.linkdict[name]
        except KeyError:
            if default is notset:
                raise AttributeError(name)
            return default

    return property(fget)


class ELink:
    """ model Link using entrypathes for referencing. """

    __slots__ = (
        "_basename",
        "_entry",
        "filestore",
        "index",
        "linkdict",
        "project",
        "user",
        "version",
    )

    _log: MutableSequence[FileLogEntry] = linkdictprop("log")
    index_relpath = linkdictprop("relpath")
    for_relpath = linkdictprop("for_relpath", default=None)
    rel = linkdictprop("rel", default=None)
    require_python = linkdictprop("require_python")
    yanked = linkdictprop("yanked")

    def __init__(self, filestore, linkdict, user, index, project, version):
        assert "hash_spec" not in linkdict
        self._entry = notset
        self.filestore = filestore
        self.linkdict = linkdict
        if self.for_relpath is not None:
            assert "#" not in self.for_relpath
        self.user = user
        self.index = index
        self.project = project
        self.version = version

    @classmethod
    def from_entry(cls, filestore, entry, linkdict):
        elink = ELink(
            filestore, linkdict, entry.user, entry.index, entry.project, entry.version
        )
        elink._entry = entry
        return elink

    @property
    def best_available_hash_type(self):
        return self.hashes.best_available_type

    @property
    def best_available_hash_spec(self):
        return self.hashes.best_available_spec

    @property
    def best_available_hash_value(self):
        return self.hashes.best_available_value

    @property
    def hashes(self):
        return self.entry.hashes

    @property
    def basename(self):
        _basename = getattr(self, "_basename", None)
        if _basename is None:
            _basename = self._basename = Path(self.relpath).name
        return _basename

    @property
    def entrypath(self):
        entrypath = self.relpath
        hash_spec = self.best_available_hash_spec
        if hash_spec:
            return f"{entrypath}#{hash_spec}"
        return entrypath

    def matches_hashes(self, hashes):
        return self.hashes == hashes

    @property
    def for_entrypath(self) -> RelPath:
        return RelPath(f"{self.user}/{self.index}/{self.for_relpath}")

    @property
    def relpath(self) -> RelPath:
        return RelPath(f"{self.user}/{self.index}/{self.index_relpath}")

    def __repr__(self) -> str:
        return "<ELink rel=%r entrypath=%r>" % (self.rel, self.entrypath)

    @property
    def entry(self) -> FileEntry:
        entry = self._entry
        if isinstance(entry, NotSet):
            entry = self.filestore.get_file_entry(self.relpath)
            assert entry is not None
            self._entry = entry
        return entry

    def add_log(
        self,
        what: str,
        who: str | None,
        *,
        count: int | None = None,
        dst: str | None = None,
        src: str | None = None,
        when: str | None = None,
    ) -> None:
        d = FileLogEntry(
            what=what, who=who, when=strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        )
        if count is not None:
            d["count"] = count
        if dst is not None:
            d["dst"] = dst
        if src is not None:
            d["src"] = src
        if when is not None:
            assert isinstance(when, str)
            d["when"] = when
        self._log.append(d)

    def add_logs(self, logs: Iterable[FileLogEntry | dict]) -> None:
        for log in logs:
            unknown_keys = set(log).difference(
                FileLogEntry.__required_keys__ | FileLogEntry.__optional_keys__
            )
            if unknown_keys:
                msg = f"Unknown keys {', '.join(sorted(unknown_keys))} for FileLogEntry"
                raise ValueError(msg)
            self.add_log(
                log["what"],
                log["who"],
                count=log.get("count"),
                dst=log.get("dst"),
                src=log.get("src"),
                when=log["when"],
            )

    def get_logs(self) -> list[FileLogEntry]:
        return list(getattr(self, '_log', []))


class LinkStore:
    filestore: FileStore

    def __init__(self, stage, project, version):
        self.stage = stage
        self.filestore = stage.filestore
        self.project = normalize_name(project)
        self.version = version
        if not self.stage.has_version_perstage(project, version):
            raise MissesRegistration(
                "%s-%s on stage %s at %s",
                project, version, stage.name, stage.keyfs.tx.at_serial)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.project} {self.stage.name} {self.version}>"

    def get_links(
        self,
        rel: Rel | None = None,
        basename: str | None = None,
        entrypath: str | None = None,
        for_entrypath: ELink | str | None = None,
    ) -> list[ELink]:
        if isinstance(for_entrypath, ELink):
            for_entrypath = for_entrypath.relpath
        elif for_entrypath is not None:
            assert "#" not in for_entrypath

        elinks = self.stage._get_elinks(self.project, self.version, rel=rel)

        def fil(link):
            return (
                (not rel or rel == link.rel)
                and (not basename or basename == link.basename)
                and (not entrypath or entrypath in (link.entrypath, link.relpath))
                and (not for_entrypath or for_entrypath == link.for_entrypath)
            )

        filestore = self.filestore
        username = self.stage.username
        index = self.stage.index
        project = self.project
        version = self.version
        return [
            elink
            for linkdict in elinks
            if fil(
                elink := ELink(filestore, linkdict, username, index, project, version)
            )
        ]

    @property
    def metadata(self):
        verdata = self.stage.get_versiondata_perstage(
            self.project, self.version, with_elinks=False
        )
        return ensure_deeply_readonly(
            {k: v for k, v in verdata.items() if not k.startswith("+")}
        )


class MutableLinkStore(LinkStore):
    def create_linked_entry(
        self,
        rel: Rel,
        basename: str,
        content_or_file: ContentOrFile,
        *,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        overwrite = None
        for link in self.get_links(rel=rel, basename=basename):
            if not self.stage.ixconfig.get("volatile"):
                exc = NonVolatile(
                    "rel=%s basename=%s on stage %s" % (rel, basename, self.stage.name)
                )
                exc.link = link
                raise exc
            assert overwrite is None
            overwrite = sum(
                x.get("count", 0)
                for x in link.get_logs()
                if x.get("what") == "overwrite"
            )
        self.remove_links(rel=rel, basename=basename)
        file_entry = self._create_file_entry(
            basename, content_or_file, hashes=hashes, last_modified=last_modified
        )
        link = self._add_link_to_file_entry(rel, file_entry)
        if overwrite is not None:
            link.add_log("overwrite", None, count=overwrite + 1)
        return link

    def key_doczip(self) -> LocatedKey[dict, DictViewReadonly]:
        return self.stage.key_doczip(self.project, self.version)

    @overload
    def key_simpledata(self, filename: str) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_simpledata(
        self, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_simpledata(
        self, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        if filename is None:
            return self.stage.key_simpledata(self.project)
        return self.stage.key_simpledata(self.project, (self.version, filename))

    @overload
    def key_toxresult(self, filename: str) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_toxresult(
        self, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_toxresult(
        self, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        return self.stage.key_toxresult(self.project, self.version, filename)

    @overload
    def key_versionfile(self, filename: str) -> LocatedKey[dict, DictViewReadonly]: ...

    @overload
    def key_versionfile(
        self, filename: None = None
    ) -> SearchKey[dict, DictViewReadonly]: ...

    def key_versionfile(
        self, filename: str | None = None
    ) -> LocatedKey[dict, DictViewReadonly] | SearchKey[dict, DictViewReadonly]:
        return self.stage.key_versionfile(self.project, self.version, filename)

    def new_reflink(
        self,
        rel: Rel,
        content_or_file: ContentOrFile,
        for_link: ELink,
        *,
        filename: str | None = None,
        hashes: Digests,
        last_modified: str | None = None,
    ) -> ELink:
        links = self.get_links(entrypath=for_link.relpath)
        assert len(links) == 1, f"need exactly one reference, got {links}"
        assert for_link.relpath == links[0].relpath
        base_entry = for_link.entry
        if filename is None:
            other_reflinks = self.get_links(rel=rel, for_entrypath=for_link.relpath)
            timestamp = strftime("%Y%m%d%H%M%S", gmtime())
            filename = "%s.%s-%s-%d" % (
                base_entry.basename,
                rel,
                timestamp,
                len(other_reflinks),
            )
        entry = self._create_file_entry(
            filename,
            content_or_file,
            hashes=hashes,
            last_modified=last_modified,
            ref_hash_spec=base_entry.ref_hash_spec,
        )
        return self._add_link_to_file_entry(rel, entry, for_link=for_link)

    def remove_links(self, rel=None, basename=None, for_entrypath=None):
        del_links = self.get_links(
            rel=rel, basename=basename, for_entrypath=for_entrypath
        )
        was_deleted = []
        key_doczip = self.key_doczip().with_resolved_parent()
        key_toxresult = self.key_toxresult().with_resolved_parent()
        key_versionfile = self.key_versionfile().with_resolved_parent()
        key_simpledata = self.key_simpledata().with_resolved_parent()
        version = self.version
        if del_links:
            for link in del_links:
                filename = link.entry.basename
                link.entry.delete()
                match link.rel:
                    case Rel.DocZip:
                        key_doczip.delete()
                    case Rel.ToxResult:
                        key_toxresult(filename).delete()
                    case Rel.ReleaseFile:
                        key_versionfile(filename).delete()
                        key_simpledata(version=version, filename=filename).delete()
                    case _:
                        raise RuntimeError(link.rel)
                was_deleted.append(link.relpath)
                threadlog.info("deleted %r link %s", link.rel, link.relpath)
        has_versionfiles = next(key_versionfile.iter_ulidkeys(), absent) is not absent
        if was_deleted and has_versionfiles:
            for relpath in was_deleted:
                self.remove_links(for_entrypath=relpath)

    def _create_file_entry(
        self,
        basename: str,
        content_or_file: ContentOrFile,
        *,
        hashes: Digests,
        last_modified: str | None = None,
        ref_hash_spec: str | None = None,
    ) -> MutableFileEntry:
        entry = self.filestore.store(
            user=self.stage.username,
            index=self.stage.index,
            basename=basename,
            content_or_file=content_or_file,
            last_modified=last_modified,
            ref_hash_spec=ref_hash_spec,
            hashes=hashes,
        )
        entry.project = self.project
        entry.version = self.version
        return entry

    def _add_link_to_file_entry(
        self, rel: Rel, file_entry: BaseFileEntry, for_link: ELink | str | None = None
    ) -> ELink:
        new_linkdict = {
            "relpath": file_entry.index_relpath,
            "log": [],
        }
        if for_link:
            assert isinstance(for_link, ELink)
            new_linkdict["for_relpath"] = for_link.index_relpath
        match rel:
            case Rel.DocZip:
                key = self.key_doczip().with_resolved_parent()
            case Rel.ToxResult:
                key = self.key_toxresult(file_entry.basename).with_resolved_parent()
            case Rel.ReleaseFile:
                key = self.key_versionfile(file_entry.basename).with_resolved_parent()
            case _:
                raise RuntimeError(rel)
        if key.exists():
            raise RuntimeError
        key.set(new_linkdict)
        threadlog.info("added %r link %s", rel, file_entry.relpath)
        stage = self.stage
        return ELink(
            self.filestore,
            new_linkdict,
            stage.username,
            stage.index,
            self.project,
            self.version,
        )


@total_ordering
class SimplelinkMeta:
    """ helper class to provide information for items from get_simplelinks() """

    __slots__ = (
        "__basename",
        "__cmpval",
        "__ext",
        "__hashes",
        "__index",
        "__name",
        "__relpath",
        "__url",
        "__user",
        "__version",
        "href",
        "key",
        "require_python",
        "yanked",
    )
    __cmpval: tuple | NotSet
    __hashes: Digests | NotSet
    __relpath: str | NotSet

    def __init__(self, link_info: tuple[str, str, RequiresPython, Yanked]) -> None:
        self.__basename = notset
        self.__cmpval = notset
        self.__ext = notset
        self.__hashes = notset
        self.__index = notset
        self.__name = notset
        self.__relpath = notset
        self.__url = notset
        self.__user = notset
        self.__version = notset
        (self.key, self.href, self.require_python, self.yanked) = link_info

    def __hash__(self) -> int:
        return hash(
            (
                self.__basename,
                self.__cmpval,
                self.__ext,
                self.__hashes,
                self.__index,
                self.__name,
                self.__relpath,
                self.__url,
                self.__user,
                self.__version,
                self.href,
                self.key,
                self.require_python,
                self.yanked,
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            other = other.cmpval
        return self.cmpval == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            other = other.cmpval
        return self.cmpval < other

    def __splitbasename(self) -> None:
        (self.__name, self.__version, self.__ext) = splitbasename(
            self.basename, checkarch=False)

    def __parse_url(self) -> None:
        url = URL(self.href)
        self.__basename = url.basename
        self.__hashes = Digests()
        if hash_type := url.hash_type:
            self.__hashes[hash_type] = url.hash_value
        parts = url.path.split("/")
        self.__user = parts[0]
        self.__index = parts[1]
        self.__relpath = "/".join(parts[2:])

    @property
    def basename(self) -> str:
        if self.__basename is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__basename, str)
        return self.__basename

    @property
    def hashes(self) -> Digests:
        if self.__hashes is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__hashes, Digests)
        return self.__hashes

    @property
    def index(self) -> str:
        if self.__index is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__index, str)
        return self.__index

    @property
    def path(self) -> str:
        if self.__relpath is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__index, str)
            assert isinstance(self.__relpath, str)
            assert isinstance(self.__user, str)
        return f"{self.__user}/{self.__index}/{self.__relpath}"

    @property
    def relpath(self) -> str:
        if self.__relpath is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__relpath, str)
        return self.__relpath

    @property
    def user(self) -> str:
        if self.__user is notset:
            self.__parse_url()
        if TYPE_CHECKING:
            assert isinstance(self.__user, str)
        return self.__user

    @property
    def name(self) -> str:
        if self.__name is notset:
            self.__splitbasename()
        if TYPE_CHECKING:
            assert isinstance(self.__name, str)
        return self.__name

    @property
    def version(self) -> str:
        if self.__version is notset:
            self.__splitbasename()
        if TYPE_CHECKING:
            assert isinstance(self.__version, str)
        return self.__version

    @property
    def ext(self) -> str:
        if self.__ext is notset:
            self.__splitbasename()
        if TYPE_CHECKING:
            assert isinstance(self.__ext, str)
        return self.__ext

    @property
    def cmpval(self) -> tuple[Version, NormalizedName, str]:
        if self.__cmpval is notset:
            self.__cmpval = (
                parse_version(self.version),
                normalize_name(self.name),
                self.ext)
        if TYPE_CHECKING:
            assert isinstance(self.__cmpval, tuple)
            assert isinstance(self.__cmpval[0], Version)
            assert isinstance(self.__cmpval[1], NormalizedName)
            assert isinstance(self.__cmpval[2], str)
        return self.__cmpval

    def __repr__(self) -> str:
        clsname = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return (
            f"<{clsname} "
            f"key={self.key!r} "
            f"href={self.href!r} "
            f"require_python={self.require_python!r} "
            f"yanked={self.yanked!r}>")


def normalize_bases(model, bases):
    # check and normalize base indices
    messages = []
    newbases = []
    for base in bases:
        try:
            stage_base = model.getstage(base)
        except ValueError:
            messages.append(f"invalid base index spec: {base!r}")
        else:
            if stage_base is None:
                messages.append(f"base index {base!r} does not exist")
            else:
                newbases.append(stage_base.name)
    if messages:
        raise InvalidIndexconfig(messages)
    return tuple(newbases)


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


class EventSubscribers:
    """ the 'on_' functions are called within in the notifier thread. """
    def __init__(self, xom):
        self.xom = xom

    def on_changed_version_config(self, ev: KeyChangeEvent) -> None:
        """ when version config is changed for a project in a stage"""
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

            # XXX slightly flaky logic for detecting metadata changes
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
        """ when a file entry is modified. """
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
                    stage=stage, project=name,
                    version=entry.version,
                    link=links[0])

    def on_mirror_initialnames(self, ev: KeyChangeEvent) -> None:
        """ when projectnames are first loaded into a mirror. """
        params = ev.data.key.params
        user = params.get("user")
        index = params.get("index")
        keyfs = self.xom.keyfs
        with keyfs.read_transaction(at_serial=ev.at_serial):
            stage = self.xom.model.getstage(user, index)
            if stage is not None and stage.index_type == "mirror":
                self.xom.config.hook.devpiserver_mirror_initialnames(
                    stage=stage,
                    projectnames=stage.list_projects_perstage()
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
