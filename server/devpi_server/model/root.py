from __future__ import annotations

from .customizer import UnknownCustomizer
from .exceptions import InvalidIndex
from .exceptions import InvalidIndexconfig
from .exceptions import InvalidUser
from .user import User
from devpi_server.log import threadlog
from time import gmtime
from time import strftime
from typing import TYPE_CHECKING
import re


if TYPE_CHECKING:
    from .local import BaseStage
    from devpi_server.main import XOM


name_char_blocklist_regexp = re.compile(
    r"[\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f"
    r"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    r' !"#$%&\'()*+,/:;<=>?\[\\\\\]^`{|}~]'
)


def is_valid_name(name: str) -> bool:
    return not name_char_blocklist_regexp.search(name)


class RootModel:
    """per-process root model object."""

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
                "Any ascii symbol besides -.@_ is blocked." % username
            )
        user = User(self, username)
        kwargs.update(created=strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()))
        user._modify(password=password, **kwargs)
        if "email" in kwargs:
            threadlog.info(
                "created user %r with email %r" % (username, kwargs["email"])
            )
        else:
            threadlog.info("created user %r" % username)
        # Call any user created hooks, passing along the newly created user object.
        self.xom.config.hook.devpiserver_user_created(user=user)
        return user

    def create_stage(
        self,
        user,
        index,
        type="stage",  # noqa: A002
        **kwargs,
    ):
        if user.key_index.with_resolved_parent()(index).exists():
            raise InvalidIndex("indexname '%s' already exists" % index)
        if not is_valid_name(index):
            raise InvalidIndex(
                "indexname '%s' contains characters that aren't allowed. "
                "Any ascii symbol besides -.@_ is blocked." % index
            )
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
            assert not user.startswith("/")
            assert not user.endswith("/")
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
