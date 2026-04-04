from __future__ import annotations

from .customizer import get_stage_customizer_class
from .exceptions import InvalidUserconfig
from .local import PrivateStage
from .remote import MirrorStage
from devpi_common.types import ensure_unicode
from devpi_server.auth import hash_password
from devpi_server.auth import verify_and_update_password_hash
from devpi_server.log import threadlog
from devpi_server.markers import notset
from devpi_server.readonly import get_mutable_deepcopy
from time import gmtime
from time import strftime
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .local import BaseStage
    from .root import RootModel
    from .schema import Schema
    from devpi_server.keyfs import KeyFS


class User:
    keyfs: KeyFS[Schema]

    # ignored_keys are skipped on create and modify
    ignored_keys = frozenset(("indexes", "username"))
    # info keys are updated on create and modify and input is ignored
    info_keys = frozenset(("created", "modified"))
    hidden_keys = frozenset(("password", "pwhash", "pwsalt"))
    public_keys = frozenset(("custom_data", "description", "email", "title"))
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
                "Unknown keys in user config: %s" % ", ".join(unknown_keys)
            )

    def _modify(self, password=None, pwhash=None, **kwargs):
        self.validate_config(**kwargs)
        modified: dict[str, object] = {}
        with self.key.update() as userconfig:
            if password is not None or pwhash:
                self._setpassword(userconfig, password, pwhash=pwhash)
                modified["password"] = "*******"  # noqa: S105
                kwargs["pwsalt"] = None
            for raw_key, raw_value in kwargs.items():
                key = ensure_unicode(raw_key)
                if raw_value:
                    if userconfig.get(key, notset) != raw_value:
                        userconfig[key] = raw_value
                        value = "*******" if key in self.hidden_keys else raw_value
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
        threadlog.info("modified user %r: %s", self.name, ", ".join(modified))

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

    def get(self, *, credentials=False):
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

    def create_stage(
        self,
        index,
        type="stage",  # noqa: A002
        **kwargs,
    ):
        return self.parent.create_stage(self, index, type=type, **kwargs)

    def _getstage(self, indexname, index_type, ixconfig):
        cls = MirrorStage if index_type == "mirror" else PrivateStage
        customizer_cls = get_stage_customizer_class(self.xom, index_type)
        return cls(
            self.xom,
            username=self.name,
            index=indexname,
            ixconfig=ixconfig,
            customizer_cls=customizer_cls,
        )

    def getstage(self, indexname: str) -> BaseStage | None:
        return self.parent.getstage(self.name, indexname)

    def getstages(self) -> list[BaseStage]:
        stages = []
        for index in self.get()["indexes"]:
            stage = self.getstage(index)
            assert stage is not None
            stages.append(stage)
        return stages
