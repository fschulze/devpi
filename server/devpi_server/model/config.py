from __future__ import annotations

from .exceptions import InvalidIndexconfig
from attrs import define
from attrs import field
from devpi_server.markers import NotSet
from devpi_server.markers import notset
from devpi_server.normalized import normalize_name
from pyramid.authorization import Authenticated
from pyramid.authorization import Everyone
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing import Any


CT = TypeVar("CT")


@define(kw_only=True)
class ConfigField(Generic[CT]):
    _missing: CT | NotSet = field(default=notset)
    default: CT | NotSet = field(default=notset)
    name: str
    normalize: Callable[[Any], CT] | None = field(default=None)
    type: type[CT] = field(default=None)


def _convert_fields(fields: Sequence[ConfigField]) -> list[ConfigField]:
    names = set()
    result = []
    for f in fields:
        if not isinstance(f, ConfigField):
            raise TypeError
        if f.name in names:
            raise ValueError(f"Field with duplicate name {f.name!r}")
        names.add(f.name)
        result.append(f)
    return result


@define
class ConfigFields:
    fields: list[ConfigField] = field(converter=_convert_fields)

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            f.name: default
            for f in self.fields
            if not isinstance(default := f.default, NotSet)
        }

    def extend(self, fields: Sequence[ConfigField], error_msg: str) -> None:
        if "{conflicting}" not in error_msg:
            raise ValueError("Missing '{conflicting}' marker in error_msg")
        conflicting = self.names.intersection(ConfigFields(fields).names)
        if conflicting:
            raise ValueError(
                error_msg.format(conflicting=", ".join(sorted(conflicting)))
            )
        self.fields.extend(fields)

    def fill_config_from_kwargs(
        self, config: dict[str, Any], kwargs: dict[str, Any]
    ) -> None:
        # prevent default values from being removed
        for key in self.defaults:
            if kwargs.get(key) is RemoveValue:
                raise InvalidIndexconfig("Default values can't be removed.")
        # now process the new settings
        for f in self.fields:
            key = f.name
            _missing = f._missing
            if key not in kwargs and isinstance(_missing, NotSet):
                continue
            value = kwargs.pop(key, _missing)
            if value is not RemoveValue:
                normalize = f.normalize
                if normalize is not None:
                    value = normalize(value)
                if value is None:
                    raise ValueError(f"The key {key!r} wasn't processed.")
            config[key] = value
        # remove keys
        for key, value in list(kwargs.items()):
            if value is RemoveValue:
                config[key] = kwargs.pop(key)

    @property
    def names(self):
        return {f.name for f in self.fields}


class RemoveValue:
    """Marker object for index configuration keys to remove."""


class ACLList(list):
    # marker class for devpiserver_indexconfig_defaults
    pass


def ensure_acl_list(data: Any) -> list[str]:
    data = ensure_list(data)
    for index, name in enumerate(data):
        if name.upper() in (":ANONYMOUS:", ":AUTHENTICATED:"):
            data[index] = name.upper()
    return data


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


def ensure_list(data: Any) -> list[str]:
    if isinstance(data, (list, tuple, set)):
        return list(data)
    if not hasattr(data, "split"):
        raise InvalidIndexconfig("Unknown list value %r." % data)
    # split and remove empty
    return list(filter(None, (x.strip() for x in data.split(","))))


def get_principals(value):
    principals = set(value)
    if ":AUTHENTICATED:" in principals:
        principals.remove(":AUTHENTICATED:")
        principals.add(Authenticated)
    if ":ANONYMOUS:" in principals:
        principals.remove(":ANONYMOUS:")
        principals.add(Everyone)
    return principals


def normalize_bases(model, bases):
    # check and normalize base indices
    messages = []
    newbases = []
    for base in ensure_list(bases):
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


def normalize_trust_inheritance(value: Any) -> str:
    value = value.lower()
    choices = {"none", "type:not remote"}
    if value not in choices:
        raise InvalidIndexconfig.for_invalid_choice(
            "trust_inheritance_rules_from", value, choices, allow_empty=True
        )
    return value


def normalize_whitelist_name(name):
    if name == "*":
        return name
    return normalize_name(name)
