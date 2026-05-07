from __future__ import annotations

from .exceptions import InvalidIndexconfig
from .exceptions import MissingValueConfigError
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
    type: type[CT] | None

    def apply_add_action(self, ixconfig: dict, value: object) -> None:
        key = self.name
        typ = self.expected_type(ixconfig)
        match typ:
            case type() if issubclass(typ, UniqueList):
                if value not in ixconfig[key]:
                    ixconfig[key].append(value)
            case type() if issubclass(typ, list):
                ixconfig[key].append(value)
            case type() if issubclass(typ, UniqueTuple):
                if value not in ixconfig[key]:
                    ixconfig[key] += (value,)
            case type() if issubclass(typ, tuple):
                ixconfig[key] += (value,)
            case _:
                raise TypeError(f"don't know how to handle type {typ!r}")

    def apply_del_action(self, ixconfig: dict, value: object) -> None:
        key = self.name
        if value not in ixconfig[key]:
            msg = f"The {key!r} setting doesn't have value {value!r}"
            raise MissingValueConfigError(msg)
        typ = self.expected_type(ixconfig)
        match typ:
            case type() if issubclass(typ, list):
                ixconfig[key].remove(value)
            case type() if issubclass(typ, tuple):
                ixconfig[key] = tuple(x for x in ixconfig[key] if x != value)
            case _:
                raise TypeError(f"don't know how to handle type {typ!r}")

    def apply_remove_action(self, ixconfig: dict, _value: object) -> None:
        ixconfig[self.name] = RemoveValue

    def apply_set_action(self, ixconfig: dict, value: object) -> None:
        ixconfig[self.name] = value

    def expected_type(self, ixconfig: dict) -> list | tuple | object:
        expected_type = self.type
        if expected_type is not None:
            return expected_type
        return type(ixconfig[self.name])


def _convert_fields(
    fields: Sequence[ConfigField] | ConfigFields | dict[str, ConfigField],
) -> dict[str, ConfigField]:
    result = {}
    if isinstance(fields, ConfigFields):
        _fields = iter(fields._fields.values())
    elif isinstance(fields, dict):
        _fields = iter(fields.values())
    else:
        _fields = iter(fields)
    for f in _fields:
        if not isinstance(f, ConfigField):
            raise TypeError
        if f.name in result:
            raise ValueError(f"Field with duplicate name {f.name!r}")
        result[f.name] = f
    return result


@define
class ConfigFields:
    _fields: dict[str, ConfigField] = field(converter=_convert_fields)

    __iter__ = None

    def __getitem__(self, name: str) -> ConfigField:
        return self._fields[name]

    def apply_actions(
        self, ixconfig: dict, actions: Sequence[tuple[str, str, object]]
    ) -> tuple[dict, bool]:
        keep_unknown = False
        used_ops = set()
        for op, key, value in actions:
            used_ops.add(op)
            field = self.get(key)
            if field is None:
                if op == "drop":
                    ixconfig[key] = RemoveValue
                    continue
                raise KeyError(f"Unknown config field {field!r}")
            match op:
                case "del":
                    field.apply_del_action(ixconfig, value)
                case "add":
                    field.apply_add_action(ixconfig, value)
                case "set":
                    field.apply_set_action(ixconfig, value)
                case "drop":
                    field.apply_remove_action(ixconfig, value)
                case _:
                    raise ValueError(f"Unknown operator {op!r}.")
        if not used_ops.difference({"add", "del", "drop"}):
            keep_unknown = True
        return (ixconfig, keep_unknown)

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            f.name: default
            for f in self._fields.values()
            if not isinstance(default := f.default, NotSet)
        }

    def extend(self, fields: Sequence[ConfigField], error_msg: str) -> None:
        _fields = {f.name: f for f in fields}
        if "{conflicting}" not in error_msg:
            raise ValueError("Missing '{conflicting}' marker in error_msg")
        conflicting = set(self._fields).intersection(_fields)
        if conflicting:
            raise ValueError(
                error_msg.format(conflicting=", ".join(sorted(conflicting)))
            )
        self._fields.update(_fields)

    def fill_config_from_kwargs(
        self, config: dict[str, Any], kwargs: dict[str, Any]
    ) -> None:
        # prevent default values from being removed
        for key in self.defaults:
            if kwargs.get(key) is RemoveValue:
                raise InvalidIndexconfig("Default values can't be removed.")
        # now process the new settings
        for f in self._fields.values():
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

    def get(self, name: str) -> ConfigField | None:
        return self._fields.get(name)

    @property
    def names(self):
        return set(self._fields)


class RemoveValue:
    """Marker object for index configuration keys to remove."""


class UniqueList(list):
    pass


class ACLList(UniqueList):
    # marker class for devpiserver_indexconfig_defaults
    pass


class UniqueTuple(tuple):
    __slots__ = ()


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
