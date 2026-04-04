from __future__ import annotations

from .exceptions import InvalidIndexconfig
from devpi_server.normalized import normalize_name
from pyramid.authorization import Authenticated
from pyramid.authorization import Everyone
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any


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
