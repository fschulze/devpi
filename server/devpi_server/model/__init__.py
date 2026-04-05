from __future__ import annotations

from importlib import import_module
from importlib.util import resolve_name
from typing import TYPE_CHECKING
import warnings


if TYPE_CHECKING:
    from typing import Any


deprecated_names = dict(
    ACLList=".config.ACLList",
    BaseStage=".base.BaseIndex",
    BaseStageCustomizer=".customizer.BaseIndexCustomizer",
    PrivateStage=".local.PrivateStage",
    ensure_acl_list=".config.ensure_acl_list",
    ensure_boolean=".config.ensure_boolean",
    ensure_list=".config.ensure_list",
    get_principals=".config.get_principals",
)


def __getattr__(name: str) -> Any:
    if name in deprecated_names:
        (new_module_name, _sep, new_name) = resolve_name(
            deprecated_names[name], __spec__.parent
        ).rpartition(".")
        warnings.warn(
            f"{__name__}.{name} is deprecated, use {new_module_name}.{new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        new_module = import_module(new_module_name)
        return getattr(new_module, new_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
