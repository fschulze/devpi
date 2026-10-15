from __future__ import annotations

from importlib import import_module
from importlib.util import resolve_name
from typing import TYPE_CHECKING
import warnings


if TYPE_CHECKING:
    from typing import Any


deprecated_names = dict(
    MyThread=".thread.DevpiThread",
    Shutdown=".thread.Shutdown",
    ThreadPool=".thread.ThreadPool",
    current_thread=".thread.current_thread",
    has_active_thread=".thread.has_active_thread",
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
        if new_module_name:
            new_module = import_module(new_module_name)
            return getattr(new_module, new_name)
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
