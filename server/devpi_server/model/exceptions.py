from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .links import ELink


class InvalidConfig(Exception, ABC):
    def __init__(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = messages
        Exception.__init__(self, messages)

    @classmethod
    def for_invalid_choice(cls, key, value, choices, *, allow_empty=False):
        if len(choices) == 0:
            raise RuntimeError
        _choices = [f"{c!r}" for c in sorted(choices)]
        if allow_empty:
            _choices.append("left unset")
        num_choices = len(_choices)
        if num_choices == 1:
            choices_str = _choices[0]
        else:
            choices_str = f"{', '.join(_choices[:-1])} or {_choices[-1]}"
        msg = f"Unknown value {value!r} for {key}, must be {choices_str}."
        return cls(msg)


class ModelException(Exception):
    """Base Exception."""

    def __init__(self, msg, *args):
        if args:
            msg = msg % args
        self.msg = msg
        Exception.__init__(self, msg)


class InvalidIndex(ModelException):
    """If a indexname is invalid or already in use."""


class InvalidIndexconfig(InvalidConfig):
    pass


class InvalidUser(ModelException):
    """If a username is invalid or already in use."""


class InvalidUserconfig(InvalidConfig):
    pass


class MissesRegistration(ModelException):
    """A prior registration of release metadata is required."""


class MissesVersion(ModelException):
    """A version number is required."""


class NotFound(ModelException):
    """If a project or version cannot be found."""


class NonVolatile(ModelException):
    """A release is overwritten on a non volatile index."""

    link: ELink  # the conflicting link


class UpstreamError(ModelException):
    """If an upstream could not be reached or didn't respond correctly."""


class UpstreamNotFoundError(UpstreamError):
    """If upstream returned a not found error."""


class UpstreamNotModified(ModelException):
    """If upstream returned a not modified reply."""

    def __init__(self, msg, *args, etag=None):
        super().__init__(msg, *args)
        self.etag = etag


class ReadonlyIndex(ModelException):
    """If a indexname is invalid or already in use."""
