from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .links import ELink


class ModelException(Exception):
    """Base Exception."""

    def __init__(self, msg, *args):
        if args:
            msg = msg % args
        self.msg = msg
        Exception.__init__(self, msg)


class InvalidIndex(ModelException):
    """If a indexname is invalid or already in use."""


class InvalidIndexconfig(Exception):
    def __init__(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = messages
        Exception.__init__(self, messages)


class InvalidUser(ModelException):
    """If a username is invalid or already in use."""


class InvalidUserconfig(Exception):
    def __init__(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = messages
        Exception.__init__(self, messages)


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
