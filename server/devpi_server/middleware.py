from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse


if TYPE_CHECKING:
    from .main import XOM
    from typing import Any


class OutsideURLMiddleware:
    def __init__(self, app: Any, xom: XOM) -> None:
        self.app = app
        self.xom = xom

    def __call__(self, environ, start_response):
        outside_url = environ.get("HTTP_X_OUTSIDE_URL")
        if not outside_url:
            outside_url = self.xom.config.outside_url
        if outside_url:
            outside_url = urlparse(outside_url)
            environ["wsgi.url_scheme"] = outside_url.scheme
            environ["HTTP_HOST"] = outside_url.netloc
            if outside_url.path:
                environ["SCRIPT_NAME"] = outside_url.path
                environ["PATH_INFO"] = environ["PATH_INFO"].removeprefix(
                    outside_url.path
                )
        return self.app(environ, start_response)
