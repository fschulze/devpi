from __future__ import annotations

from typing import TYPE_CHECKING
from webob.headers import EnvironHeaders
from webob.headers import ResponseHeaders


if TYPE_CHECKING:
    from httpx import Response
    from pyramid.request import Request


hop_by_hop = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


def clean_request_headers(request: Request) -> EnvironHeaders:
    result = EnvironHeaders({})
    result.update(request.headers)
    result.pop("host", None)
    return result


def clean_response_headers(response: Response) -> ResponseHeaders:
    headers = ResponseHeaders()
    # remove hop by hop headers, see:
    # https://www.mnot.net/blog/2011/07/11/what_proxies_must_do
    hop_keys = set(hop_by_hop)
    connection = response.headers.get("connection")
    if connection and connection.lower() != "close":
        hop_keys.update(x.strip().lower() for x in connection.split(","))
    for k, v in response.headers.items():
        if k.lower() in hop_keys:
            continue
        headers[k] = v
    return headers
