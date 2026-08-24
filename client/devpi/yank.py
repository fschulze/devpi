from __future__ import annotations

from .common import get_versions_to_process
from devpi_common.metadata import parse_requirement
from devpi_common.url import URL
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .common import VersionsToProcess
    from .main import Hub
    from argparse import Namespace
    from devpi_common.metadata import Requirement
    from typing_extensions import Literal


def confirm_file(hub: Hub, url: str, action: str) -> bool:
    hub.info(f"About to {action} the following release:")
    hub.info(url)
    return hub.ask_confirm("Are you sure")


def confirm_version(
    hub: Hub, versions_to_process: VersionsToProcess, action: str
) -> bool:
    if not versions_to_process:
        hub.error(f"No versions found matching '{hub.args.spec_or_url}'.")
        return False
    hub.info(f"About to {action} the following versions and releases")
    for ver, links in versions_to_process:
        hub.info(f"version: {ver}")
        if links:
            for link in links:
                hub.info("  - " + link.href)
        else:
            hub.info("  - No releases")
    return hub.ask_confirm("Are you sure")


def _yank(hub: Hub, *, reason: Literal[False] | str) -> int | None:
    hub.require_valid_current_with_index()
    if "yank" not in hub.current.features:
        hub.fatal("Server doesn't support 'yank'.")
    action = "unyank" if reason is False else "yank"
    spec_or_url = parse_spec_or_url(hub.args.spec_or_url)
    if isinstance(spec_or_url, URL):
        # yank specified file
        url = spec_or_url.url
        if confirm_file(hub, url, action):
            r = hub.http_api("post", url, kvdict={"type": "yank", "reason": reason})
            hub.info("success", r.get_error_message(hub.args.debug))
        return None
    project = spec_or_url.project_name
    proj_url = hub.current.get_project_url(project, indexname=hub.args.index)
    reply = hub.http_api(
        "get", proj_url.replace(query=dict(ignore_bases="")), type="projectconfig"
    )
    versions_to_process = get_versions_to_process(hub, reply.result, spec_or_url)
    if confirm_version(hub, versions_to_process, action):
        for ver, _links in versions_to_process:
            hub.info(f"yanking version {ver} of {spec_or_url.project_name}")
            r = hub.http_api(
                "post", proj_url.addpath(ver), kvdict={"type": "yank", "reason": reason}
            )
            hub.info("success", r.get_error_message(hub.args.debug))
    return None


def main_unyank(
    hub: Hub,
    args: Namespace,  # noqa: ARG001 - API
) -> int | None:
    return _yank(hub, reason=False)


def main_yank(hub: Hub, args: Namespace) -> int | None:
    return _yank(hub, reason=args.reason)


def parse_spec_or_url(spec_or_url: str) -> URL | Requirement:
    if spec_or_url.startswith(("http://", "https://")):
        return URL(spec_or_url)
    return parse_requirement(spec_or_url)
