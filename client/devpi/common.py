from __future__ import annotations

from devpi_common.metadata import Version
from devpi_common.viewhelp import ViewLinkStore
from operator import attrgetter
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .main import Hub
    from collections.abc import Sequence
    from devpi_common.metadata import Requirement
    from devpi_common.viewhelp import ViewLink
    from typing import Tuple

    VersionsToProcess = Sequence[Tuple[str, Sequence[ViewLink]]]


def get_versions_to_process(
    hub: Hub, result: dict, requirement: Requirement
) -> VersionsToProcess:
    index_url = hub.current.get_index_url(indexname=getattr(hub.args, "index", None))
    basepath = index_url.path.lstrip("/")
    versions_to_process = []
    for version, verdata in result.items():
        if version in requirement:
            vv = ViewLinkStore(basepath, verdata)
            files_to_delete = sorted(
                (
                    link
                    for link in vv.get_links()
                    if link.href.startswith(index_url.url)
                ),
                key=attrgetter("basename"),
            )
            versions_to_process.append((version, files_to_delete))
    # filter versions with no releases and sort by version
    return sorted((x for x in versions_to_process if x[1]), key=lambda x: Version(x[0]))
