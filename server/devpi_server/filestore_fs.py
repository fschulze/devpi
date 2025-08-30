from __future__ import annotations

from .filestore_fs_base import FSIOFileBase
from .filestore_fs_base import LazyChangesFormatter  # noqa: F401
from .filestore_fs_base import check_pending_renames  # noqa: F401
from .filestore_fs_base import commit_renames  # noqa: F401
from .filestore_fs_base import make_rel_renames  # noqa: F401
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .keyfs_types import FilePathInfo


class FSIOFile(FSIOFileBase):
    def _make_path(self, path: FilePathInfo) -> str:
        return str(self.basedir / path.relpath)
