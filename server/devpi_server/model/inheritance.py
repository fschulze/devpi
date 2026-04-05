from __future__ import annotations

from .exceptions import UpstreamError
from abc import ABC
from abc import abstractmethod
from attrs import define
from attrs import field
from devpi_common.types import cached_property
from devpi_server.log import threadlog
from devpi_server.markers import NotSet
from devpi_server.markers import Unknown
from devpi_server.markers import notset
from devpi_server.markers import unknown
from repoze.lru import LRUCache
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .base import BaseIndex
    from .local import PrivateStageType
    from .root import RootModel
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from devpi_server.normalized import NormalizedName
    from types import TracebackType
    from typing import Literal
    from typing import Self


class check_upstream_error:
    def __init__(self, current: BaseIndex, other: BaseIndex) -> None:
        self.current = current
        self.other = other
        self.failed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        cls: type[BaseException] | None,
        val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if not isinstance(val, UpstreamError):
            return False
        if self.other is self.current:
            # If we are currently checking ourself raise the error, it is fatal
            return False
        threadlog.warn(
            "Failed to check mirror whitelist. Assume it does not exist (%s)", val
        )
        self.failed = True
        return True


@define(slots=False)
class InheritanceInfo:
    traversal_infos: list[tuple[TraversalInfo, bool | NotSet | Unknown]]

    @cached_property
    def blocked_mirror_name(self) -> str | None:
        for traversal_info, has_project in self._iter_mirrors():
            if isinstance(traversal_info, BlockedTraversal):
                return traversal_info.stage.name
            if isinstance(has_project, Unknown):
                return None
            if not isinstance(traversal_info, (AllowedTraversal, UntrustedTraversal)):
                return traversal_info.stage.name
            return None
        return None

    @cached_property
    def has_project_from_mirror(self) -> bool | Unknown:
        for traversal_info, has_project in self._iter_mirrors():
            if isinstance(traversal_info, BlockedTraversal):
                return unknown
            if isinstance(has_project, NotSet):
                raise TypeError
            return has_project
        return False

    def _iter_mirrors(self) -> Iterator[tuple[TraversedStage, bool | NotSet | Unknown]]:
        for traversal_info, has_project in self._unique_traversed_stages:
            if traversal_info.stage.index_type != "mirror":
                continue
            yield (traversal_info, has_project)

    def iter_stages(self, opname: str) -> Iterator[BaseIndex]:
        for traversal_info, has_project in self._unique_traversed_stages:
            if isinstance(traversal_info, BlockedTraversal):
                threadlog.debug("%s: %s", opname, traversal_info.reason)
                continue
            if isinstance(traversal_info, AllowedTraversal) and isinstance(
                traversal_info.reason, WhitelistAllowed
            ):
                threadlog.debug("%s: %s", opname, traversal_info.reason)
            if has_project is True or (
                has_project is unknown
                and not isinstance(traversal_info, UntrustedTraversal)
            ):
                yield traversal_info.stage

    @cached_property
    def _unique_traversed_stages(
        self,
    ) -> list[tuple[TraversedStage, bool | NotSet | Unknown]]:
        return [
            (traversal_info, has_project)
            for traversal_info, has_project in self.traversal_infos
            if isinstance(traversal_info, TraversedStage) and not traversal_info.seen
        ]


@define(frozen=True, kw_only=True)
class PermissionReason(ABC):
    name: str
    project: NormalizedName

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


@define(frozen=True, kw_only=True)
class PermissionAllowed(PermissionReason):
    pass


@define(frozen=True, kw_only=True)
class PermissionDenied(PermissionReason):
    pass


@define(frozen=True, kw_only=True)
class DirectAccess(PermissionAllowed):
    def __str__(self) -> str:
        return f"Direct access to package {str(self.project)!r}, allowing {self.name}"


@define(frozen=True, kw_only=True)
class WhitelistAllowed(PermissionAllowed):
    src_name: str

    def __str__(self) -> str:
        return f"private package {str(self.project)!r} whitelisted at stage {self.src_name}, allowing {self.name}"


@define(frozen=True, kw_only=True)
class WhitelistBlocked(PermissionDenied):
    src_name: str

    def __str__(self) -> str:
        return f"private package {str(self.project)!r} from {self.src_name} not whitelisted, blocking {self.name}"


@define(frozen=True, kw_only=True)
class SkipReason(ABC):
    name: str

    @abstractmethod
    def __str__(self) -> str: ...


class InheritanceCycle(SkipReason):
    def __str__(self) -> str:
        return f"skipped {self.name} due to an inheritance cycle"


class MissingIndex(SkipReason):
    def __str__(self) -> str:
        return f"skipped {self.name} because it does not exist"


class SROSkipHook(SkipReason):
    def __str__(self) -> str:
        return f"skipped {self.name} by devpiserver_sro_skip result"


@define(frozen=True, kw_only=True)
class TraversalInfo(ABC):
    name: str


@define(frozen=True, kw_only=True)
class PostponedTraversal(TraversalInfo):
    pass


@define(frozen=True, kw_only=True)
class TraversedStage(TraversalInfo):
    seen: bool
    stage: BaseIndex

    def allow(self, *, reason: PermissionAllowed) -> AllowedTraversal:
        return AllowedTraversal(
            name=self.name,
            reason=reason,
            seen=self.seen,
            stage=self.stage,
        )

    def block(self, *, reason: PermissionDenied) -> BlockedTraversal:
        return BlockedTraversal(
            name=self.name,
            reason=reason,
            seen=self.seen,
            stage=self.stage,
        )


@define(frozen=True, kw_only=True)
class AllowedTraversal(TraversedStage):
    reason: PermissionAllowed


@define(frozen=True, kw_only=True)
class BlockedTraversal(TraversedStage):
    reason: PermissionDenied


@define(frozen=True, kw_only=True)
class UntrustedTraversal(TraversedStage):
    pass


@define(frozen=True, kw_only=True)
class SkippedTraversal(TraversalInfo):
    reason: SkipReason
    src: str


@define(kw_only=True)
class InheritancePolicy:
    PrivateStage: PrivateStageType = field(init=False)
    private_hit: BaseIndex | Literal[False] = field(default=False, init=False)
    project: NormalizedName
    stage: BaseIndex
    whitelist: set[str] = field(init=False)
    whitelist_merger: Callable[[set[str]], set[str]] = field(init=False)
    whitelisted: BaseIndex | Literal[False] = field(default=False, init=False)

    def __attrs_post_init__(self) -> None:
        from .local import PrivateStage

        self.PrivateStage = PrivateStage
        self.whitelist = self._get_whitelist(self.stage)
        match self.stage.ixconfig.get("mirror_whitelist_inheritance", "union"):
            case "intersection":
                self.whitelist_merger = self.whitelist.intersection
            case "union":
                self.whitelist_merger = self.whitelist.union
            case whitelist_inheritance:
                msg = f"Unknown whitelist_inheritance setting {whitelist_inheritance!r}"
                raise RuntimeError(msg)

    def _get_whitelist(self, stage: BaseIndex) -> set[str]:
        return set(stage.ixconfig.get("mirror_whitelist", set()))

    def update(
        self, traversed_stage: TraversedStage
    ) -> tuple[PermissionReason | None, bool | NotSet | Unknown]:
        stage = traversed_stage.stage
        untrusted = isinstance(traversed_stage, UntrustedTraversal)
        if untrusted and self.private_hit is not False and self.whitelisted is False:
            return (
                WhitelistBlocked(
                    name=stage.name,
                    project=self.project,
                    src_name=self.private_hit.name,
                ),
                notset,
            )
        with check_upstream_error(self.stage, stage) as checker:
            exists = stage.has_project_perstage(self.project)
        if checker.failed:
            return (None, unknown)
        if isinstance(stage, self.PrivateStage):
            self.whitelist = self.whitelist_merger(self._get_whitelist(stage))
            if self.whitelist.intersection({"*", self.project}):
                self.whitelisted = stage
            elif exists:
                self.private_hit = stage
        if self.private_hit is False and exists is unknown and stage.no_project_list:
            # direct fetching is allowed
            return (DirectAccess(name=stage.name, project=self.project), exists)
        if not exists or not untrusted or self.whitelisted is False:
            return (None, exists)
        return (
            WhitelistAllowed(
                name=stage.name, project=self.project, src_name=self.whitelisted.name
            ),
            exists,
        )


class IndexBases:
    def __init__(
        self, stage: BaseIndex, *, devpiserver_sro_skip: Callable, model: RootModel
    ) -> None:
        self._per_project_mergability_cache = LRUCache(8)
        self.devpiserver_sro_skip = devpiserver_sro_skip
        self.model = model
        self.stage = stage

    def __iter__(self) -> Iterator[str]:
        return iter(self.bases)

    def __reversed__(self) -> Iterator[str]:
        return iter(reversed(self.bases))

    @cached_property
    def bases(self) -> tuple[str]:
        """Returns bases as tuple of strings."""
        return tuple(self.stage.ixconfig.get("bases", ()))

    def _get_inheritance_infos(self, project: NormalizedName) -> InheritanceInfo:
        filtered_project = not self.stage.filter_projects([project])
        policy = InheritancePolicy(project=project, stage=self.stage)
        traversal_infos: list[tuple[TraversalInfo, bool | NotSet | Unknown]] = []
        for traversal_info in self.traversal_infos:
            if filtered_project or not isinstance(traversal_info, TraversedStage):
                traversal_infos.append((traversal_info, notset))
                continue
            (reason, exists) = policy.update(traversal_info)
            match reason:
                case PermissionDenied():
                    traversal_infos.append(
                        (traversal_info.block(reason=reason), exists)
                    )
                case PermissionAllowed():
                    traversal_infos.append(
                        (traversal_info.allow(reason=reason), exists)
                    )
                case _:
                    traversal_infos.append((traversal_info, exists))
        return InheritanceInfo(traversal_infos)

    def get_inheritance_infos(self, project: NormalizedName) -> InheritanceInfo:
        result = self._per_project_mergability_cache.get(project)
        if result is None:
            result = self._get_inheritance_infos(project)
            self._per_project_mergability_cache.put(project, result)
        return result

    def get_mergeable_stages(
        self, project: NormalizedName, opname: str
    ) -> Iterable[BaseIndex]:
        return self.get_inheritance_infos(project).iter_stages(opname)

    def iter_stages(self) -> Iterator[BaseIndex]:
        """Iterates stages in defined order without loops."""
        for traversal_info in self.traversal_infos:
            match traversal_info:
                case (
                    PostponedTraversal()
                    | SkippedTraversal(reason=InheritanceCycle())
                    | TraversedStage(seen=True)
                ):
                    continue
                case SkippedTraversal(name=name, reason=MissingIndex(), src=src):
                    threadlog.warn(
                        "Index %s refers to non-existing base %s.", src, name
                    )
                    continue
                case SkippedTraversal(name=name, reason=SROSkipHook(), src=src):
                    threadlog.warn(
                        "Index %s base %s excluded via devpiserver_sro_skip.",
                        src,
                        name,
                    )
                    continue
                case TraversedStage(seen=False, stage=stage):
                    yield stage
                case _:
                    raise RuntimeError(traversal_info)

    def is_untrusted(self, stage: BaseIndex) -> bool:
        # we have to postpone mirrors, as there
        # may be private releases in other paths
        return stage.index_type == "mirror"

    @cached_property
    def traversal_infos(self) -> list[TraversalInfo]:
        """Returns traversal information."""
        devpiserver_sro_skip = self.devpiserver_sro_skip
        getstage = self.model.getstage
        info: list[TraversalInfo] = []
        postponed: list[tuple[BaseIndex, list[str]]] = []
        seen = set()
        is_untrusted = self.is_untrusted
        stage = self.stage
        todo = [(stage, list(reversed(stage.index_bases)))]
        while todo:
            (current_stage, bases) = todo[-1]
            current_name = current_stage.name
            if bases or current_name not in seen:
                info.append(
                    (
                        UntrustedTraversal
                        if is_untrusted(current_stage)
                        else TraversedStage
                    )(
                        name=current_name,
                        seen=current_name in seen,
                        stage=current_stage,
                    )
                )
            seen.add(current_name)
            if not bases:
                todo.pop()
                if not todo:
                    todo.extend(postponed)
                    postponed.clear()
                continue
            next_name = bases.pop()
            if next_name in seen:
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=InheritanceCycle(name=next_name),
                        src=current_name,
                    )
                )
                continue
            next_stage = getstage(next_name)
            if next_stage is None:
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=MissingIndex(name=next_name),
                        src=current_name,
                    )
                )
                seen.add(next_name)
                continue
            if devpiserver_sro_skip(stage=stage, base_stage=next_stage):
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=SROSkipHook(name=next_name),
                        src=current_name,
                    )
                )
                seen.add(next_name)
                continue
            if is_untrusted(next_stage):
                info.append(PostponedTraversal(name=next_name))
                postponed.append((next_stage, list(reversed(next_stage.index_bases))))
            else:
                todo.append((next_stage, list(reversed(next_stage.index_bases))))
        return info
