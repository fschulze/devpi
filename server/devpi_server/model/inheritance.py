from __future__ import annotations

from .exceptions import UpstreamError
from abc import ABC
from abc import abstractmethod
from attrs import define
from attrs import evolve
from attrs import field
from collections import defaultdict
from devpi_common.types import cached_property
from devpi_server.log import threadlog
from devpi_server.markers import NotSet
from devpi_server.markers import Unknown
from devpi_server.markers import notset
from devpi_server.markers import unknown
from devpi_server.readonly import DictViewReadonly
from repoze.lru import LRUCache
from typing import NewType
from typing import TYPE_CHECKING
from typing import overload


if TYPE_CHECKING:
    from .base import BaseIndex
    from .local import LocalIndexType
    from .root import RootModel
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from devpi_server.normalized import NormalizedName
    from types import TracebackType
    from typing import Literal
    from typing import Self
    from typing import TypeVar

    FilterItem = TypeVar("FilterItem")
    FilterResultType = TypeVar("FilterResultType")
    FilterResult = list[tuple[BaseIndex, FilterResultType]]


def apply_filter_iter(
    items: Iterable[FilterItem], filter_iter: Iterator[bool]
) -> Iterator[FilterItem]:
    for item in items:
        if next(filter_iter, True):
            yield item


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
            "Failed to check remote project existence to apply inheritance rules. Assume it does not exist (%s)",
            val,
        )
        self.failed = True
        return True


def filter_result(
    filter_name: str,
    paths: TraversalPaths,
    project: NormalizedName | None,
    result: FilterResult,
) -> dict[BaseIndex, list[FilterResultType]]:
    per_path_result = {(i, p.path_key): r for p in paths for i, r in result if i in p}
    for filter_index, path_key, path_index in iter_filter_path_indexes(paths):
        result_key = (path_index, path_key)
        data = per_path_result.get(result_key)
        if not data:
            continue
        match filter_name:
            case "get_projects_filter_iter":
                iterator = filter_index.customizer.get_projects_filter_iter(data)
            case "get_simple_links_filter_iter":
                assert project is not None
                iterator = filter_index.customizer.get_simple_links_filter_iter(
                    project, data
                )
            case "get_versions_filter_iter":
                assert project is not None
                iterator = filter_index.customizer.get_versions_filter_iter(
                    project, data
                )
            case _:
                raise RuntimeError(f"Invalid filter name {filter_name!r}")
        if iterator is None:
            continue
        if isinstance(data, (DictViewReadonly, dict)):
            data = dict(apply_filter_iter(data.items(), iterator))
        else:
            data = data.__class__(apply_filter_iter(data, iterator))
        per_path_result[result_key] = data
    per_index_results = defaultdict(list)
    for (result_index, _path_key), data in per_path_result.items():
        per_index_results[result_index].append(data)
    return per_index_results


def iter_filter_path_indexes(
    paths: TraversalPaths,
) -> Iterator[tuple[BaseIndex, PathKey, BaseIndex]]:
    for path in paths:
        for i, filter_index in enumerate(path):
            for path_index in path[i:]:
                yield (filter_index, path.path_key, path_index)


@define(kw_only=True, slots=False)
class IndexInheritanceInfo:
    index_bases_map: dict[str, tuple[str, ...] | None]
    paths: TraversalPaths
    traversal_infos: list[TraversalInfo]

    def filter_result(
        self, filter_name: str, result: FilterResult
    ) -> dict[BaseIndex, list[FilterResultType]]:
        return filter_result(filter_name, self.paths, None, result)

    def iter_indexes(self) -> Iterator[TraversedIndex]:
        """Iterates indexes in defined order without loops."""
        for traversal_info in self.traversal_infos:
            match traversal_info:
                case (
                    PostponedTraversal()
                    | SkippedTraversal(reason=InheritanceCycle())
                    | TraversedIndex(seen=True)
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
                case TraversedIndex(seen=False):
                    yield traversal_info
                case _:
                    raise RuntimeError(traversal_info)

    def jsonable(self) -> dict:
        return dict(
            index_bases=self.index_bases_map,
            traversal_infos=self.jsonable_traversal_infos(),
        )

    def jsonable_traversal_infos(self) -> list:
        return [ti.jsonable() for ti in self.traversal_infos]


has_project_str_map = {
    False: "false",
    True: "true",
    notset: "notset",
    unknown: "unknown",
}


@define(kw_only=True, slots=False)
class ProjectInheritanceInfo:
    index_bases_map: dict[str, tuple[str, ...] | None]
    paths: TraversalPaths
    project: NormalizedName
    traversal_infos: list[tuple[TraversalInfo, bool | NotSet | Unknown]]

    @cached_property
    def blocked_remote_name(self) -> str | None:
        for traversal_info, has_project in self._iter_remotes():
            if isinstance(traversal_info, BlockedTraversal):
                return traversal_info.index.name
            if isinstance(has_project, Unknown):
                return None
            if not isinstance(traversal_info, (AllowedTraversal, UntrustedTraversal)):
                return traversal_info.index.name
            return None
        return None

    @cached_property
    def has_project_from_remote(self) -> bool | Unknown:
        for traversal_info, has_project in self._iter_remotes():
            if isinstance(traversal_info, BlockedTraversal):
                return unknown
            if isinstance(has_project, NotSet):
                raise TypeError
            return has_project
        return False

    def _iter_remotes(self) -> Iterator[tuple[TraversedIndex, bool | NotSet | Unknown]]:
        for traversal_info, has_project in self._unique_traversed_indexes:
            if traversal_info.index.index_type != "remote":
                continue
            yield (traversal_info, has_project)

    def filter_result(
        self, filter_name: str, result: FilterResult
    ) -> dict[BaseIndex, list[FilterResultType]]:
        return filter_result(filter_name, self.paths, self.project, result)

    def iter_indexes(self, opname: str) -> Iterator[TraversedIndex]:
        for traversal_info, has_project in self._unique_traversed_indexes:
            if isinstance(traversal_info, BlockedTraversal):
                threadlog.debug("%s: %s", opname, traversal_info.reason)
                continue
            if isinstance(traversal_info, AllowedTraversal) and isinstance(
                traversal_info.reason, RulesAllow
            ):
                threadlog.debug("%s: %s", opname, traversal_info.reason)
            if has_project is True or (
                has_project is unknown
                and not isinstance(traversal_info, UntrustedTraversal)
            ):
                yield traversal_info

    def jsonable(self) -> dict:
        return dict(
            has_project_from_remote=has_project_str_map[self.has_project_from_remote],
            index_bases=self.index_bases_map,
            traversal_infos=self.jsonable_traversal_infos(),
        )

    def jsonable_traversal_infos(self) -> list:
        return [
            ti.jsonable() | dict(has_project=has_project_str_map[has_project])
            for ti, has_project in self.traversal_infos
        ]

    @cached_property
    def _unique_traversed_indexes(
        self,
    ) -> list[tuple[TraversedIndex, bool | NotSet | Unknown]]:
        return [
            (traversal_info, has_project)
            for traversal_info, has_project in self.traversal_infos
            if isinstance(traversal_info, TraversedIndex) and not traversal_info.seen
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
class RulesAllow(PermissionAllowed):
    rule: str
    src_name: str

    def __str__(self) -> str:
        return f"package {str(self.project)!r} {self.rule} from {self.src_name} allows merging releases from index {self.name}"


@define(frozen=True, kw_only=True)
class FilterBlock(PermissionDenied):
    def __str__(self) -> str:
        return f"package {str(self.project)!r} was filtered by index {self.name}"


@define(frozen=True, kw_only=True)
class LocalHitBlock(PermissionDenied):
    src_name: str

    def __str__(self) -> str:
        return f"package {str(self.project)!r} located in {self.src_name} blocks releases from index {self.name}"


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

    @abstractmethod
    def jsonable(self) -> dict: ...


@define(frozen=True, kw_only=True)
class PostponedTraversal(TraversalInfo):
    def jsonable(self) -> dict:
        return dict(action="postponed", name=self.name)


@define(frozen=True, kw_only=True)
class TraversedIndex(TraversalInfo):
    index: BaseIndex
    path: TraversalPath
    seen: bool

    def allow(self, *, reason: PermissionAllowed) -> AllowedTraversal:
        return AllowedTraversal(
            index=self.index,
            name=self.name,
            path=self.path,
            reason=reason,
            seen=self.seen,
        )

    def block(self, *, reason: PermissionDenied) -> BlockedTraversal:
        return BlockedTraversal(
            index=self.index,
            name=self.name,
            path=self.path,
            reason=reason,
            seen=self.seen,
        )

    def jsonable(self) -> dict:
        return dict(action="traversed", name=self.name)

    def with_seen(
        self,
        seen: bool,  # noqa: FBT001
    ) -> Self:
        return evolve(self, seen=seen)


@define(frozen=True, kw_only=True)
class AllowedTraversal(TraversedIndex):
    reason: PermissionAllowed

    def jsonable(self) -> dict:
        return dict(action="allowed", name=self.name, reason=str(self.reason))


@define(frozen=True, kw_only=True)
class BlockedTraversal(TraversedIndex):
    reason: PermissionDenied

    def jsonable(self) -> dict:
        return dict(action="blocked", name=self.name, reason=str(self.reason))


@define(frozen=True, kw_only=True)
class UntrustedTraversal(TraversedIndex):
    def jsonable(self) -> dict:
        return dict(action="untrusted", name=self.name)


@define(frozen=True, kw_only=True)
class SkippedTraversal(TraversalInfo):
    reason: SkipReason
    src: str

    def jsonable(self) -> dict:
        return dict(action="skipped", name=self.name, reason=str(self.reason))


def _convert_path(path: Sequence[BaseIndex]) -> tuple[BaseIndex, ...]:
    return tuple(path)


PathKey = NewType("PathKey", tuple[str, ...])


@define
class TraversalPath:
    _path: tuple[BaseIndex, ...] = field(converter=_convert_path)
    _path_key: PathKey | NotSet = field(default=notset, init=False)

    @overload
    def __getitem__(self, index: int) -> BaseIndex: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[BaseIndex]: ...

    def __getitem__(self, index: int | slice) -> BaseIndex | tuple[BaseIndex, ...]:
        return self._path[index]

    def __iter__(self) -> Iterator[BaseIndex]:
        return iter(self._path)

    def join_index(self, index: BaseIndex) -> TraversalPath:
        return TraversalPath((*self._path, index))

    @property
    def path_key(self) -> PathKey:
        if isinstance(self._path_key, NotSet):
            self._path_key = PathKey(tuple(x.name for x in self))
        return self._path_key


TraversalPaths = NewType("TraversalPaths", tuple["TraversalPath", ...])


@define(kw_only=True)
class TraversalInfos:
    paths: TraversalPaths
    steps: list[TraversalInfo]


@define
class Rule(ABC):
    rule: str


class InheritRule(Rule, ABC):
    def allow(
        self, index: BaseIndex, local_exists_at: BaseIndex | Literal[False]
    ) -> bool | None:
        match self.rule:
            case "allow all":
                return True
            case "block type:remote if local_exists":
                if index.index_type == "remote":
                    return local_exists_at is not False
            case _:
                msg = f"Unknown rule setting {self.rule!r}"
                raise RuntimeError(msg)
        return None

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class IndexRule(InheritRule):
    def __str__(self) -> str:
        return f"index rule {self.rule!r}"


class ProjectRule(InheritRule):
    def __str__(self) -> str:
        return f"project rule {self.rule!r}"


class TrustFromRule(Rule):
    pass


def _convert_rules(rules: Sequence[InheritRule]) -> list[InheritRule]:
    return list(rules)


@define
class InheritRules:
    rules: list[InheritRule] = field(converter=_convert_rules)

    def allow(
        self, index: BaseIndex, local_exists_at: BaseIndex | Literal[False]
    ) -> InheritRule | None:
        for rule in self.rules:
            match rule.allow(index=index, local_exists_at=local_exists_at):
                case None:
                    continue
                case False:
                    raise NotImplementedError
                case True:
                    return rule
        return None

    def merge(
        self,
        rules: Sequence[InheritRule],
        *,
        index: BaseIndex,
        trust_from: TrustFromRule,
    ) -> None:
        match trust_from.rule:
            case "none":
                pass
            case "type:not remote":
                if index != "remote":
                    self.rules = _convert_rules(rules)
            case _:
                msg = (
                    f"Unknown trust_inheritance_rules_from setting {trust_from.rule!r}"
                )
                raise RuntimeError(msg)


@define(kw_only=True)
class InheritancePolicy:
    LocalIndex: LocalIndexType = field(init=False, repr=False)
    allowed_by: tuple[BaseIndex, InheritRule] | Literal[False] = field(
        default=False, init=False
    )
    index: BaseIndex
    local_exists_at: BaseIndex | Literal[False] = field(default=False, init=False)
    project: NormalizedName
    project_dict: dict[str, NormalizedName] = field(init=False)
    rules: InheritRules = field(init=False)
    trust_from: TrustFromRule = field(init=False)

    def __attrs_post_init__(self) -> None:
        from .local import LocalIndex

        self.LocalIndex = LocalIndex
        self.project_dict = {self.project.original: self.project}
        self.rules = InheritRules(self._get_rules(self.index))
        self.trust_from = TrustFromRule(
            self.index.ixconfig.get("trust_inheritance_rules_from", "none")
        )

    def _get_rules(self, index: BaseIndex) -> Sequence[InheritRule]:
        rules = index.get_projectconfig(self.project).get("inheritance_rules")
        if rules is not None:
            return [ProjectRule(r) for r in rules]
        return [
            IndexRule(r) for r in index.ixconfig.get("project_inheritance_rules", ())
        ]

    def update(
        self, traversed_index: TraversedIndex
    ) -> tuple[PermissionReason | None, bool | NotSet | Unknown]:
        index = traversed_index.index
        if not index.filter_projects(self.project_dict):
            return (FilterBlock(name=index.name, project=self.project), notset)
        untrusted = isinstance(traversed_index, UntrustedTraversal)
        if untrusted and self.local_exists_at is not False and self.allowed_by is False:
            return (
                LocalHitBlock(
                    name=index.name,
                    project=self.project,
                    src_name=self.local_exists_at.name,
                ),
                notset,
            )
        with check_upstream_error(self.index, index) as checker:
            exists = index.has_project_perstage(self.project)
        if checker.failed:
            return (None, unknown)
        if isinstance(index, self.LocalIndex):
            self.rules.merge(
                self._get_rules(index), index=index, trust_from=self.trust_from
            )
            if (
                rule := self.rules.allow(
                    index=index, local_exists_at=self.local_exists_at
                )
            ) is not None:
                self.allowed_by = (index, rule)
            elif exists:
                self.local_exists_at = index
        if (
            self.local_exists_at is False
            and exists is unknown
            and index.no_project_list
        ):
            # direct fetching is allowed
            return (DirectAccess(name=index.name, project=self.project), exists)
        if not exists or not untrusted or self.allowed_by is False:
            return (None, exists)
        return (
            RulesAllow(
                name=index.name,
                project=self.project,
                rule=str(self.allowed_by[1]),
                src_name=self.allowed_by[0].name,
            ),
            exists,
        )


class IndexBases:
    def __init__(
        self, index: BaseIndex, *, devpiserver_sro_skip: Callable, model: RootModel
    ) -> None:
        self._per_project_mergability_cache = LRUCache(8)
        self.devpiserver_sro_skip = devpiserver_sro_skip
        self.index = index
        self.model = model

    def __iter__(self) -> Iterator[str]:
        return iter(self.bases)

    def __reversed__(self) -> Iterator[str]:
        return iter(reversed(self.bases))

    @cached_property
    def _index_bases_map(self) -> dict[str, tuple[str, ...] | None]:
        getindex = self.model.getstage
        result: dict[str, tuple[str, ...] | None] = {self.index.name: tuple(self)}
        todo = list(reversed(self))
        while todo:
            current_name = todo.pop()
            if current_name in result:
                continue
            current_index = getindex(current_name)
            if current_index is None:
                result[current_name] = None
                continue
            bases = current_index.index_bases
            todo.extend(reversed(bases))
            result[current_name] = tuple(bases)
        return result

    @cached_property
    def bases(self) -> tuple[str]:
        """Returns bases as tuple of strings."""
        return tuple(self.index.ixconfig.get("bases", ()))

    def _get_project_inheritance_info(
        self, project: NormalizedName
    ) -> ProjectInheritanceInfo:
        blocked = set()
        policy = InheritancePolicy(index=self.index, project=project)
        seen = set()
        traversal_infos: list[tuple[TraversalInfo, bool | NotSet | Unknown]] = []
        for traversal_info in self.traversal_infos.steps:
            if not isinstance(traversal_info, TraversedIndex):
                traversal_infos.append((traversal_info, notset))
                continue
            if any(x in blocked for x in traversal_info.path):
                continue
            name = traversal_info.name
            (reason, exists) = policy.update(traversal_info)
            match reason:
                case PermissionDenied():
                    blocked.add(traversal_info.index)
                    traversal_infos.append(
                        (
                            traversal_info.block(reason=reason).with_seen(name in seen),
                            exists,
                        )
                    )
                case PermissionAllowed():
                    traversal_infos.append(
                        (
                            traversal_info.allow(reason=reason).with_seen(name in seen),
                            exists,
                        )
                    )
                case _:
                    traversal_infos.append(
                        (traversal_info.with_seen(name in seen), exists)
                    )
            seen.add(name)
        return ProjectInheritanceInfo(
            index_bases_map=self._index_bases_map,
            paths=self.traversal_infos.paths,
            project=project,
            traversal_infos=traversal_infos,
        )

    def get_project_inheritance_info(
        self, project: NormalizedName
    ) -> ProjectInheritanceInfo:
        result = self._per_project_mergability_cache.get(project)
        if result is None:
            result = self._get_project_inheritance_info(project)
            self._per_project_mergability_cache.put(project, result)
        return result

    def iter_mergeable_indexes(
        self, project: NormalizedName, opname: str
    ) -> Iterable[TraversedIndex]:
        return self.get_project_inheritance_info(project).iter_indexes(opname)

    @cached_property
    def inheritance_info(self) -> IndexInheritanceInfo:
        return IndexInheritanceInfo(
            index_bases_map=self._index_bases_map,
            paths=self.traversal_infos.paths,
            traversal_infos=self.traversal_infos.steps,
        )

    def iter_indexes(self) -> Iterator[TraversedIndex]:
        return self.inheritance_info.iter_indexes()

    def is_untrusted(self, index: BaseIndex) -> bool:
        # we have to postpone remotes, as there
        # may be private releases in other paths
        return index.index_type == "remote"

    @cached_property
    def traversal_infos(self) -> TraversalInfos:
        """Returns traversal information."""
        devpiserver_sro_skip = self.devpiserver_sro_skip
        getindex = self.model.getstage
        info: list[TraversalInfo] = []
        paths: list[TraversalPath] = []
        postponed: list[tuple[BaseIndex, list[str], TraversalPath]] = []
        seen_indexes: set[str] = set()
        seen_paths: set[PathKey] = set()
        is_untrusted = self.is_untrusted
        index = self.index
        todo: list[tuple[BaseIndex, list[str], TraversalPath | None]] = [
            (index, list(reversed(index.index_bases)), None)
        ]
        while todo:
            (current_index, bases, path) = todo[-1]
            if path is None:
                path = TraversalPath([x[0] for x in todo[:-1]])
            current_name = current_index.name
            path_key = path.join_index(current_index).path_key
            if bases or path_key not in seen_paths:
                info.append(
                    (
                        UntrustedTraversal
                        if is_untrusted(current_index)
                        else TraversedIndex
                    )(
                        index=current_index,
                        name=current_name,
                        path=path,
                        seen=current_name in seen_indexes,
                    )
                )
            seen_indexes.add(current_name)
            seen_paths.add(path_key)
            if not bases:
                paths.append(TraversalPath([*path, current_index]))
                todo.pop()
                if not todo:
                    todo.extend(postponed)
                    postponed.clear()
                continue
            next_name = bases.pop()
            if next_name in seen_indexes:
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=InheritanceCycle(name=next_name),
                        src=current_name,
                    )
                )
                continue
            next_index = getindex(next_name)
            if next_index is None:
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=MissingIndex(name=next_name),
                        src=current_name,
                    )
                )
                seen_indexes.add(next_name)
                continue
            if devpiserver_sro_skip(stage=index, base_stage=next_index):
                info.append(
                    SkippedTraversal(
                        name=next_name,
                        reason=SROSkipHook(name=next_name),
                        src=current_name,
                    )
                )
                seen_indexes.add(next_name)
                continue
            if is_untrusted(next_index):
                info.append(PostponedTraversal(name=next_name))
                postponed.append(
                    (
                        next_index,
                        list(reversed(next_index.index_bases)),
                        TraversalPath([x[0] for x in todo]),
                    )
                )
            else:
                todo.append((next_index, list(reversed(next_index.index_bases)), None))
        path_key_indices = {p.path_key: i for i, p in enumerate(paths)}
        # filter partial paths caused by postponed checks
        for path_len, path_key in sorted(
            ((len(p), p) for p in path_key_indices), reverse=True
        ):
            for i in range(path_len):
                path_key_indices.pop(PathKey(path_key[:i]), None)
        return TraversalInfos(
            paths=TraversalPaths(tuple(paths[i] for i in path_key_indices.values())),
            steps=info,
        )
