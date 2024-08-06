from .markers import Absent
from .markers import Deleted
from .readonly import DictViewReadonly
from .readonly import SeqViewReadonly
from .readonly import SetViewReadonly
from sys import getsizeof


def _iter_dict(d):
    for k, v in d.items():
        yield k
        yield v


try:
    def gettotalsizeof(
            obj, maxlen=None,
            _default_size=getsizeof(0),
            _sequences=(frozenset, list, set, tuple, SetViewReadonly, SeqViewReadonly),
            _singles=(Absent, Deleted, bytes, complex, float, int, str, type(None)),  # noqa: B008
            _dicts=(dict, DictViewReadonly)):
        stack = [iter((obj,))]
        result = 0
        seen = set()
        while stack:
            try:
                item = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            if id(item) in seen:
                continue
            seen.add(id(item))
            result += getsizeof(item, _default_size)
            if maxlen and result >= maxlen:
                return None
            if isinstance(item, _singles):
                continue
            if isinstance(item, _sequences):
                if len(item):
                    stack.append(iter(item))
                continue
            if isinstance(item, _dicts):
                if len(item):
                    stack.append(_iter_dict(item))
                continue
            raise ValueError(f"don't know how to handle type {type(item)!r}")
        return result
except TypeError:
    # PyPy doesn't implement getsizeof, use dumplen as estimate
    from .fileutil import dumplen as gettotalsizeof  # type: ignore[assignment] # noqa: F401
