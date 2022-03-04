from .readonly import DictViewReadonly
from .readonly import SeqViewReadonly
from .readonly import SetViewReadonly
from .readonly import _immutable
from sys import getsizeof


try:
    def gettotalsizeof(
            obj, maxlen=None,
            _default_size=getsizeof(0),
            _sequences=(frozenset, list, set, tuple, SetViewReadonly, SeqViewReadonly),
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
                return
            if isinstance(item, _immutable):
                continue
            elif isinstance(item, _sequences):
                stack.append(iter(item))
                continue
            elif isinstance(item, _dicts):
                stack.append(iter(item.items()))
                continue
            raise ValueError(f"don't know how to handle type {type(item)!r}")
        return result
except TypeError:
    # PyPy doesn't implement getsizeof, use dumplen as estimate
    from .fileutil import dumplen as gettotalsizeof  # noqa
