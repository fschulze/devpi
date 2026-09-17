import time


class Timeout:
    default: float
    _limit: float | None = None
    _started_at: float

    def __init__(self, default: float) -> None:
        if default <= 0:
            raise ValueError("Timeout default must be positive")
        self.default = default

    @property
    def limit(self) -> float:
        return self.default if self._limit is None else self._limit

    @limit.setter
    def limit(self, value: float) -> None:
        self._limit = value if self._limit is None else min(self._limit, value)

    @property
    def remaining(self) -> float | None:
        if not hasattr(self, "_started_at"):
            raise RuntimeError("Timeout hasn't been started")
        delta = time.monotonic() - self._started_at
        return max(0, self.limit - delta)

    def start(self) -> None:
        if hasattr(self, "_started_at"):
            raise RuntimeError("Can't restart Timeout")
        self._started_at = time.monotonic()
