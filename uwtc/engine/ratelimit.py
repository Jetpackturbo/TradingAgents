from __future__ import annotations


class TokenBucket:
    """Token bucket on an externally supplied millisecond clock."""

    __slots__ = ("rate", "capacity", "tokens", "last_ms", "violations")

    def __init__(self, rate_per_s: float, capacity: int) -> None:
        self.rate = rate_per_s / 1000.0
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_ms: float | None = None
        self.violations = 0

    def allow(self, now_ms: float, cost: float = 1.0) -> bool:
        if self.last_ms is None:
            self.last_ms = now_ms
        elif now_ms > self.last_ms:
            self.tokens = min(self.capacity, self.tokens + (now_ms - self.last_ms) * self.rate)
            self.last_ms = now_ms
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        self.violations += 1
        return False
