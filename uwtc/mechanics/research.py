"""Research allocation: spend cash to slow / offset signal decay.

Spend committed during day ``d`` takes effect from day ``d + 1`` onwards::

    boost(R) = max_boost * (1 - exp(-R / scale))

which is increasing and concave (diminishing returns).
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..config import ResearchConfig


def research_boost(spend_cents: int, cfg: ResearchConfig) -> float:
    if spend_cents <= 0:
        return 0.0
    return cfg.max_boost * (1.0 - math.exp(-spend_cents / cfg.scale_cents))


class ResearchLedger:
    def __init__(self, cfg: ResearchConfig) -> None:
        self.cfg = cfg
        self.effective: dict[str, int] = defaultdict(int)
        self.pending: dict[str, int] = defaultdict(int)

    def total(self, team: str) -> int:
        return self.effective[team] + self.pending[team]

    def allocate(self, team: str, amount: int) -> str | None:
        if not self.cfg.enabled:
            return "research_disabled"
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            return "bad_amount"
        if self.total(team) + amount > self.cfg.max_total_cents:
            return "research_cap"
        self.pending[team] += amount
        return None

    def commit_day(self) -> None:
        """Called at day start: pending spend becomes effective."""
        for team, amt in self.pending.items():
            self.effective[team] += amt
        self.pending.clear()

    def boost(self, team: str) -> float:
        return research_boost(self.effective[team], self.cfg)
