from __future__ import annotations


class FuturesAccess:
    """Per-round futures 'connection fee'."""

    def __init__(self, enabled: bool, fee_cents: int) -> None:
        self.enabled = enabled
        self.fee_cents = fee_cents
        self.unlocked: set[str] = set()

    def is_unlocked(self, team: str) -> bool:
        return team in self.unlocked

    def unlock(self, team: str) -> tuple[str | None, int]:
        if not self.enabled:
            return "futures_disabled", 0
        if team in self.unlocked:
            return "already_unlocked", 0
        self.unlocked.add(team)
        return None, self.fee_cents
