from __future__ import annotations

from ..config import SignalShopConfig

FUTURES_SIGNALS = "futures_signals"


class SignalShop:
    """Sells round-long subscriptions to signal products."""

    def __init__(self, cfg: SignalShopConfig, futures_enabled: bool) -> None:
        self.cfg = cfg
        self.futures_enabled = futures_enabled
        self.owners: dict[str, set[str]] = {FUTURES_SIGNALS: set()}

    def catalog(self, day: int) -> list[dict]:
        if not self.cfg.enabled or not self.futures_enabled or day > self.cfg.last_purchase_day:
            return []
        return [{"product": FUTURES_SIGNALS, "price": self.cfg.futures_signal_price_cents,
                 "last_purchase_day": self.cfg.last_purchase_day}]

    def price(self, product: str) -> int:
        if product == FUTURES_SIGNALS:
            return self.cfg.futures_signal_price_cents
        raise KeyError(product)

    def check(self, team: str, product: str, day: int) -> str | None:
        if not self.cfg.enabled:
            return "shop_closed"
        if product not in self.owners:
            return "unknown_product"
        if product == FUTURES_SIGNALS and not self.futures_enabled:
            return "unknown_product"
        if day > self.cfg.last_purchase_day:
            return "shop_closed"
        if team in self.owners[product]:
            return "already_owned"
        return None

    def purchase(self, team: str, product: str, day: int) -> tuple[str | None, int]:
        reason = self.check(team, product, day)
        if reason:
            return reason, 0
        self.owners[product].add(team)
        return None, self.price(product)

    def owns(self, team: str, product: str) -> bool:
        return team in self.owners.get(product, ())
