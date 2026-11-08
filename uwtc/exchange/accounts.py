from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .types import Order, Side


@dataclass
class Account:
    account_id: str
    kind: str  # "team" | "bot" | "house"
    cash: int = 0
    starting_cash: int = 0
    risk_exempt: bool = False
    positions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    permissions: set[str] = field(default_factory=set)
    open_orders: dict[int, Order] = field(default_factory=dict)
    open_buy_qty: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    open_sell_qty: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fees_paid: int = 0
    # Cash charges by reason (auction, research, futures_unlock, signal_shop, ...).
    charges: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    volume: int = 0
    trades: int = 0

    def track_open(self, order: Order) -> None:
        self.open_orders[order.order_id] = order
        if order.side is Side.BUY:
            self.open_buy_qty[order.symbol] += order.remaining
        else:
            self.open_sell_qty[order.symbol] += order.remaining

    def reduce_open(self, order: Order, qty: int) -> None:
        if order.side is Side.BUY:
            self.open_buy_qty[order.symbol] -= qty
        else:
            self.open_sell_qty[order.symbol] -= qty

    def untrack(self, order: Order) -> None:
        if self.open_orders.pop(order.order_id, None) is not None:
            self.reduce_open(order, order.remaining)

    def value(self, marks: dict[str, int]) -> int:
        total = self.cash
        for sym, q in self.positions.items():
            if q:
                total += q * marks[sym]
        return total

    def pnl(self, marks: dict[str, int]) -> int:
        return self.value(marks) - self.starting_cash

    def gross_notional(self, marks: dict[str, int]) -> int:
        return sum(abs(q) * marks.get(s, 0) for s, q in self.positions.items() if q)

    def snapshot(self, marks: dict[str, int] | None = None) -> dict:
        out = {
            "account": self.account_id,
            "cash": self.cash,
            "positions": {s: q for s, q in self.positions.items() if q},
            "fees_paid": self.fees_paid,
            "charges": {k: v for k, v in self.charges.items() if v},
            "open_orders": len(self.open_orders),
            "volume": self.volume,
        }
        if marks is not None:
            out["pnl"] = self.pnl(marks)
        return out
