from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "limit"
    IOC = "ioc"
    MARKET = "market"


@dataclass(slots=True)
class Order:
    order_id: int
    account: str
    symbol: str
    side: Side
    price: int
    qty: int
    remaining: int
    ts: int
    otype: OrderType
    client_id: str | None = None
