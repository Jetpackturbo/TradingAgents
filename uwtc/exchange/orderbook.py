"""Price-time priority limit order book on integer prices."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator

from sortedcontainers import SortedDict

from .types import Order, Side


class _Level:
    __slots__ = ("orders", "qty")

    def __init__(self) -> None:
        self.orders: OrderedDict[int, Order] = OrderedDict()
        self.qty = 0


class OrderBook:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        # Bids keyed by -price so that iteration order is best-first for both sides.
        self._bids: SortedDict = SortedDict()
        self._asks: SortedDict = SortedDict()
        self._orders: dict[int, Order] = {}

    def _side(self, side: Side) -> SortedDict:
        return self._bids if side is Side.BUY else self._asks

    @staticmethod
    def _key(side: Side, price: int) -> int:
        return -price if side is Side.BUY else price

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        return len(self._orders)

    def get(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    def add(self, order: Order) -> None:
        if order.remaining <= 0:
            raise ValueError("cannot rest an empty order")
        if order.order_id in self._orders:
            raise ValueError("duplicate order id")
        book = self._side(order.side)
        key = self._key(order.side, order.price)
        lvl = book.get(key)
        if lvl is None:
            lvl = _Level()
            book[key] = lvl
        lvl.orders[order.order_id] = order
        lvl.qty += order.remaining
        self._orders[order.order_id] = order

    def remove(self, order_id: int) -> Order | None:
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        book = self._side(order.side)
        key = self._key(order.side, order.price)
        lvl = book[key]
        del lvl.orders[order_id]
        lvl.qty -= order.remaining
        if not lvl.orders:
            del book[key]
        return order

    def reduce(self, order: Order, qty: int) -> None:
        """Reduce a resting order's remaining quantity after a fill."""
        if qty <= 0 or qty > order.remaining:
            raise ValueError("bad reduce quantity")
        lvl = self._side(order.side)[self._key(order.side, order.price)]
        order.remaining -= qty
        lvl.qty -= qty
        if order.remaining == 0:
            del lvl.orders[order.order_id]
            del self._orders[order.order_id]
            if not lvl.orders:
                del self._side(order.side)[self._key(order.side, order.price)]

    def best(self, side: Side) -> int | None:
        book = self._side(side)
        if not book:
            return None
        k = book.keys()[0]
        return -k if side is Side.BUY else k

    def best_bid(self) -> int | None:
        return self.best(Side.BUY)

    def best_ask(self) -> int | None:
        return self.best(Side.SELL)

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2

    def iter_side(self, side: Side) -> Iterator[Order]:
        """Iterate resting orders best-price first, then time priority."""
        book = self._side(side)
        for key in list(book.keys()):
            lvl = book.get(key)
            if lvl is None:
                continue
            for order in list(lvl.orders.values()):
                yield order

    def depth(self, side: Side, levels: int) -> list[tuple[int, int]]:
        out = []
        for key, lvl in self._side(side).items():
            if len(out) >= levels:
                break
            out.append((-key if side is Side.BUY else key, lvl.qty))
        return out

    def snapshot(self, levels: int) -> dict:
        return {
            "bids": [[p, q] for p, q in self.depth(Side.BUY, levels)],
            "asks": [[p, q] for p, q in self.depth(Side.SELL, levels)],
        }

    def orders(self) -> list[Order]:
        return list(self._orders.values())

    def is_crossed(self) -> bool:
        b, a = self.best_bid(), self.best_ask()
        return b is not None and a is not None and b >= a

    def check_integrity(self) -> None:
        """Assert internal consistency (used by tests)."""
        seen = 0
        for side in (Side.BUY, Side.SELL):
            for key, lvl in self._side(side).items():
                assert lvl.orders, "empty level"
                total = 0
                for o in lvl.orders.values():
                    assert o.side is side
                    assert self._key(side, o.price) == key
                    assert o.remaining > 0
                    assert self._orders[o.order_id] is o
                    total += o.remaining
                    seen += 1
                assert total == lvl.qty, "level qty mismatch"
        assert seen == len(self._orders)
        assert not self.is_crossed(), "book crossed"
