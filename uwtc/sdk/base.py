"""Strategy interface for competitors.

Subclass :class:`BaseStrategy` in ``competitor_strategy.py`` and override any
callbacks you need. Every callback receives a :class:`Context` which holds the
latest known state and exposes the trading actions.

All prices and cash values are INTEGER CENTS (``$101.25`` is ``10125``).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from . import protocol as P

log = logging.getLogger("uwtc.strategy")


@dataclass
class Book:
    symbol: str
    bids: list[list[int]] = field(default_factory=list)  # [[price, qty], ...] best first
    asks: list[list[int]] = field(default_factory=list)

    @property
    def best_bid(self) -> int | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> int | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return None

    @property
    def spread(self) -> int | None:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return None


@dataclass
class OpenOrder:
    client_id: str
    symbol: str
    side: str
    price: int | None
    qty: int
    remaining: int
    order_type: str
    order_id: int | None = None


class Context:
    """State + actions handed to every strategy callback."""

    def __init__(self, send: Callable[[dict], None], cid_prefix: str = "c") -> None:
        self._send = send
        self._cid = itertools.count(1)
        self._cid_prefix = cid_prefix
        self.team: str | None = None
        self.round_number: int | None = None
        self.phase: str = "waiting"
        self.day: int = 0
        self.days: int = 0
        self.time_ms: int = 0
        self.day_ms: int = 0
        self.cash: int = 0
        self.starting_cash: int = 0
        self.pnl: int = 0
        self.positions: dict[str, int] = {}
        self.open_orders: dict[str, OpenOrder] = {}  # keyed by client_id
        self.books: dict[str, Book] = {}
        self.last_trades: dict[str, int] = {}
        self.signals: dict[str, dict] = {}
        self.futures_signals: dict[str, dict] = {}
        self.commodities: dict[str, float] = {}
        self.inav: dict[str, float] = {}
        self.instruments: dict[str, dict] = {}
        self.tradable: set[str] = set()
        self.rules: dict[str, Any] = {}
        self.betas: dict[str, float] = {}
        self.offers: list[dict] = []
        self.futures_unlocked: bool = False
        self.premium: bool = False
        self.close_marks: dict[int, dict[str, int]] = {}
        self.user: dict[str, Any] = {}  # free space for strategy state

    # ------------------------------------------------------------- queries

    def book(self, symbol: str) -> Book:
        return self.books.get(symbol) or Book(symbol)

    def position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def orders_for(self, symbol: str | None = None) -> list[OpenOrder]:
        return [o for o in self.open_orders.values() if symbol is None or o.symbol == symbol]

    def limit(self, symbol: str) -> int:
        overrides = self.rules.get("position_limit_overrides", {})
        return overrides.get(symbol, self.rules.get("position_limit", 0))

    @property
    def ms_left_in_day(self) -> int:
        return max(0, self.day_ms - self.time_ms)

    # ------------------------------------------------------------- actions

    def place_order(self, symbol: str, side: str, qty: int, price: int | None = None,
                    order_type: str = "limit") -> str:
        """Submit an order. Returns the client id (use it to cancel)."""
        cid = f"{self._cid_prefix}{next(self._cid)}"
        msg = {"type": P.PLACE_ORDER, "symbol": symbol, "side": side, "qty": int(qty),
               "order_type": order_type, "client_id": cid}
        if price is not None:
            msg["price"] = int(price)
        self.open_orders[cid] = OpenOrder(cid, symbol, side, msg.get("price"), int(qty), int(qty), order_type)
        self._send(msg)
        return cid

    def buy(self, symbol: str, qty: int, price: int | None = None, order_type: str = "limit") -> str:
        if price is None and order_type == "limit":
            order_type = "market"
        return self.place_order(symbol, "buy", qty, price, order_type)

    def sell(self, symbol: str, qty: int, price: int | None = None, order_type: str = "limit") -> str:
        if price is None and order_type == "limit":
            order_type = "market"
        return self.place_order(symbol, "sell", qty, price, order_type)

    def cancel(self, client_id: str) -> bool:
        o = self.open_orders.get(client_id)
        if o is None or o.order_id is None:
            return False
        self._send({"type": P.CANCEL_ORDER, "order_id": o.order_id})
        return True

    def cancel_all(self, symbol: str | None = None) -> None:
        msg: dict = {"type": P.CANCEL_ALL}
        if symbol is not None:
            msg["symbol"] = symbol
        self._send(msg)

    def unlock_futures(self) -> None:
        self._send({"type": P.UNLOCK_FUTURES})

    def submit_auction_bid(self, amount_cents: int) -> None:
        self._send({"type": P.AUCTION_BID, "amount": int(amount_cents)})

    def buy_signal(self, product: str = P.FUTURES_SIGNALS) -> None:
        self._send({"type": P.BUY_SIGNAL, "product": product})

    def allocate_research(self, amount_cents: int) -> None:
        self._send({"type": P.ALLOCATE_RESEARCH, "amount": int(amount_cents)})

    def log(self, *args: Any) -> None:
        log.info(" ".join(str(a) for a in args))


class BaseStrategy:
    """Override any subset of these callbacks."""

    #: ``on_timer`` is called every ``timer_interval_ms`` of round time while trading.
    timer_interval_ms: int = 1000

    def on_round_start(self, ctx: Context, info: dict) -> None: ...
    def on_auction_open(self, ctx: Context, info: dict) -> None: ...
    def on_auction_result(self, ctx: Context, result: dict) -> None: ...
    def on_day_start(self, ctx: Context, day: int, signals: dict[str, dict]) -> None: ...
    def on_signal_offer(self, ctx: Context, offers: list[dict]) -> None: ...
    def on_signal(self, ctx: Context, signal: dict) -> None: ...
    def on_market_data(self, ctx: Context, data: dict) -> None: ...
    def on_trade(self, ctx: Context, trade: dict) -> None: ...
    def on_fill(self, ctx: Context, fill: dict) -> None: ...
    def on_order_update(self, ctx: Context, update: dict) -> None: ...
    def on_timer(self, ctx: Context) -> None: ...
    def on_day_end(self, ctx: Context, day: int, summary: dict) -> None: ...
    def on_round_end(self, ctx: Context, summary: dict) -> None: ...
    def on_error(self, ctx: Context, message: dict) -> None: ...
