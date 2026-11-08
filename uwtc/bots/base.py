"""House bot framework.

Bots live inside the exchange process and submit orders through the same
:class:`~uwtc.exchange.Exchange` path as teams (so all matching rules apply).
They are driven by the round engine on the simulated clock, which keeps every
run deterministic for a given round seed.
"""

from __future__ import annotations

import random
import zlib
from typing import TYPE_CHECKING, Any

from ..exchange.instruments import InstrumentKind
from ..exchange.types import Side

if TYPE_CHECKING:
    from ..engine.round import RoundEngine


class BotContext:
    def __init__(self, engine: "RoundEngine", account: str, seed: int) -> None:
        self.engine = engine
        self.account = account
        self.rng = random.Random(seed)

    # --------------------------------------------------------------- views
    @property
    def day(self) -> int:
        return self.engine.day

    @property
    def days(self) -> int:
        return self.engine.cfg.days

    @property
    def day_ms(self) -> int:
        return self.engine.cfg.day_ms

    @property
    def time_in_day(self) -> int:
        return self.engine.time_in_day

    @property
    def ms_to_close(self) -> int:
        return self.engine.cfg.day_ms - self.engine.time_in_day

    @property
    def tick(self) -> int:
        return self.engine.cfg.market.tick_cents

    @property
    def step(self) -> int:
        return self.engine.step

    def fair(self, symbol: str, lag_steps: int = 0) -> float:
        """Hidden fair value in dollars, ``lag_steps`` model steps ago (never
        before the start of the current day's open step minus the lag)."""
        return self.engine.fair(symbol, self.engine.step - lag_steps)

    def fair_cents(self, symbol: str, lag_steps: int = 0) -> int:
        return self.engine.exchange.round_tick(self.fair(symbol, lag_steps) * 100)

    def instruments(self, kind: InstrumentKind | None = None) -> list[str]:
        return [s for s, i in self.engine.instruments.items() if kind is None or i.kind is kind]

    def tradable(self, symbol: str) -> bool:
        return self.engine.exchange.tradable.get(symbol, False)

    def book(self, symbol: str):
        return self.engine.exchange.books[symbol]

    def mid(self, symbol: str) -> float | None:
        return self.engine.exchange.books[symbol].mid()

    def ref_mid(self, symbol: str) -> float:
        """Mid if both sides are quoted, else the last trade, else the previous
        close: never collapses to a stale reference when one side is swept."""
        m = self.mid(symbol)
        if m is not None:
            return m
        last = self.engine.exchange.last_price.get(symbol)
        return float(last) if last else float(self.engine.prev_close(symbol))

    def position(self, symbol: str) -> int:
        return self.engine.exchange.accounts[self.account].positions.get(symbol, 0)

    def open_orders(self, symbol: str | None = None) -> list:
        return [o for o in self.engine.exchange.accounts[self.account].open_orders.values()
                if symbol is None or o.symbol == symbol]

    def round_tick(self, price_cents: float) -> int:
        return self.engine.exchange.round_tick(price_cents)

    # --------------------------------------------------------------- actions
    def submit(self, symbol: str, side: Side | str, qty: int, price: int | None = None,
               otype: str = "limit") -> list[dict]:
        if qty <= 0 or not self.tradable(symbol):
            return []
        return self.engine.exchange.submit(self.account, symbol, side, int(qty), price, otype)

    def cancel(self, order_id: int) -> list[dict]:
        return self.engine.exchange.cancel(self.account, order_id)

    def cancel_all(self, symbol: str | None = None) -> list[dict]:
        if not self.open_orders(symbol):
            return []
        return self.engine.exchange.cancel_all(self.account, symbol)


class Bot:
    """Base class. ``interval_ms`` controls how often :meth:`on_tick` runs."""

    default_params: dict[str, Any] = {}

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{type(self).__name__} {name}: unknown params {sorted(unknown)}")
        self.name = name
        self.p = {**self.default_params, **params}
        self.interval_ms = int(self.p.get("interval_ms", 1000))
        self.ctx: BotContext | None = None

    def on_round_start(self, ctx: BotContext) -> None: ...
    def on_day_start(self, ctx: BotContext) -> None: ...
    def on_tick(self, ctx: BotContext) -> None: ...
    def on_day_end(self, ctx: BotContext) -> None: ...
    def on_event(self, ctx: BotContext, event: dict) -> None: ...


def bot_seed(round_seed: int, name: str) -> int:
    return (round_seed * 1_000_003) ^ zlib.crc32(name.encode())
