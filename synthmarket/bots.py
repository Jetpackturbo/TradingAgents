"""synthmarket bots, registered into ``uwtc.bots.registry.BOT_TYPES`` at import.

* ``flow_market_maker`` (:class:`FlowMarketMaker`): learns its reference price only from
  the public trade tape (price, qty, aggressor side) and its own inventory.
* ``target_trader`` (:class:`TargetTrader`): informed trader that sees the target schedule.
* ``resting_drift_trader`` (:class:`RestingDriftTrader`): posts never-cancelled resting limits
  around the target schedule; on its own it forms the simplest drift world.
"""

from __future__ import annotations

import math

from uwtc.bots.base import Bot, BotContext
from uwtc.bots.behaviors import resolve_symbols
from uwtc.bots.registry import BOT_TYPES
from uwtc.exchange.types import Side


class FlowMarketMaker(Bot):
    """Ladder quoted around a reference learned from public order flow.

    For every new public trade (price ``p``, size ``q``, aggressor sign ``s``)::

        ref <- ref + ewma_alpha * (p - ref)                       # follow prints
        ref <- ref * (1 + s * min(kyle_bps_per_unit * q, kyle_max_bps) / 1e4)   # Kyle impact

    Quotes are centred on ``ref * (1 - skew_bps_per_unit * inventory / 1e4)``. There is no
    off-book hedging and the bot never reads the target schedule.

    Combining ``ewma_alpha > 0`` with inventory skew makes ``ref`` ratchet towards its own
    skewed quotes (drift proportional to inventory, which noise flow does not revert), so
    prices become trending even with no information in the market. Default is pure Kyle.
    """

    default_params = {
        "symbols": "equities",
        "interval_ms": 200,
        "ewma_alpha": 0.0,
        "kyle_bps_per_unit": 0.3,
        "kyle_max_bps": 30.0,
        "half_spread_bps": 5.0,
        "levels": 3,
        "level_spacing_bps": 4.0,
        "size": 20,
        "skew_bps_per_unit": 0.02,
        "max_skew_bps": 50.0,
        "max_inventory": 1_000_000,
        "requote_ticks": 1,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        self.symbols = resolve_symbols(ctx, self.p["symbols"])
        self.ref = {s: float(ctx.engine.public_mark(s)) for s in self.symbols}
        self._cursor = 0

    def _learn(self, ctx: BotContext) -> None:
        p = self.p
        log = ctx.engine.trade_log
        n = len(log)
        if self._cursor >= n:
            return
        a, k, kmax = p["ewma_alpha"], p["kyle_bps_per_unit"], p["kyle_max_bps"]
        for i in range(self._cursor, n):
            tr = log[i]
            r = self.ref.get(tr["symbol"])
            if r is None:
                continue
            r += a * (tr["price"] - r)
            imp = min(k * tr["qty"], kmax) / 1e4
            r *= 1 + imp if tr["aggressor"] == "buy" else 1 - imp
            self.ref[tr["symbol"]] = r
        self._cursor = n

    def _desired(self, ctx: BotContext, sym: str) -> list[tuple[Side, int, int]]:
        p = self.p
        inv = ctx.position(sym)
        skew = max(-p["max_skew_bps"], min(p["max_skew_bps"], p["skew_bps_per_unit"] * inv))
        center = self.ref[sym] * (1 - skew / 1e4)
        t = ctx.tick
        out = []
        for i in range(p["levels"]):
            off = (p["half_spread_bps"] + i * p["level_spacing_bps"]) / 1e4
            bid = int(math.floor(center * (1 - off) / t)) * t
            ask = int(math.ceil(center * (1 + off) / t)) * t
            if ask <= bid:
                ask = bid + t
            if inv < p["max_inventory"] and bid > 0:
                out.append((Side.BUY, bid, p["size"]))
            if inv > -p["max_inventory"]:
                out.append((Side.SELL, ask, p["size"]))
        return out

    def on_tick(self, ctx: BotContext) -> None:
        self._learn(ctx)
        tol = self.p["requote_ticks"] * ctx.tick
        for sym in self.symbols:
            if not ctx.tradable(sym):
                continue
            desired = self._desired(ctx, sym)
            current = ctx.open_orders(sym)
            if len(current) == len(desired):
                cur = sorted((o.side.value, o.price) for o in current)
                des = sorted((s.value, px) for s, px, _ in desired)
                if all(a[0] == b[0] and abs(a[1] - b[1]) < tol for a, b in zip(cur, des)) and \
                        all(o.remaining * 2 >= self.p["size"] for o in current):
                    continue
            ctx.cancel_all(sym)
            book = ctx.book(sym)
            best_bid, best_ask = book.best_bid(), book.best_ask()
            placed: set[tuple[Side, int]] = set()
            for side, px, qty in desired:
                # Post-only: never cross other participants' resting quotes.
                if side is Side.BUY and best_ask is not None and px >= best_ask:
                    px = best_ask - ctx.tick
                elif side is Side.SELL and best_bid is not None and px <= best_bid:
                    px = best_bid + ctx.tick
                if px <= 0 or (side, px) in placed:
                    continue
                placed.add((side, px))
                ctx.submit(sym, side, qty, px, "limit")


class TargetTrader(Bot):
    """Informed trader: arrives with probability ``arrivals_per_bar * interval_ms / bar_ms``
    per tick and per symbol, observes ``F[bar - lag_steps] * exp(N(0, target_noise_bps))``
    and sends an IOC limit capped at ``F * (1 -/+ edge)`` when the book is through it."""

    default_params = {
        "symbols": "equities",
        "interval_ms": 100,
        "arrivals_per_bar": 0.3,
        "size_min": 10,
        "size_max": 30,
        "edge_bps": 5.0,
        "lag_steps": 1,
        "target_noise_bps": 0.0,
        "side": "both",
    }

    def on_round_start(self, ctx: BotContext) -> None:
        if self.p["side"] not in ("both", "buy", "sell"):
            raise ValueError(f"TargetTrader {self.name}: side must be both/buy/sell")
        self.symbols = resolve_symbols(ctx, self.p["symbols"])
        self.view = ctx.engine.target_view(self)
        self.prob = self.p["arrivals_per_bar"] * self.interval_ms / ctx.engine.bar_ms

    def on_tick(self, ctx: BotContext) -> None:
        p = self.p
        rng = ctx.rng
        eng = ctx.engine
        for sym in self.symbols:
            if not ctx.tradable(sym) or rng.random() >= self.prob:
                continue
            f = self.view.value(sym, eng.current_bar() - p["lag_steps"]) * 100.0
            if p["target_noise_bps"] > 0:
                f *= math.exp(rng.gauss(0.0, p["target_noise_bps"]) / 1e4)
            qty = rng.randint(p["size_min"], p["size_max"])
            e = p["edge_bps"] / 1e4
            t = ctx.tick
            book = ctx.book(sym)
            ask, bid = book.best_ask(), book.best_bid()
            if p["side"] != "sell" and ask is not None and ask < f * (1 - e):
                ctx.submit(sym, Side.BUY, qty, int(math.floor(f * (1 - e) / t)) * t, "ioc")
            elif p["side"] != "buy" and bid is not None and bid > f * (1 + e):
                ctx.submit(sym, Side.SELL, qty, int(math.ceil(f * (1 + e) / t)) * t, "ioc")


class RestingDriftTrader(TargetTrader):
    """Simplest price former: posts resting limit orders around its internal target and never cancels.

    Arrives with probability ``arrivals_per_bar * interval_ms / bar_ms`` per tick and per symbol,
    picks a side, and posts a limit at ``F * (1 + u)`` (buy) or ``F * (1 - u)`` (sell) with
    ``u ~ U(-depth_bps, cross_bps) / 1e4``: most orders rest behind F, some reach through it and
    trade against older resting orders. With a drifting F this is the whole price process.
    """

    default_params = {
        "symbols": "equities",
        "interval_ms": 100,
        "arrivals_per_bar": 0.5,
        "size_min": 1,
        "size_max": 10,
        "depth_bps": 20.0,
        "cross_bps": 10.0,
        "lag_steps": 0,
        "side": "both",
    }

    def on_tick(self, ctx: BotContext) -> None:
        p = self.p
        rng = ctx.rng
        eng = ctx.engine
        t = ctx.tick
        for sym in self.symbols:
            if not ctx.tradable(sym) or rng.random() >= self.prob:
                continue
            f = self.view.value(sym, eng.current_bar() - p["lag_steps"]) * 100.0
            u = rng.uniform(-p["depth_bps"], p["cross_bps"]) / 1e4
            buy = p["side"] == "buy" or (p["side"] == "both" and rng.random() < 0.5)
            qty = rng.randint(p["size_min"], p["size_max"])
            if buy:
                ctx.submit(sym, Side.BUY, qty, max(t, int(math.floor(f * (1 + u) / t)) * t), "limit")
            else:
                ctx.submit(sym, Side.SELL, qty, max(t, int(math.ceil(f * (1 - u) / t)) * t), "limit")


BOT_TYPES.setdefault("flow_market_maker", FlowMarketMaker)
BOT_TYPES.setdefault("target_trader", TargetTrader)
BOT_TYPES.setdefault("resting_drift_trader", RestingDriftTrader)
