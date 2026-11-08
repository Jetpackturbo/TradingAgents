"""House bot behaviours. All parameters are configurable from ``config/bots.yaml``."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..exchange.instruments import InstrumentKind
from ..exchange.types import Side
from ..market.etf import index_nav, index_rebalance_shares, letf_rebalance_shares
from .base import Bot, BotContext


def resolve_symbols(ctx: BotContext, spec) -> list[str]:
    if isinstance(spec, list):
        return [s for s in spec if s in ctx.engine.instruments]
    groups = {
        "equities": [InstrumentKind.EQUITY],
        "etfs": [InstrumentKind.ETF, InstrumentKind.LETF],
        "futures": [InstrumentKind.FUTURE],
        "all": list(InstrumentKind),
        "spot": [InstrumentKind.EQUITY, InstrumentKind.ETF, InstrumentKind.LETF],
    }
    kinds = groups.get(spec)
    if kinds is None:
        raise ValueError(f"unknown symbol group {spec!r}")
    return [s for s, i in ctx.engine.instruments.items() if i.kind in kinds]


def _cross_price(ctx: BotContext, symbol: str, side: Side, slippage_bps: float) -> int | None:
    """Marketable limit price a few bps through the opposite best."""
    book = ctx.book(symbol)
    best = book.best_ask() if side is Side.BUY else book.best_bid()
    if best is None:
        return None
    adj = 1 + slippage_bps / 1e4 if side is Side.BUY else 1 - slippage_bps / 1e4
    return ctx.round_tick(best * adj)


class TargetExecutor:
    """Works towards a (re-estimated) total quantity per symbol in IOC slices.

    The target is recomputed by the caller on every slice from the latest
    prices, as real index / leveraged-fund desks size their close trades off
    the most recent move rather than a stale snapshot.
    """

    def __init__(self) -> None:
        self.done: dict[str, int] = {}
        self.slices_left = 0
        self.slices_total = 0
        self.last_targets: dict[str, int] = {}
        #: per-day (target, done) for diagnostics
        self.history: list[dict[str, tuple[int, int]]] = []

    def start(self, slices: int) -> None:
        self.done = {}
        self.slices_left = self.slices_total = max(1, slices)

    @property
    def active(self) -> bool:
        return self.slices_left > 0

    def step(self, ctx: BotContext, targets: dict[str, int], slippage_bps: float, ramp: float = 0.0) -> None:
        """``ramp`` > 0 widens the allowed slippage linearly up to
        ``slippage_bps * (1 + ramp)`` on the final slice (urgency into the close)."""
        if self.slices_left <= 0:
            return
        n = self.slices_left
        progress = (self.slices_total - n) / max(1, self.slices_total - 1)
        slip = slippage_bps * (1 + ramp * progress)
        self.last_targets = dict(targets)
        for sym, total in targets.items():
            rem = total - self.done.get(sym, 0)
            if rem == 0:
                continue
            child = int(math.ceil(abs(rem) / n)) * (1 if rem > 0 else -1)
            side = Side.BUY if child > 0 else Side.SELL
            px = _cross_price(ctx, sym, side, slip)
            if px is None:
                continue
            for ev in ctx.submit(sym, side, abs(child), px, "ioc"):
                if ev["type"] == "fill" and ev["account"] == ctx.account:
                    self.done[sym] = self.done.get(sym, 0) + ev["qty"] * side.sign
        self.slices_left -= 1
        if self.slices_left == 0:
            self.history.append({s: (t, self.done.get(s, 0)) for s, t in self.last_targets.items()})

    def clear(self) -> None:
        self.done = {}
        self.slices_left = 0


# --------------------------------------------------------------------------- #


class MarketMaker(Bot):
    """Quotes a ladder around a lagged, noisy view of fair value, skewed by inventory."""

    default_params = {
        "symbols": "all",
        "interval_ms": 500,
        "lag_steps": 2,
        "noise_bps": 3.0,
        "half_spread_bps": 8.0,
        "levels": 3,
        "level_spacing_bps": 6.0,
        "size": 25,
        "skew_bps_per_unit": 0.02,
        "max_inventory": 3000,
        "requote_ticks": 1,
        # Take other participants' quotes mispriced by more than this vs our ref
        # (keeps stale resting orders from pinning the book). <= 0 disables.
        "take_edge_bps": 20.0,
        "take_size": 100,
        # Off-book hedging of inventory above the threshold (an outside market).
        "hedge_threshold": 1000,
        "hedge_rate": 0.05,
    }

    def _hedge(self, ctx: BotContext, sym: str) -> None:
        p = self.p
        inv = ctx.position(sym)
        excess = abs(inv) - p["hedge_threshold"]
        if p["hedge_rate"] <= 0 or excess <= 0:
            return
        q = max(1, int(excess * p["hedge_rate"]))
        px = ctx.fair_cents(sym, p["lag_steps"])
        hedge = ctx.engine.bots.hedge_account
        if inv > 0:
            ctx.engine.exchange.block(hedge, ctx.account, sym, q, px)
        else:
            ctx.engine.exchange.block(ctx.account, hedge, sym, q, px)

    def on_round_start(self, ctx: BotContext) -> None:
        self.symbols = resolve_symbols(ctx, self.p["symbols"])

    def _ref(self, ctx: BotContext, sym: str) -> float:
        p = self.p
        return ctx.fair(sym, p["lag_steps"]) * 100 * (1 + ctx.rng.gauss(0, p["noise_bps"]) / 1e4)

    def _take(self, ctx: BotContext, sym: str, ref: float) -> None:
        p = self.p
        edge = p["take_edge_bps"] / 1e4
        if edge <= 0:
            return
        book = ctx.book(sym)
        inv = ctx.position(sym)
        bid, ask = book.best_bid(), book.best_ask()
        mine = {o.order_id for o in ctx.open_orders(sym)}
        if bid is not None and bid > ref * (1 + edge) and inv > -p["max_inventory"]:
            if not any(o.order_id in mine for o in book.iter_side(Side.BUY) if o.price == bid):
                ctx.submit(sym, Side.SELL, min(p["take_size"], p["max_inventory"] + inv),
                           ctx.round_tick(ref * (1 + edge)), "ioc")
        if ask is not None and ask < ref * (1 - edge) and inv < p["max_inventory"]:
            if not any(o.order_id in mine for o in book.iter_side(Side.SELL) if o.price == ask):
                ctx.submit(sym, Side.BUY, min(p["take_size"], p["max_inventory"] - inv),
                           ctx.round_tick(ref * (1 - edge)), "ioc")

    def _desired(self, ctx: BotContext, sym: str, ref: float) -> list[tuple[Side, int, int]]:
        p = self.p
        inv = ctx.position(sym)
        ref *= 1 - p["skew_bps_per_unit"] * inv / 1e4
        out = []
        t = ctx.tick
        for i in range(p["levels"]):
            off = (p["half_spread_bps"] + i * p["level_spacing_bps"]) / 1e4
            bid = int(math.floor(ref * (1 - off) / t)) * t
            ask = int(math.ceil(ref * (1 + off) / t)) * t
            if ask <= bid:
                ask = bid + t
            if inv < p["max_inventory"] and bid > 0:
                out.append((Side.BUY, bid, p["size"]))
            if inv > -p["max_inventory"]:
                out.append((Side.SELL, ask, p["size"]))
        return out

    def on_tick(self, ctx: BotContext) -> None:
        tol = self.p["requote_ticks"] * ctx.tick
        for sym in self.symbols:
            if not ctx.tradable(sym):
                continue
            self._hedge(ctx, sym)
            ref = self._ref(ctx, sym)
            self._take(ctx, sym, ref)
            desired = self._desired(ctx, sym, ref)
            current = ctx.open_orders(sym)
            if len(current) == len(desired):
                cur = sorted((o.side.value, o.price) for o in current)
                des = sorted((s.value, px) for s, px, _ in desired)
                if all(a[0] == b[0] and abs(a[1] - b[1]) < tol for a, b in zip(cur, des)) and \
                        all(o.remaining * 2 >= self.p["size"] for o in current):
                    continue
            ctx.cancel_all(sym)
            # Post-only: never cross other participants' resting quotes.
            book = ctx.book(sym)
            best_bid, best_ask = book.best_bid(), book.best_ask()
            placed: set[tuple[Side, int]] = set()
            for side, px, qty in desired:
                if side is Side.BUY and best_ask is not None and px >= best_ask:
                    px = best_ask - ctx.tick
                elif side is Side.SELL and best_bid is not None and px <= best_bid:
                    px = best_bid + ctx.tick
                if px <= 0 or (side, px) in placed:
                    continue
                placed.add((side, px))
                ctx.submit(sym, side, qty, px, "limit")


class RetailBuyer(Bot):
    """Naive retail flow: resting bids placed at random times each day."""

    default_params = {
        "symbols": "spot",
        "interval_ms": 1000,
        "orders_per_day": 6.0,
        "size_min": 5,
        "size_max": 40,
        "offset_bps_min": 0.0,
        "offset_bps_max": 25.0,
        "marketable_prob": 0.15,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        self.symbols = resolve_symbols(ctx, self.p["symbols"])

    def on_tick(self, ctx: BotContext) -> None:
        p = self.p
        prob = p["orders_per_day"] * self.interval_ms / ctx.day_ms
        for sym in self.symbols:
            if not ctx.tradable(sym) or ctx.rng.random() >= prob:
                continue
            qty = ctx.rng.randint(p["size_min"], p["size_max"])
            book = ctx.book(sym)
            if ctx.rng.random() < p["marketable_prob"] and book.best_ask() is not None:
                ctx.submit(sym, Side.BUY, qty, book.best_ask(), "ioc")
                continue
            ref = book.mid() or book.best_bid() or ctx.fair_cents(sym, 5)
            off = ctx.rng.uniform(p["offset_bps_min"], p["offset_bps_max"]) / 1e4
            ctx.submit(sym, Side.BUY, qty, ctx.round_tick(ref * (1 - off)), "limit")


class DayTrader(Bot):
    """Buys below / sells above an intraday moving average; flat by the close."""

    default_params = {
        "symbols": "equities",
        "interval_ms": 1000,
        "window": 15,
        "threshold_bps": 15.0,
        "size": 20,
        "max_inventory": 200,
        "flatten_ms": 5000,
        "slippage_bps": 5.0,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        self.symbols = resolve_symbols(ctx, self.p["symbols"])
        self.mids: dict[str, deque] = {s: deque(maxlen=self.p["window"]) for s in self.symbols}

    def on_day_start(self, ctx: BotContext) -> None:
        for d in self.mids.values():
            d.clear()

    def on_tick(self, ctx: BotContext) -> None:
        p = self.p
        flatten = ctx.ms_to_close <= p["flatten_ms"]
        for sym in self.symbols:
            if not ctx.tradable(sym):
                continue
            inv = ctx.position(sym)
            if flatten:
                if inv:
                    side = Side.SELL if inv > 0 else Side.BUY
                    px = _cross_price(ctx, sym, side, p["slippage_bps"] * 4)
                    if px:
                        ctx.submit(sym, side, abs(inv), px, "ioc")
                continue
            m = ctx.mid(sym)
            if m is None:
                continue
            hist = self.mids[sym]
            hist.append(m)
            if len(hist) < hist.maxlen:
                continue
            ma = sum(hist) / len(hist)
            th = p["threshold_bps"] / 1e4
            if m < ma * (1 - th) and inv + p["size"] <= p["max_inventory"]:
                px = _cross_price(ctx, sym, Side.BUY, p["slippage_bps"])
                if px:
                    ctx.submit(sym, Side.BUY, p["size"], px, "ioc")
            elif m > ma * (1 + th) and inv - p["size"] >= -p["max_inventory"]:
                px = _cross_price(ctx, sym, Side.SELL, p["slippage_bps"])
                if px:
                    ctx.submit(sym, Side.SELL, p["size"], px, "ioc")


class NoiseTrader(Bot):
    """Random marketable orders: uninformed volume."""

    default_params = {
        "symbols": "spot",
        "interval_ms": 1000,
        "orders_per_minute": 6.0,
        "size_min": 1,
        "size_max": 20,
        "slippage_bps": 10.0,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        self.symbols = resolve_symbols(ctx, self.p["symbols"])

    def on_tick(self, ctx: BotContext) -> None:
        p = self.p
        prob = p["orders_per_minute"] * self.interval_ms / 60_000
        for sym in self.symbols:
            if not ctx.tradable(sym) or ctx.rng.random() >= prob:
                continue
            side = Side.BUY if ctx.rng.random() < 0.5 else Side.SELL
            px = _cross_price(ctx, sym, side, p["slippage_bps"])
            if px:
                ctx.submit(sym, side, ctx.rng.randint(p["size_min"], p["size_max"]), px, "ioc")


class IndexRebalancer(Bot):
    """ETF sponsor for the index: arbitrages ETF vs basket iNAV during the day
    and, for an equal-weight index, rebalances the basket into the close."""

    default_params = {
        "interval_ms": 1000,
        "aum_dollars": 2_000_000.0,
        "arb_threshold_bps": 20.0,
        "arb_size": 20,
        "max_etf_inventory": 1000,
        "rebalance_window_ms": 10_000,
        "slices": 5,
        "slippage_bps": 10.0,
        "slippage_ramp": 0.0,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        ic = ctx.engine.cfg.instruments
        self.etf = ic.etf_symbol if ic.etf_enabled else None
        self.constituents = list(ic.equities)
        self.weighting = ic.etf_weighting
        self.exec = TargetExecutor()

    def on_day_start(self, ctx: BotContext) -> None:
        self.rebalanced = False
        self.exec.clear()

    def _rebalance_targets(self, ctx: BotContext) -> dict[str, int]:
        mids = [ctx.ref_mid(s) for s in self.constituents]
        opens = [ctx.engine.prev_close(s) for s in self.constituents]
        shares = index_rebalance_shares(self.p["aum_dollars"] * 100, np.array(opens), np.array(mids))
        return {s: int(round(q)) for s, q in zip(self.constituents, shares)}

    def _inav_cents(self, ctx: BotContext) -> float | None:
        mids = [ctx.mid(s) for s in self.constituents]
        if any(m is None for m in mids):
            return None
        opens = [ctx.engine.prev_close(s) for s in self.constituents]
        return index_nav(mids, opens, ctx.engine.prev_close(self.etf), self.weighting)

    def on_tick(self, ctx: BotContext) -> None:
        if self.etf is None:
            return
        p = self.p
        if self.weighting == "equal_weight_daily" and not self.rebalanced and \
                ctx.ms_to_close <= p["rebalance_window_ms"]:
            self.rebalanced = True
            self.exec.start(p["slices"])
        if self.exec.active:
            self.exec.step(ctx, self._rebalance_targets(ctx), p["slippage_bps"], p["slippage_ramp"])
        if not ctx.tradable(self.etf):
            return
        inav = self._inav_cents(ctx)
        if inav is None:
            return
        book = ctx.book(self.etf)
        inv = ctx.position(self.etf)
        th = p["arb_threshold_bps"] / 1e4
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and bid > inav * (1 + th) and inv - p["arb_size"] >= -p["max_etf_inventory"]:
            ctx.submit(self.etf, Side.SELL, p["arb_size"], bid, "ioc")
        elif ask is not None and ask < inav * (1 - th) and inv + p["arb_size"] <= p["max_etf_inventory"]:
            ctx.submit(self.etf, Side.BUY, p["arb_size"], ask, "ioc")


class LETFRebalancer(Bot):
    """Daily-reset leveraged ETF sponsor: trades the underlying into the close
    to restore target leverage (buys after up days, sells after down days)."""

    default_params = {
        "interval_ms": 1000,
        "aum_dollars": 3_000_000.0,
        "rebalance_window_ms": 10_000,
        "slices": 5,
        "slippage_bps": 10.0,
        "slippage_ramp": 0.0,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        ic = ctx.engine.cfg.instruments
        self.enabled = ic.letf_enabled
        self.underlying = ic.letf_underlying
        self.leverage = ic.letf_leverage
        self.aum_open = self.p["aum_dollars"] * 100
        self.exec = TargetExecutor()

    def on_day_start(self, ctx: BotContext) -> None:
        self.started = False
        self.exec.clear()

    def target_qty(self, ctx: BotContext) -> int:
        p0 = ctx.engine.prev_close(self.underlying)
        p1 = ctx.ref_mid(self.underlying)
        return int(round(letf_rebalance_shares(self.aum_open, self.leverage, p0, p1)))

    def on_tick(self, ctx: BotContext) -> None:
        if not self.enabled:
            return
        p = self.p
        if not self.started and ctx.ms_to_close <= p["rebalance_window_ms"]:
            self.started = True
            self.exec.start(p["slices"])
        if self.exec.active:
            self.exec.step(ctx, {self.underlying: self.target_qty(ctx)}, p["slippage_bps"], p["slippage_ramp"])

    def on_day_end(self, ctx: BotContext) -> None:
        if not self.enabled:
            return
        p0 = ctx.engine.prev_close(self.underlying)
        p1 = ctx.engine.fair(self.underlying, ctx.engine.step) * 100
        self.aum_open *= 1 + self.leverage * (p1 / p0 - 1)


class FuturesArb(Bot):
    """Keeps futures within a band of the underlying's mid (zero carry)."""

    default_params = {
        "interval_ms": 1000,
        "band_bps": 15.0,
        "size": 10,
        "max_inventory": 300,
    }

    def on_round_start(self, ctx: BotContext) -> None:
        self.futures = resolve_symbols(ctx, "futures")

    def on_tick(self, ctx: BotContext) -> None:
        p = self.p
        th = p["band_bps"] / 1e4
        for f in self.futures:
            if not ctx.tradable(f):
                continue
            und = ctx.engine.instruments[f].underlying
            m = ctx.mid(und)
            if m is None:
                continue
            book = ctx.book(f)
            inv = ctx.position(f)
            bid, ask = book.best_bid(), book.best_ask()
            if bid is not None and bid > m * (1 + th) and inv - p["size"] >= -p["max_inventory"]:
                ctx.submit(f, Side.SELL, p["size"], bid, "ioc")
            elif ask is not None and ask < m * (1 - th) and inv + p["size"] <= p["max_inventory"]:
                ctx.submit(f, Side.BUY, p["size"], ask, "ioc")
