"""Round engine: drives one 15-day round on a simulated millisecond clock.

The engine is single-threaded and deterministic. It never reads the wall clock;
callers advance it with :meth:`RoundEngine.advance` and deliver team requests
with :meth:`RoundEngine.handle`. Outbound team messages go to ``sink(team, msg)``.

Timeline (round time, ms)::

    0 ............ pre_open_ms      PRE_OPEN (auction window)
    day d start = pre_open_ms + (d-1) * (day_ms + day_gap_ms)
    day d end   = start + day_ms    EOD processing, then gap, then next day
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

from ..bots import BotManager
from ..config import MarkMethod, RoundConfig
from ..exchange import Exchange, build_instruments
from ..exchange.instruments import FUTURES_PERMISSION, InstrumentKind
from ..market import SignalEngine, generate_paths
from ..mechanics import FuturesAccess, ResearchLedger, SealedBidAuction, SignalShop
from ..sdk import protocol as P

log = logging.getLogger("uwtc.round")

Sink = Callable[[str, dict], None]

_EVENT_MAP = {
    "ack": P.ORDER_ACK,
    "reject": P.ORDER_REJECT,
    "fill": P.FILL,
    "cancelled": P.ORDER_CANCELLED,
    "cancel_reject": P.CANCEL_REJECT,
    "settlement": P.SETTLEMENT,
    "cash": P.CASH,
}
# Rule keys disclosed to teams (everything else in MarketConfig stays hidden).
_PUBLIC_RISK_KEYS = ("max_order_qty", "max_open_orders", "position_limit", "position_limit_overrides",
                     "max_gross_notional_cents", "msgs_per_second", "msg_burst")
_LOGGED_TYPES = {P.ORDER_ACK, P.ORDER_REJECT, P.FILL, P.ORDER_CANCELLED, P.CANCEL_REJECT, P.SETTLEMENT,
                 P.CASH, P.FUTURES_UNLOCKED, P.SIGNAL, P.REQUEST_REJECT, P.DAY_START, P.DAY_END,
                 P.AUCTION_RESULT, P.ROUND_END}


class Phase(str, Enum):
    INIT = "init"
    PRE_OPEN = "pre_open"
    TRADING = "trading"
    EOD = "eod"
    CLOSED = "closed"


class RoundEngine:
    def __init__(self, cfg: RoundConfig, teams: list[str], sink: Sink | None = None) -> None:
        if len(set(teams)) != len(teams):
            raise ValueError("duplicate team ids")
        for t in teams:
            if not isinstance(t, str) or not t or t.startswith("bot:") or t.startswith("__"):
                raise ValueError(f"invalid team id {t!r}")
        g = cfg.day_ms + cfg.market.day_gap_ms
        if cfg.pre_open_ms % cfg.tick_ms or cfg.market.day_gap_ms % cfg.tick_ms:
            raise ValueError("pre_open_ms and day_gap_ms must be multiples of tick_ms")
        self.cfg = cfg
        self.teams = list(teams)
        self.sink: Sink = sink or (lambda team, msg: None)
        self.paths = generate_paths(cfg)
        self.signal_engine = SignalEngine(cfg, self.paths)
        self.instruments = build_instruments(cfg)
        self.exchange = Exchange(cfg.market, self.instruments)
        self.exchange.listeners.append(self._on_exchange_events)
        mc = cfg.mechanics
        self.auction = SealedBidAuction(mc.auction, seed=cfg.seed,
                                        available_cash=lambda t: self.exchange.accounts[t].cash)
        self.research = ResearchLedger(mc.research)
        self.shop = SignalShop(mc.signal_shop, cfg.instruments.futures_enabled)
        self.futures_access = FuturesAccess(cfg.instruments.futures_enabled,
                                            cfg.instruments.futures_unlock_fee_cents)
        self.bots = BotManager(self, cfg.bots)

        self.day_starts = {d: cfg.pre_open_ms + (d - 1) * g for d in range(1, cfg.days + 1)}
        self.day_ends = {d: s + cfg.day_ms for d, s in self.day_starts.items()}
        self.end_ms = self.day_ends[cfg.days]

        self.phase = Phase.INIT
        self.day = 0
        self.now = 0
        self.step = 0
        self._next_tick = 0
        self._next_md = 0
        self._next_acct = 0
        self.tier: dict[str, str] = {t: "base" for t in teams}
        self.close_marks: dict[int, dict[str, int]] = {}
        self._prev_close: dict[str, int] = {}
        self._pending_trades: list[dict] = []
        self.trade_log: list[dict] = []
        self._team_set = set(teams)
        self.team_log: dict[str, list[dict]] = defaultdict(list)
        self.pnl_curve: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        self.daily_pnl: dict[str, list[int]] = defaultdict(list)
        self.results: dict[str, dict] | None = None
        self.counters: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Views                                                               #
    # ------------------------------------------------------------------ #

    @property
    def time_in_day(self) -> int:
        if self.day < 1:
            return 0
        return min(self.cfg.day_ms, max(0, self.now - self.day_starts[self.day]))

    def fair(self, symbol: str, step: int | None = None) -> float:
        step = self.step if step is None else step
        if self.day >= 1:
            step = max(step, 0)
        inst = self.instruments.get(symbol)
        if inst is not None and inst.kind is InstrumentKind.FUTURE:
            symbol = inst.underlying
        return self.paths.price(symbol, step)

    def fair_cents(self, symbol: str, step: int | None = None) -> int:
        return self.exchange.round_tick(self.fair(symbol, step) * 100)

    def prev_close(self, symbol: str) -> int:
        """Official close of the previous day (cents); fair open for day 1."""
        if symbol in self._prev_close:
            return self._prev_close[symbol]
        return self.fair_cents(symbol, 0)

    def public_mark(self, symbol: str) -> int:
        ex = self.exchange
        m = ex.books[symbol].mid()
        if m is not None:
            return ex.round_tick(m)
        if symbol in ex.last_price:
            return ex.last_price[symbol]
        return ex.ref_price.get(symbol) or self.prev_close(symbol)

    def public_marks(self) -> dict[str, int]:
        return {s: self.public_mark(s) for s in self.instruments}

    def listed_futures(self, day: int) -> list[tuple[str, str, int]]:
        return [(s, i.underlying, i.expiry_day) for s, i in self.instruments.items()
                if i.kind is InstrumentKind.FUTURE and i.is_listed(day) and s not in self.exchange.delisted]

    def public_rules(self) -> dict:
        m = self.cfg.market
        rules: dict[str, Any] = {k: getattr(m.risk, k) for k in _PUBLIC_RISK_KEYS}
        rules.update({
            "tick_cents": m.tick_cents,
            "lot_size": m.lot_size,
            "maker_fee_cents": m.fees.maker_cents_per_unit,
            "taker_fee_cents": m.fees.taker_cents_per_unit,
            "cancel_on_eod": m.cancel_on_eod,
            "futures_unlock_fee_cents": self.cfg.instruments.futures_unlock_fee_cents,
            "letf_leverage": self.cfg.instruments.letf_leverage,
            "etf_weighting": self.cfg.instruments.etf_weighting,
            "book_publish_ms": m.market_data.publish_ms,
        })
        return rules

    # ------------------------------------------------------------------ #
    # Outbound                                                            #
    # ------------------------------------------------------------------ #

    def _emit(self, team: str, msg: dict) -> None:
        msg.setdefault("day", self.day)
        msg.setdefault("time_ms", self.time_in_day)
        if msg["type"] in _LOGGED_TYPES:
            self.team_log[team].append(msg)
        self.sink(team, msg)

    def _broadcast(self, msg_factory: Callable[[str], dict]) -> None:
        for t in self.teams:
            self._emit(t, msg_factory(t))

    def _on_exchange_events(self, events: list[dict]) -> None:
        for ev in events:
            et = ev["type"]
            if et == "trade":
                self._pending_trades.append(ev)
                self.trade_log.append(ev)
                self.counters["trades"] += 1
                continue
            acct = ev.get("account")
            if acct is None:
                continue
            if acct in self._team_set:
                mtype = _EVENT_MAP.get(et)
                if mtype is None:
                    continue
                msg = {k: v for k, v in ev.items() if k not in ("account", "type", "seq", "ts")}
                msg["type"] = mtype
                self._emit(acct, msg)
            else:
                self.bots.on_events(acct, ev)

    def _request_reject(self, team: str, request: str, reason: str, **extra) -> None:
        self._emit(team, {"type": P.REQUEST_REJECT, "request": request, "reason": reason, **extra})

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self.phase is not Phase.INIT:
            raise RuntimeError("round already started")
        cash = self.cfg.market.starting_cash_cents
        for t in self.teams:
            self.exchange.add_account(t, "team", cash=cash, ts=0)
        self.bots.setup()
        self.phase = Phase.PRE_OPEN
        self._set_refs()
        info = {
            "type": P.ROUND_START,
            "round": self.cfg.round_number,
            "name": self.cfg.name,
            "days": self.cfg.days,
            "day_ms": self.cfg.day_ms,
            "pre_open_ms": self.cfg.pre_open_ms,
            "day_gap_ms": self.cfg.market.day_gap_ms,
            "starting_cash": cash,
            "instruments": [i.public() for i in self.instruments.values()],
            "betas": {s: round(b, 4) for s, b in self.paths.betas_vs_index.items()},
            "rules": self.public_rules(),
            "mechanics": {
                "auction": self.cfg.mechanics.auction.enabled,
                "signal_shop": self.cfg.mechanics.signal_shop.enabled,
                "research": self.cfg.mechanics.research.enabled,
                "futures": self.cfg.instruments.futures_enabled,
            },
            "prev_close": {s: self.prev_close(s) for s in self.instruments},
        }
        self._broadcast(lambda t: dict(info))
        if self.cfg.mechanics.auction.enabled:
            self.auction.open()
            a = self.cfg.mechanics.auction
            self._broadcast(lambda t: {"type": P.AUCTION_OPEN, "winners": a.winners,
                                       "payment_rule": a.payment_rule.value, "reserve": a.reserve_cents,
                                       "max_bid": a.max_bid_cents, "closes_at_ms": self.cfg.pre_open_ms,
                                       "product": "premium_signals"})
        self._next_tick = self.cfg.tick_ms

    def advance(self, to_ms: int) -> None:
        """Run all scheduled work with timestamps <= ``to_ms``."""
        if self.phase is Phase.INIT:
            raise RuntimeError("call start() first")
        while self.phase is not Phase.CLOSED and self._next_tick <= to_ms:
            t = self._next_tick
            self._next_tick += self.cfg.tick_ms
            self._tick(t)
        if self.phase is not Phase.CLOSED:
            self.now = max(self.now, min(to_ms, self._next_tick - 1))

    def run_to_end(self) -> None:
        self.advance(self.end_ms)

    @property
    def closed(self) -> bool:
        return self.phase is Phase.CLOSED

    def _tick(self, t: int) -> None:
        self.now = t
        nxt = self.day + 1
        if self.phase is Phase.TRADING and t >= self.day_ends[self.day]:
            self._end_of_day()
            if self.day == self.cfg.days:
                self._close_round()
                return
        if self.phase in (Phase.PRE_OPEN, Phase.EOD) and nxt <= self.cfg.days and t >= self.day_starts[nxt]:
            self._start_day(nxt)
        if self.phase is not Phase.TRADING:
            return
        k = (t - self.day_starts[self.day]) // self.cfg.step_ms
        step = self.paths.open_step(self.day) + min(k, self.cfg.model.steps_per_day - 1)
        if step != self.step:
            self.step = step
            self._set_refs()
        self.bots.tick(self.time_in_day)
        if t >= self._next_md:
            self._next_md = t + self.cfg.market.market_data.publish_ms
            self._publish_market_data()
        if t >= self._next_acct:
            self._next_acct = t + self.cfg.market.market_data.account_publish_ms
            self._publish_accounts()

    def _set_refs(self) -> None:
        self.exchange.set_ref({s: self.fair_cents(s) for s in self.instruments if s not in self.exchange.delisted})

    # ------------------------------------------------------------------ #

    def _close_auction(self) -> None:
        res = self.auction.close()
        for w in res.winners:
            self.tier[w["team"]] = "premium"
            if w["price"] > 0:
                self.exchange.charge(w["team"], w["price"], "auction")
        winners = {w["team"]: w for w in res.winners}
        pub = res.public()

        def msg(team: str) -> dict:
            w = winners.get(team)
            return {"type": P.AUCTION_RESULT, "won": w is not None, "price": w["price"] if w else 0,
                    "bid": w["bid"] if w else None, **pub}
        self._broadcast(msg)

    def _start_day(self, d: int) -> None:
        if d == 1 and self.auction.is_open:
            self._close_auction()
        self.day = d
        self.phase = Phase.TRADING
        self.research.commit_day()
        for s, inst in self.instruments.items():
            if inst.is_listed(d) and s not in self.exchange.delisted and not self.exchange.tradable[s]:
                self.exchange.set_tradable(s, True)
        self.step = self.paths.open_step(d)
        self._set_refs()
        tradable = sorted(s for s, ok in self.exchange.tradable.items() if ok)
        futures = self.listed_futures(d)
        offers = self.shop.catalog(d)
        rcfg = self.cfg.mechanics.research
        for team in self.teams:
            tier = self.tier[team]
            boost = self.research.boost(team) if rcfg.enabled else 0.0
            cap = rcfg.ic_cap if rcfg.enabled else 0.99
            ic = self.signal_engine.effective_ic(tier, d, boost, cap)
            msg = {"type": P.DAY_START, "day": d, "time_ms": 0,
                   "signals": self.signal_engine.equity_signals(d, tier, team, ic),
                   "tradable": tradable, "offers": offers,
                   "prev_close": {s: self.prev_close(s) for s in tradable}}
            if self.shop.owns(team, P.FUTURES_SIGNALS):
                msg["futures_signals"] = self.signal_engine.futures_signals(d, team, futures)
            if rcfg.enabled:
                msg["research"] = {"spent": self.research.effective[team], "boost": round(boost, 4)}
            self._emit(team, msg)
        self.bots.day_start()
        self._next_md = self.now
        self._next_acct = self.now

    def _mark(self, symbol: str, method: MarkMethod) -> int:
        fair = self.fair_cents(symbol, self.paths.close_step(self.day))
        if method is MarkMethod.FAIR:
            return fair
        ex = self.exchange
        if method is MarkMethod.VWAP:
            v = ex.vwap(symbol, self.now - self.cfg.market.vwap_window_ms)
            return v if v is not None else fair
        m = ex.books[symbol].mid()
        return ex.round_tick(m) if m is not None else fair

    def _end_of_day(self) -> None:
        d = self.day
        self.step = self.paths.close_step(d)
        self._set_refs()
        self._publish_market_data()
        self.bots.day_end()
        m = self.cfg.market
        live = [s for s in self.instruments if s not in self.exchange.delisted]
        marks = {s: self._mark(s, m.close_mark_method) for s in live}
        settle_px: dict[str, int] = {}
        expiring = [s for s in live if self.instruments[s].expiry_day == d]
        for s in expiring:
            und = self.instruments[s].underlying
            settle_px[s] = self._mark(und, m.settlement_method)
            marks[s] = settle_px[s]
        if m.cancel_on_eod:
            self.exchange.cancel_symbol(None, "eod")
        for s in expiring:
            self.exchange.settle(s, settle_px[s], reason="expiry")
        self.close_marks[d] = marks
        all_marks = {**self.public_marks(), **marks}
        self._prev_close.update(marks)
        self.phase = Phase.EOD
        for team in self.teams:
            acct = self.exchange.accounts[team]
            pnl = acct.pnl(all_marks)
            self.daily_pnl[team].append(pnl)
            self.pnl_curve[team].append((d, self.cfg.day_ms, pnl))
            self._emit(team, {"type": P.DAY_END, "day": d, "time_ms": self.cfg.day_ms,
                              "close_marks": marks, "pnl": pnl, "cash": acct.cash,
                              "positions": {k: v for k, v in acct.positions.items() if v},
                              "expired": expiring, "settlements": settle_px})

    def _close_round(self) -> None:
        marks = self.close_marks[self.day]
        for s in self.instruments:
            if s not in self.exchange.delisted:
                self.exchange.settle(s, marks[s], reason="round_close")
        self.exchange.cancel_symbol(None, "round_close")
        results = {}
        for team in self.teams:
            a = self.exchange.accounts[team]
            pnl = a.cash - a.starting_cash
            charges = dict(a.charges)
            results[team] = {
                "team": team,
                "pnl": pnl,
                "fees": a.fees_paid,
                "charges": charges,
                "trading_pnl": pnl + a.fees_paid + sum(charges.values()),
                "volume": a.volume,
                "trades": a.trades,
                "tier": self.tier[team],
                "daily_pnl": list(self.daily_pnl[team]),
            }
        ranked = sorted(results.values(), key=lambda r: (-r["pnl"], r["team"]))
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        self.results = results
        board = [{"rank": r["rank"], "team": r["team"], "pnl": r["pnl"]} for r in ranked]
        self.phase = Phase.CLOSED
        self._broadcast(lambda t: {"type": P.ROUND_END, "result": results[t], "leaderboard": board})

    # ------------------------------------------------------------------ #
    # Market data                                                         #
    # ------------------------------------------------------------------ #

    def _publish_market_data(self) -> None:
        md = self.cfg.market.market_data
        ex = self.exchange
        books = {s: ex.books[s].snapshot(md.book_depth) for s, ok in ex.tradable.items() if ok}
        trades = []
        for tr in self._pending_trades:
            item = {"symbol": tr["symbol"], "price": tr["price"], "qty": tr["qty"],
                    "aggressor": tr["aggressor"], "ts": tr["ts"]}
            if not md.anonymize_trades:
                item["buyer"], item["seller"] = tr["buyer"], tr["seller"]
            trades.append(item)
        self._pending_trades = []
        inav = self._inav()
        base = {"type": P.MARKET_DATA, "books": books, "trades": trades, "inav": inav}
        commod = None
        sc = self.cfg.signals
        if sc.commodity_feed and any(self.tier[t] == "premium" for t in self.teams):
            commod = self.signal_engine.commodity_quote(self.step)
        for team in self.teams:
            msg = dict(base)
            if commod is not None and self.tier[team] == "premium":
                msg["commodities"] = commod
            self._emit(team, msg)

    def _inav(self) -> dict[str, float]:
        from ..market.etf import index_nav, letf_nav

        ic = self.cfg.instruments
        out: dict[str, float] = {}
        mids = {s: self.exchange.books[s].mid() for s in ic.equities}
        if ic.etf_enabled and all(v is not None for v in mids.values()):
            out[ic.etf_symbol] = round(index_nav(
                [mids[s] for s in ic.equities], [self.prev_close(s) for s in ic.equities],
                self.prev_close(ic.etf_symbol), ic.etf_weighting), 2)
        u = mids.get(ic.letf_underlying)
        if ic.letf_enabled and u is not None:
            out[ic.letf_symbol] = round(letf_nav(u, self.prev_close(ic.letf_underlying),
                                                 self.prev_close(ic.letf_symbol), ic.letf_leverage), 2)
        return out

    def _publish_accounts(self) -> None:
        marks = self.public_marks()
        for team in self.teams:
            a = self.exchange.accounts[team]
            pnl = a.pnl(marks)
            self.pnl_curve[team].append((self.day, self.time_in_day, pnl))
            self._emit(team, {"type": P.ACCOUNT, "cash": a.cash,
                              "positions": {k: v for k, v in a.positions.items() if v},
                              "pnl": pnl, "fees_paid": a.fees_paid,
                              "charges": {k: v for k, v in a.charges.items() if v},
                              "open_orders": len(a.open_orders)})

    # ------------------------------------------------------------------ #
    # Team requests                                                       #
    # ------------------------------------------------------------------ #

    def handle(self, team: str, msg: Any) -> None:
        if team not in self._team_set:
            raise KeyError(f"unknown team {team!r}")
        if not isinstance(msg, dict):
            self._request_reject(team, "?", "bad_request")
            return
        t = msg.get("type")
        self.counters[f"req:{t}"] += 1
        if self.phase is Phase.CLOSED or self.phase is Phase.INIT:
            if t == P.PLACE_ORDER:
                self._emit(team, {"type": P.ORDER_REJECT, "client_id": _cid(msg), "symbol": msg.get("symbol"),
                                  "reason": "market_closed"})
            elif t == P.PING:
                self._emit(team, {"type": P.PONG, "nonce": msg.get("nonce")})
            else:
                self._request_reject(team, str(t), "round_closed")
            return
        handler = _HANDLERS.get(t)
        if handler is None:
            self._request_reject(team, str(t)[:32], "unknown_type")
            return
        handler(self, team, msg)

    def _h_place(self, team: str, msg: dict) -> None:
        cid = _cid(msg)
        sym = msg.get("symbol")
        base = {"type": P.ORDER_REJECT, "client_id": cid, "symbol": sym if isinstance(sym, str) else None}
        if self.phase is not Phase.TRADING:
            self._emit(team, {**base, "reason": "market_closed"})
            return
        side = msg.get("side")
        otype = msg.get("order_type", "limit")
        qty = msg.get("qty")
        price = msg.get("price")
        if (not isinstance(sym, str) or side not in P.SIDES or otype not in P.ORDER_TYPES
                or cid is _BAD or not _is_int(qty) or (price is not None and not _is_int(price))):
            self._emit(team, {**base, "client_id": None if cid is _BAD else cid, "reason": "bad_request"})
            return
        if otype == "market":
            price = None
        self.exchange.submit(team, sym, side, qty, price, otype, cid)

    def _h_cancel(self, team: str, msg: dict) -> None:
        oid = msg.get("order_id")
        if not _is_int(oid):
            self._emit(team, {"type": P.CANCEL_REJECT, "order_id": None, "reason": "bad_request"})
            return
        self.exchange.cancel(team, oid)

    def _h_cancel_all(self, team: str, msg: dict) -> None:
        sym = msg.get("symbol")
        if sym is not None and not isinstance(sym, str):
            self._request_reject(team, P.CANCEL_ALL, "bad_request")
            return
        if self.exchange.accounts[team].open_orders:
            self.exchange.cancel_all(team, sym)

    def _spendable(self, team: str) -> int:
        """Cash available for mechanic purchases (open auction bids are reserved)."""
        return self.exchange.accounts[team].cash - self.auction.committed(team)

    def _h_unlock(self, team: str, msg: dict) -> None:
        if self.cfg.instruments.futures_enabled and not self.futures_access.is_unlocked(team) \
                and self._spendable(team) < self.futures_access.fee_cents:
            self._request_reject(team, P.UNLOCK_FUTURES, "insufficient_cash")
            return
        reason, fee = self.futures_access.unlock(team)
        if reason:
            self._request_reject(team, P.UNLOCK_FUTURES, reason)
            return
        self.exchange.charge(team, fee, "futures_unlock")
        self.exchange.grant(team, FUTURES_PERMISSION)
        self._emit(team, {"type": P.FUTURES_UNLOCKED, "fee": fee})

    def _h_bid(self, team: str, msg: dict) -> None:
        if not self.cfg.mechanics.auction.enabled:
            self._request_reject(team, P.AUCTION_BID, "auction_disabled")
            return
        reason = self.auction.submit(team, msg.get("amount"), self.now)
        if reason:
            self._request_reject(team, P.AUCTION_BID, reason)
        else:
            self._emit(team, {"type": "auction_bid_ack", "amount": msg.get("amount")})

    def _h_buy_signal(self, team: str, msg: dict) -> None:
        product = msg.get("product")
        product = product if isinstance(product, str) else ""
        day = max(self.day, 1)
        reason = self.shop.check(team, product, day)
        if reason is None and self._spendable(team) < self.shop.price(product):
            reason = "insufficient_cash"
        if reason:
            self._request_reject(team, P.BUY_SIGNAL, reason)
            return
        reason, price = self.shop.purchase(team, product, day)
        if reason:
            self._request_reject(team, P.BUY_SIGNAL, reason)
            return
        if price:
            self.exchange.charge(team, price, "signal_shop")
        sigs = []
        if self.phase is Phase.TRADING and product == P.FUTURES_SIGNALS:
            sigs = self.signal_engine.futures_signals(self.day, team, self.listed_futures(self.day))
        self._emit(team, {"type": P.SIGNAL, "product": product, "price": price, "signals": sigs})

    def _h_research(self, team: str, msg: dict) -> None:
        amount = msg.get("amount")
        if self.day >= self.cfg.days and self.phase is not Phase.PRE_OPEN:
            self._request_reject(team, P.ALLOCATE_RESEARCH, "too_late")
            return
        if _is_int(amount) and amount > 0 and self.cfg.mechanics.research.enabled \
                and self._spendable(team) < amount:
            self._request_reject(team, P.ALLOCATE_RESEARCH, "insufficient_cash")
            return
        reason = self.research.allocate(team, amount)
        if reason:
            self._request_reject(team, P.ALLOCATE_RESEARCH, reason)
            return
        self.exchange.charge(team, amount, "research")
        self._emit(team, {"type": "research_ack", "amount": amount, "total": self.research.total(team)})

    def _h_ping(self, team: str, msg: dict) -> None:
        self._emit(team, {"type": P.PONG, "nonce": msg.get("nonce")})


_BAD = object()


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _cid(msg: dict) -> Any:
    c = msg.get("client_id")
    if c is None:
        return None
    if isinstance(c, str) and len(c) <= 64:
        return c
    return _BAD


_HANDLERS: dict[str, Callable[[RoundEngine, str, dict], None]] = {
    P.PLACE_ORDER: RoundEngine._h_place,
    P.CANCEL_ORDER: RoundEngine._h_cancel,
    P.CANCEL_ALL: RoundEngine._h_cancel_all,
    P.UNLOCK_FUTURES: RoundEngine._h_unlock,
    P.AUCTION_BID: RoundEngine._h_bid,
    P.BUY_SIGNAL: RoundEngine._h_buy_signal,
    P.ALLOCATE_RESEARCH: RoundEngine._h_research,
    P.PING: RoundEngine._h_ping,
}
