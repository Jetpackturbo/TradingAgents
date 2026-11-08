"""Deterministic exchange core.

The :class:`Exchange` is a pure state machine. Every state change goes through
:meth:`Exchange.apply` with a JSON-serialisable command dict; it returns a list
of JSON-serialisable event dicts. Replaying the recorded command log on a fresh
instance reproduces exactly the same events (see :func:`replay`).

Units: prices are integer cents, quantities integer units, cash integer cents.
Every cash movement is a transfer between two accounts (fees and mechanic
charges go to the HOUSE account), so total cash is exactly conserved and the
sum of positions per instrument is always zero.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Callable

from ..config import MarketConfig, STPMode
from .accounts import Account
from .instruments import Instrument
from .orderbook import OrderBook
from .risk import check_risk, validate_shape
from .types import Order, OrderType, Side

HOUSE_ACCOUNT = "__house__"


def _copy_json(v: Any) -> Any:
    """Deep copy for JSON-shaped data (much cheaper than ``copy.deepcopy``)."""
    if isinstance(v, dict):
        return {k: _copy_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_copy_json(x) for x in v]
    return v


class Exchange:
    def __init__(self, cfg: MarketConfig, instruments: dict[str, Instrument], record: bool = True) -> None:
        self.cfg = cfg
        self.instruments = dict(instruments)
        self.books: dict[str, OrderBook] = {s: OrderBook(s) for s in instruments}
        self.accounts: dict[str, Account] = {}
        self.tradable: dict[str, bool] = {s: False for s in instruments}
        self.delisted: set[str] = set()
        self.ref_price: dict[str, int] = {}
        self.last_price: dict[str, int] = {}
        self.tape: dict[str, deque] = defaultdict(lambda: deque(maxlen=50_000))
        self.now = 0
        self.seq = 0
        self._next_oid = 1
        self.command_log: list[dict] | None = [] if record else None
        self.listeners: list[Callable[[list[dict]], None]] = []
        self._ops: dict[str, Callable[..., list[dict]]] = {
            "add_account": self._op_add_account,
            "set_tradable": self._op_set_tradable,
            "set_ref": self._op_set_ref,
            "submit": self._op_submit,
            "cancel": self._op_cancel,
            "cancel_all": self._op_cancel_all,
            "cancel_symbol": self._op_cancel_symbol,
            "charge": self._op_charge,
            "grant": self._op_grant,
            "settle": self._op_settle,
            "block": self._op_block,
        }
        self._apply({"op": "add_account", "ts": 0, "account": HOUSE_ACCOUNT, "kind": "house",
                     "cash": 0, "risk_exempt": True, "permissions": []})

    # ------------------------------------------------------------------ #
    # Command plumbing                                                    #
    # ------------------------------------------------------------------ #

    def apply(self, cmd: dict) -> list[dict]:
        return self._apply(cmd)

    def _apply(self, cmd: dict) -> list[dict]:
        op = cmd.get("op")
        handler = self._ops.get(op)
        if handler is None:
            raise ValueError(f"unknown op {op!r}")
        ts = int(cmd.get("ts", self.now))
        if ts < self.now:
            ts = self.now
        self.now = ts
        if self.command_log is not None:
            logged = _copy_json(cmd)
            logged["ts"] = ts
            self.command_log.append(logged)
        args = {k: v for k, v in cmd.items() if k not in ("op", "ts")}
        events = handler(**args)
        for ev in events:
            self.seq += 1
            ev["seq"] = self.seq
            ev.setdefault("ts", ts)
        for listener in self.listeners:
            listener(events)
        return events

    # Convenience wrappers ------------------------------------------------

    def add_account(self, account: str, kind: str = "team", cash: int = 0, risk_exempt: bool = False,
                    permissions: list[str] | None = None, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "add_account", "ts": self.now if ts is None else ts, "account": account,
                            "kind": kind, "cash": cash, "risk_exempt": risk_exempt,
                            "permissions": list(permissions or [])})

    def set_tradable(self, symbol: str, tradable: bool, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "set_tradable", "ts": self.now if ts is None else ts,
                            "symbol": symbol, "tradable": tradable})

    def set_ref(self, prices: dict[str, int], ts: int | None = None) -> list[dict]:
        return self._apply({"op": "set_ref", "ts": self.now if ts is None else ts, "prices": dict(prices)})

    def submit(self, account: str, symbol: str, side: Side | str, qty: int, price: int | None = None,
               otype: OrderType | str = OrderType.LIMIT, client_id: str | None = None,
               ts: int | None = None) -> list[dict]:
        return self._apply({"op": "submit", "ts": self.now if ts is None else ts, "account": account,
                            "symbol": symbol, "side": Side(side).value, "qty": qty, "price": price,
                            "otype": OrderType(otype).value, "client_id": client_id})

    def cancel(self, account: str, order_id: int, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "cancel", "ts": self.now if ts is None else ts,
                            "account": account, "order_id": order_id})

    def cancel_all(self, account: str, symbol: str | None = None, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "cancel_all", "ts": self.now if ts is None else ts,
                            "account": account, "symbol": symbol})

    def cancel_symbol(self, symbol: str | None, reason: str, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "cancel_symbol", "ts": self.now if ts is None else ts,
                            "symbol": symbol, "reason": reason})

    def charge(self, account: str, amount: int, reason: str, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "charge", "ts": self.now if ts is None else ts,
                            "account": account, "amount": amount, "reason": reason})

    def grant(self, account: str, permission: str, ts: int | None = None) -> list[dict]:
        return self._apply({"op": "grant", "ts": self.now if ts is None else ts,
                            "account": account, "permission": permission})

    def block(self, buyer: str, seller: str, symbol: str, qty: int, price: int, ts: int | None = None) -> list[dict]:
        """Off-book bilateral trade (used for house-bot hedging; never printed)."""
        return self._apply({"op": "block", "ts": self.now if ts is None else ts, "buyer": buyer,
                            "seller": seller, "symbol": symbol, "qty": qty, "price": price})

    def settle(self, symbol: str, price: int, reason: str = "settlement", delist: bool = True,
               ts: int | None = None) -> list[dict]:
        return self._apply({"op": "settle", "ts": self.now if ts is None else ts, "symbol": symbol,
                            "price": price, "reason": reason, "delist": delist})

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #

    def marks_for_risk(self) -> dict[str, int]:
        out = {}
        for s in self.instruments:
            p = self.ref_price.get(s) or self.last_price.get(s)
            if p is None:
                m = self.books[s].mid()
                p = int(m) if m is not None else 0
            out[s] = p
        return out

    def vwap(self, symbol: str, since_ts: int) -> int | None:
        num = den = 0
        for ts, price, qty in reversed(self.tape[symbol]):
            if ts < since_ts:
                break
            num += price * qty
            den += qty
        if not den:
            return None
        return self.round_tick(num / den)

    def round_tick(self, price: float) -> int:
        t = self.cfg.tick_cents
        return max(t, int(round(price / t)) * t)

    def total_cash(self) -> int:
        return sum(a.cash for a in self.accounts.values())

    def net_position(self, symbol: str) -> int:
        return sum(a.positions.get(symbol, 0) for a in self.accounts.values())

    def check_invariants(self, expected_total_cash: int | None = None) -> None:
        for s, book in self.books.items():
            book.check_integrity()
            assert self.net_position(s) == 0, f"net position {s} != 0"
        for a in self.accounts.values():
            for s in self.instruments:
                ob = sum(o.remaining for o in a.open_orders.values() if o.symbol == s and o.side is Side.BUY)
                os_ = sum(o.remaining for o in a.open_orders.values() if o.symbol == s and o.side is Side.SELL)
                assert a.open_buy_qty.get(s, 0) == ob, f"open buy mismatch {a.account_id} {s}"
                assert a.open_sell_qty.get(s, 0) == os_, f"open sell mismatch {a.account_id} {s}"
            for oid, o in a.open_orders.items():
                assert self.books[o.symbol].get(oid) is o, "open order not in book"
        n_book = sum(len(b) for b in self.books.values())
        n_open = sum(len(a.open_orders) for a in self.accounts.values())
        assert n_book == n_open, "book/open-order count mismatch"
        if expected_total_cash is not None:
            assert self.total_cash() == expected_total_cash, "cash not conserved"

    # ------------------------------------------------------------------ #
    # Ops                                                                 #
    # ------------------------------------------------------------------ #

    def _op_add_account(self, account: str, kind: str, cash: int, risk_exempt: bool,
                        permissions: list[str]) -> list[dict]:
        if account in self.accounts:
            raise ValueError(f"duplicate account {account}")
        # Starting cash is minted from the HOUSE so that total cash stays conserved.
        acct = Account(account, kind, cash=cash, starting_cash=cash, risk_exempt=risk_exempt,
                       permissions=set(permissions))
        self.accounts[account] = acct
        if account != HOUSE_ACCOUNT:
            self.accounts[HOUSE_ACCOUNT].cash -= cash
        return [{"type": "account_added", "account": account, "kind": kind, "cash": cash}]

    def _op_set_tradable(self, symbol: str, tradable: bool) -> list[dict]:
        if symbol not in self.instruments:
            raise ValueError(f"unknown symbol {symbol}")
        if tradable and symbol in self.delisted:
            raise ValueError(f"{symbol} is delisted")
        events: list[dict] = []
        if not tradable and self.tradable[symbol]:
            events += self._cancel_where(lambda o: o.symbol == symbol, "halted")
        self.tradable[symbol] = tradable
        events.append({"type": "status", "symbol": symbol, "tradable": tradable})
        return events

    def _op_set_ref(self, prices: dict[str, int]) -> list[dict]:
        for s, p in prices.items():
            if s in self.instruments and p and p > 0:
                self.ref_price[s] = int(p)
        return []

    def _op_grant(self, account: str, permission: str) -> list[dict]:
        self.accounts[account].permissions.add(permission)
        return [{"type": "permission", "account": account, "permission": permission}]

    def _op_charge(self, account: str, amount: int, reason: str) -> list[dict]:
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise ValueError("charge amount must be int cents")
        acct = self.accounts[account]
        acct.cash -= amount
        acct.charges[reason] += amount
        self.accounts[HOUSE_ACCOUNT].cash += amount
        return [{"type": "cash", "account": account, "amount": -amount, "reason": reason,
                 "cash_after": acct.cash}]

    def _op_cancel(self, account: str, order_id: int) -> list[dict]:
        acct = self.accounts.get(account)
        order = acct.open_orders.get(order_id) if acct else None
        if order is None:
            return [{"type": "cancel_reject", "account": account, "order_id": order_id,
                     "reason": "unknown_order"}]
        return [self._cancel_order(order, "user")]

    def _op_cancel_all(self, account: str, symbol: str | None) -> list[dict]:
        return self._cancel_where(
            lambda o: o.account == account and (symbol is None or o.symbol == symbol), "user")

    def _op_cancel_symbol(self, symbol: str | None, reason: str) -> list[dict]:
        return self._cancel_where(lambda o: symbol is None or o.symbol == symbol, reason)

    def _cancel_where(self, pred: Callable[[Order], bool], reason: str) -> list[dict]:
        victims = [o for a in self.accounts.values() for o in a.open_orders.values() if pred(o)]
        victims.sort(key=lambda o: o.order_id)
        return [self._cancel_order(o, reason) for o in victims]

    def _cancel_order(self, order: Order, reason: str) -> dict:
        self.books[order.symbol].remove(order.order_id)
        self.accounts[order.account].untrack(order)
        return {"type": "cancelled", "account": order.account, "order_id": order.order_id,
                "client_id": order.client_id, "symbol": order.symbol, "remaining": order.remaining,
                "reason": reason}

    def _op_settle(self, symbol: str, price: int, reason: str, delist: bool) -> list[dict]:
        if not isinstance(price, int) or price <= 0:
            raise ValueError("settlement price must be positive int cents")
        events = self._cancel_where(lambda o: o.symbol == symbol, reason)
        for acct in self.accounts.values():
            q = acct.positions.get(symbol, 0)
            if q:
                acct.cash += q * price
                acct.positions[symbol] = 0
                events.append({"type": "settlement", "account": acct.account_id, "symbol": symbol,
                               "qty": q, "price": price, "cash": q * price, "reason": reason})
        if delist:
            self.tradable[symbol] = False
            self.delisted.add(symbol)
        events.append({"type": "settled", "symbol": symbol, "price": price, "reason": reason,
                       "delisted": delist})
        return events

    def _op_block(self, buyer: str, seller: str, symbol: str, qty: int, price: int) -> list[dict]:
        if buyer == seller or buyer not in self.accounts or seller not in self.accounts:
            raise ValueError("bad block counterparties")
        if symbol not in self.instruments or symbol in self.delisted:
            raise ValueError("bad block symbol")
        if not isinstance(qty, int) or qty <= 0 or not isinstance(price, int) or price <= 0:
            raise ValueError("bad block qty/price")
        b, s = self.accounts[buyer], self.accounts[seller]
        b.positions[symbol] += qty
        s.positions[symbol] -= qty
        b.cash -= qty * price
        s.cash += qty * price
        return [{"type": "block", "account": buyer, "symbol": symbol, "side": "buy", "qty": qty, "price": price},
                {"type": "block", "account": seller, "symbol": symbol, "side": "sell", "qty": qty, "price": price}]

    def _op_submit(self, account: str, symbol: str, side: str, qty: Any, price: Any, otype: str,
                   client_id: str | None) -> list[dict]:
        base = {"account": account, "client_id": client_id, "symbol": symbol}
        acct = self.accounts.get(account)
        if acct is None:
            return [{"type": "reject", **base, "reason": "unknown_account"}]
        try:
            side_e = Side(side)
            otype_e = OrderType(otype)
        except ValueError:
            return [{"type": "reject", **base, "reason": "bad_enum"}]
        reason = validate_shape(self, symbol, side_e, qty, price, otype_e)
        if reason is None:
            reason = check_risk(self, acct, symbol, side_e, qty, price, otype_e)
        book = self.books.get(symbol)
        if reason is None and otype_e is OrderType.MARKET:
            best = book.best(side_e.opposite)
            if best is None:
                reason = "no_liquidity"
            else:
                band = self.cfg.market_order_band
                t = self.cfg.tick_cents
                if side_e is Side.BUY:
                    price = int(math.floor(best * (1 + band) / t)) * t
                else:
                    price = max(t, int(math.ceil(best * (1 - band) / t)) * t)
        if reason is not None:
            return [{"type": "reject", **base, "side": side, "qty": qty, "price": price, "reason": reason}]

        oid = self._next_oid
        self._next_oid += 1
        order = Order(oid, account, symbol, side_e, int(price), qty, qty, self.now, otype_e, client_id)
        events: list[dict] = [{"type": "ack", **base, "order_id": oid, "side": side_e.value, "qty": qty,
                               "price": order.price, "otype": otype_e.value}]
        events += self._match(order)
        if order.remaining > 0:
            if otype_e is OrderType.LIMIT:
                book.add(order)
                acct.track_open(order)
            else:
                events.append({"type": "cancelled", "account": account, "order_id": oid,
                               "client_id": client_id, "symbol": symbol, "remaining": order.remaining,
                               "reason": "ioc"})
        return events

    def _match(self, taker: Order) -> list[dict]:
        events: list[dict] = []
        book = self.books[taker.symbol]
        opp = taker.side.opposite
        stp = self.cfg.stp_mode
        for maker in book.iter_side(opp):
            if taker.remaining == 0:
                break
            if taker.side is Side.BUY and maker.price > taker.price:
                break
            if taker.side is Side.SELL and maker.price < taker.price:
                break
            if maker.account == taker.account and stp is not STPMode.OFF:
                if stp in (STPMode.CANCEL_RESTING, STPMode.CANCEL_BOTH):
                    events.append(self._cancel_order(maker, "self_trade"))
                if stp in (STPMode.CANCEL_INCOMING, STPMode.CANCEL_BOTH):
                    events.append({"type": "cancelled", "account": taker.account, "order_id": taker.order_id,
                                   "client_id": taker.client_id, "symbol": taker.symbol,
                                   "remaining": taker.remaining, "reason": "self_trade"})
                    taker.remaining = 0
                    break
                continue
            qty = min(taker.remaining, maker.remaining)
            events += self._execute(taker, maker, qty)
        return events

    def _execute(self, taker: Order, maker: Order, qty: int) -> list[dict]:
        price = maker.price
        book = self.books[taker.symbol]
        t_acct = self.accounts[taker.account]
        m_acct = self.accounts[maker.account]
        house = self.accounts[HOUSE_ACCOUNT]
        book.reduce(maker, qty)
        m_acct.reduce_open(maker, qty)
        if maker.remaining == 0:
            m_acct.open_orders.pop(maker.order_id, None)
        taker.remaining -= qty
        notional = price * qty
        fees = self.cfg.fees
        t_fee = fees.taker_cents_per_unit * qty
        m_fee = fees.maker_cents_per_unit * qty
        for acct, order, fee in ((t_acct, taker, t_fee), (m_acct, maker, m_fee)):
            sign = order.side.sign
            acct.positions[order.symbol] += sign * qty
            acct.cash -= sign * notional + fee
            acct.fees_paid += fee
            acct.volume += qty
            acct.trades += 1
            house.cash += fee
        self.last_price[taker.symbol] = price
        self.tape[taker.symbol].append((self.now, price, qty))
        buyer, seller = (taker, maker) if taker.side is Side.BUY else (maker, taker)
        return [
            {"type": "fill", "account": taker.account, "order_id": taker.order_id, "client_id": taker.client_id,
             "symbol": taker.symbol, "side": taker.side.value, "price": price, "qty": qty, "fee": t_fee,
             "liquidity": "taker", "remaining": taker.remaining},
            {"type": "fill", "account": maker.account, "order_id": maker.order_id, "client_id": maker.client_id,
             "symbol": maker.symbol, "side": maker.side.value, "price": price, "qty": qty, "fee": m_fee,
             "liquidity": "maker", "remaining": maker.remaining},
            {"type": "trade", "symbol": taker.symbol, "price": price, "qty": qty, "aggressor": taker.side.value,
             "buyer": buyer.account, "seller": seller.account},
        ]


def replay(cfg: MarketConfig, instruments: dict[str, Instrument], commands: list[dict]) -> tuple[Exchange, list[dict]]:
    """Re-run a command log on a fresh exchange. The HOUSE account creation
    (always the first command) is skipped because the constructor performs it."""
    ex = Exchange(cfg, instruments)
    events: list[dict] = []
    for cmd in commands[1:]:
        events += ex.apply(cmd)
    return ex, events
