"""Sealed-bid multi-unit auction for the premium signal tier.

The top ``k`` bidders win. Payment rules:

* ``uniform_k_plus_1`` - every winner pays the (k+1)-th highest bid (or the
  reserve if there are not more than k bids). This is the natural extension of
  a second-price auction to k identical units and makes truthful bidding a
  dominant strategy for unit-demand bidders.
* ``own_bid`` - pay-as-bid.
* ``gsp`` - generalised second price: each winner pays the bid ranked just below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..config import AuctionConfig, AuctionRule


@dataclass
class AuctionResult:
    winners: list[dict] = field(default_factory=list)  # {"team", "bid", "price", "rank"}
    losers: list[str] = field(default_factory=list)
    clearing_price: int | None = None
    n_bids: int = 0

    def payment(self, team: str) -> int:
        for w in self.winners:
            if w["team"] == team:
                return w["price"]
        return 0

    def public(self) -> dict:
        return {"n_bids": self.n_bids, "n_winners": len(self.winners), "clearing_price": self.clearing_price}


class SealedBidAuction:
    def __init__(self, cfg: AuctionConfig, seed: int = 0,
                 available_cash: Callable[[str], int] | None = None) -> None:
        self.cfg = cfg
        self._rng = random.Random(seed)
        self._available_cash = available_cash
        self._bids: dict[str, tuple[int, int, int]] = {}  # team -> (amount, ts, arrival)
        self._arrival = 0
        self.is_open = False
        self.result: AuctionResult | None = None

    def open(self) -> None:
        if self.result is not None:
            raise RuntimeError("auction already closed")
        self.is_open = True

    def submit(self, team: str, amount: int, ts: int) -> str | None:
        """Place or replace a sealed bid. Returns a reject reason or None."""
        if not self.is_open:
            return "auction_closed"
        if not isinstance(amount, int) or isinstance(amount, bool):
            return "bad_amount"
        if amount < 0:
            return "bad_amount"
        if amount < self.cfg.reserve_cents:
            return "below_reserve"
        if amount > self.cfg.max_bid_cents:
            return "above_max_bid"
        if self._available_cash is not None and amount > self._available_cash(team):
            return "insufficient_cash"
        self._arrival += 1
        self._bids[team] = (amount, ts, self._arrival)
        return None

    def withdraw(self, team: str) -> bool:
        if not self.is_open:
            return False
        return self._bids.pop(team, None) is not None

    def has_bid(self, team: str) -> bool:
        return team in self._bids

    def committed(self, team: str) -> int:
        """Cash a team may still be charged by the open auction."""
        return self._bids[team][0] if self.is_open and team in self._bids else 0

    def close(self) -> AuctionResult:
        if self.result is not None:
            return self.result
        self.is_open = False
        entries = list(self._bids.items())
        if self.cfg.tie_break == "random":
            keyed = [(t, b, self._rng.random()) for t, b in sorted(entries)]
            keyed.sort(key=lambda x: (-x[1][0], x[2]))
        else:
            keyed = [(t, b, b[2]) for t, b in entries]
            keyed.sort(key=lambda x: (-x[1][0], x[1][1], x[2]))
        ordered = [(t, b[0]) for t, b, _ in keyed]
        k = self.cfg.winners
        reserve = self.cfg.reserve_cents
        winners = ordered[:k]
        losers = [t for t, _ in ordered[k:]]
        res = AuctionResult(losers=losers, n_bids=len(ordered))
        rule = self.cfg.payment_rule
        if rule is AuctionRule.UNIFORM_K_PLUS_1:
            price = max(reserve, ordered[k][1]) if len(ordered) > k else reserve
            res.clearing_price = price
        for rank, (team, bid) in enumerate(winners):
            if rule is AuctionRule.UNIFORM_K_PLUS_1:
                pay = res.clearing_price
            elif rule is AuctionRule.OWN_BID:
                pay = bid
            else:
                pay = max(reserve, ordered[rank + 1][1]) if rank + 1 < len(ordered) else reserve
            res.winners.append({"team": team, "bid": bid, "price": int(min(pay, bid)), "rank": rank + 1})
        if rule is not AuctionRule.UNIFORM_K_PLUS_1 and res.winners:
            res.clearing_price = min(w["price"] for w in res.winners)
        self.result = res
        return res
