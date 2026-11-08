"""Gateway-side round wrapper shared by the live server and the headless simulator.

:class:`LiveRound` owns everything between a team's socket and the
:class:`RoundEngine`:

* ingress: per-team pending cap (flood protection), speed bump + jitter,
  token-bucket rate limiting evaluated at the moment a message reaches the
  matching engine, violation counting;
* egress: fan-out of engine messages to the currently attached connection,
  plus a state snapshot when a team (re)connects mid-round.

It is still driven by an external clock (``pump(to_ms)``) so that it stays
deterministic; :func:`run_realtime` binds it to the wall clock.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import random
import time
from collections import defaultdict
from typing import Callable

from ..config import RoundConfig
from ..sdk import protocol as P
from .ratelimit import TokenBucket
from .round import Phase, RoundEngine

Deliver = Callable[[dict], None]


class LiveRound:
    def __init__(self, cfg: RoundConfig, teams: list[str], seed: int = 0, enforce_rate_limits: bool = True,
                 max_pending_per_team: int | None = None,
                 on_violation_limit: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.engine = RoundEngine(cfg, teams, sink=self._sink)
        self.outputs: dict[str, Deliver] = {}
        self._pending: list = []
        self._pending_count: dict[str, int] = defaultdict(int)
        self._seq = itertools.count()
        self._rng = random.Random(seed)
        rc = cfg.market.risk
        self.enforce_rate_limits = enforce_rate_limits
        self.max_pending = max_pending_per_team or max(2 * rc.msg_burst, 20)
        self.buckets = {t: TokenBucket(rc.msgs_per_second, rc.msg_burst) for t in teams}
        self.rate_limited: dict[str, int] = defaultdict(int)
        self.messages_received: dict[str, int] = defaultdict(int)
        self.on_violation_limit = on_violation_limit
        self._violations: dict[str, int] = defaultdict(int)
        # Snapshot state replayed on reconnect.
        self._round_start: dict | None = None
        self._auction_open: dict | None = None
        self._last: dict[str, dict[str, dict]] = defaultdict(dict)

    # ------------------------------------------------------------ egress

    _SNAPSHOT_TYPES = (P.AUCTION_RESULT, P.DAY_START, P.ACCOUNT, P.DAY_END, P.ROUND_END, P.FUTURES_UNLOCKED)

    def _sink(self, team: str, msg: dict) -> None:
        t = msg["type"]
        if t == P.ROUND_START:
            self._round_start = msg
        elif t == P.AUCTION_OPEN:
            self._auction_open = msg
        elif t in self._SNAPSHOT_TYPES:
            self._last[team][t] = msg
        out = self.outputs.get(team)
        if out is not None:
            out(msg)

    def attach(self, team: str, deliver: Deliver) -> None:
        """Route ``team``'s messages to ``deliver``; replays state if mid-round."""
        if team not in self.buckets:
            raise KeyError(team)
        self.outputs[team] = deliver
        if self._round_start is None:
            return
        deliver(dict(self._round_start, reconnect=True))
        last = self._last[team]
        if self._auction_open is not None and self.engine.auction.is_open:
            deliver(self._auction_open)
        for t in (P.AUCTION_RESULT, P.FUTURES_UNLOCKED):
            if t in last:
                deliver(last[t])
        eng = self.engine
        if eng.phase is Phase.TRADING and P.DAY_START in last:
            deliver(last[P.DAY_START])
        elif eng.phase is Phase.EOD and P.DAY_END in last:
            deliver(last[P.DAY_END])
        if eng.phase is not Phase.INIT and not eng.closed:
            a = eng.exchange.accounts[team]
            deliver({"type": P.ACCOUNT, "day": eng.day, "time_ms": eng.time_in_day, "cash": a.cash,
                     "positions": {k: v for k, v in a.positions.items() if v},
                     "pnl": a.pnl(eng.public_marks()), "fees_paid": a.fees_paid,
                     "charges": {k: v for k, v in a.charges.items() if v},
                     "open_orders": len(a.open_orders),
                     "orders": [{"order_id": o.order_id, "client_id": o.client_id, "symbol": o.symbol,
                                 "side": o.side.value, "price": o.price, "remaining": o.remaining}
                                for o in a.open_orders.values()]})
        if eng.closed and P.ROUND_END in last:
            deliver(last[P.ROUND_END])

    def detach(self, team: str, deliver: Deliver | None = None) -> None:
        if deliver is None or self.outputs.get(team) is deliver:
            self.outputs.pop(team, None)

    # ------------------------------------------------------------ ingress

    def start(self) -> None:
        self.engine.start()

    @property
    def closed(self) -> bool:
        return self.engine.closed

    @property
    def now(self) -> int:
        return self.engine.now

    def submit(self, team: str, msg: dict, extra_delay_ms: int = 0) -> None:
        """Accept a request from ``team`` at the current round time.

        ``extra_delay_ms`` models network latency (simulator only)."""
        if team not in self.buckets:
            raise KeyError(team)
        self.messages_received[team] += 1
        if not isinstance(msg, dict) or self.engine.closed or self.engine.phase is Phase.INIT:
            self.engine.handle(team, msg)
            return
        if self._pending_count[team] >= self.max_pending:
            self._reject_rate(team, msg)
            return
        sb = self.cfg.market.speed_bump
        delay = max(0, int(extra_delay_ms))
        if msg.get("type") not in (P.CANCEL_ORDER, P.CANCEL_ALL) or sb.apply_to_cancels:
            delay += sb.delay_ms + (self._rng.randint(0, sb.jitter_ms) if sb.jitter_ms else 0)
        self._pending_count[team] += 1
        heapq.heappush(self._pending, (self.engine.now + delay, next(self._seq), team, msg))

    def _reject_rate(self, team: str, msg) -> None:
        self.rate_limited[team] += 1
        self._violations[team] += 1
        self._sink(team, {"type": P.REQUEST_REJECT, "day": self.engine.day, "time_ms": self.engine.time_in_day,
                          "request": msg.get("type") if isinstance(msg, dict) else "?",
                          "reason": "rate_limited",
                          "client_id": msg.get("client_id") if isinstance(msg, dict) else None})
        limit = self.cfg.market.risk.max_rate_violations
        if limit and self._violations[team] >= limit:
            self._violations[team] = 0
            if self.on_violation_limit is not None:
                self.on_violation_limit(team)

    @property
    def next_due_ms(self) -> int | None:
        return self._pending[0][0] if self._pending else None

    def process_due(self, before_ms: int, after_each: Callable[[], None] | None = None) -> None:
        """Hand every pending request due strictly before ``before_ms`` to the engine."""
        eng = self.engine
        while self._pending and self._pending[0][0] < before_ms and not eng.closed:
            due, _, team, msg = heapq.heappop(self._pending)
            self._pending_count[team] -= 1
            eng.advance(due)
            if eng.closed:
                break
            if self.enforce_rate_limits and not self.buckets[team].allow(eng.now):
                self._reject_rate(team, msg)
            else:
                eng.handle(team, msg)
            if after_each is not None:
                after_each()

    def pump(self, to_ms: int) -> None:
        """Advance round time to ``to_ms``, interleaving due requests with ticks."""
        eng = self.engine
        while not eng.closed:
            nxt = eng._next_tick
            if nxt > to_ms:
                self.process_due(to_ms + 1)
                if not eng.closed:
                    eng.advance(to_ms)
                return
            self.process_due(nxt)
            if not eng.closed:
                eng.advance(nxt)


class RealtimeDriver:
    """Binds a :class:`LiveRound` to the wall clock (optionally accelerated).

    All methods must be called from the event loop thread that runs :meth:`run`.
    """

    def __init__(self, live: LiveRound, speed: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.live = live
        self.speed = float(speed)
        self._clock = clock
        self._base_round = 0
        self._base_wall: float | None = None
        self.paused = False
        self.aborted = False
        self._wake = asyncio.Event()
        self.max_lag_ms = 0

    def now_ms(self) -> int:
        """Current round time according to the wall clock."""
        if self._base_wall is None or self.paused:
            return self.live.now
        return self._base_round + int((self._clock() - self._base_wall) * 1000.0 * self.speed)

    def _rebase(self) -> None:
        self._base_round = self.live.now
        self._base_wall = self._clock()

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.catch_up()
        self._rebase()
        self.speed = float(speed)
        self._wake.set()

    def pause(self) -> None:
        if not self.paused:
            self.catch_up()
            self.paused = True

    def resume(self) -> None:
        if self.paused:
            self.paused = False
            self._rebase()
            self._wake.set()

    def abort(self) -> None:
        self.aborted = True
        self._wake.set()

    def catch_up(self) -> None:
        if not self.paused and self._base_wall is not None and not self.live.closed:
            self.live.pump(self.now_ms())

    def submit(self, team: str, msg: dict) -> None:
        """Gateway entry point: timestamp at the current wall-clock round time."""
        self.catch_up()
        self.live.submit(team, msg)
        if not self.paused:
            self.live.process_due(self.live.now + 1)
        self._wake.set()

    async def run(self) -> None:
        live = self.live
        if live.engine.phase is Phase.INIT:
            live.start()
        self._rebase()
        while not live.closed and not self.aborted:
            if self.paused:
                self._wake.clear()
                await self._wait(0.25)
                continue
            target = self.now_ms()
            # How late (round ms) the oldest due tick is being processed.
            self.max_lag_ms = max(self.max_lag_ms, target - live.engine._next_tick)
            live.pump(target)
            if live.closed:
                break
            wake = live.engine._next_tick
            due = live.next_due_ms
            if due is not None:
                wake = min(wake, due)
            self._wake.clear()
            await self._wait(max(0.0, (wake - self.now_ms()) / 1000.0 / self.speed))

    async def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
