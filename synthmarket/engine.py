"""Endogenous-price UWTC round engine.

UWTC's :class:`RoundEngine` reads a hidden, pre-drawn fair path. Here ``fair()`` is
the PUBLIC reference (book mid, else last trade, else the configured initial price),
so ``_set_refs`` / ``prev_close`` / price bands / ``ctx.fair`` all follow the book and
the price is formed only by bot order flow.

The ground-truth target schedule is held by the engine but only handed out, through
:meth:`EndogenousRoundEngine.target_view`, to :class:`synthmarket.bots.TargetTrader`.
Every read is counted per account in ``schedule_reads``.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from uwtc.config import RoundConfig
from uwtc.engine.round import RoundEngine


class ScheduleView:
    def __init__(self, engine: "EndogenousRoundEngine", account: str) -> None:
        self._engine = engine
        self._account = account

    def value(self, symbol: str, bar: int) -> float:
        """Target F (dollars) in force during bar ``bar`` (clamped to the schedule)."""
        eng = self._engine
        eng.schedule_reads[self._account] += 1
        path = eng._schedule[symbol]
        return float(path[min(max(bar, 0), len(path) - 1)])


class EndogenousRoundEngine(RoundEngine):
    def __init__(self, cfg: RoundConfig, teams: list[str] | None = None, sink=None, *,
                 bar_ms: int, schedule: dict[str, np.ndarray] | None = None,
                 record_commands: bool = False) -> None:
        self._initial = {s: float(m.initial_price) for s, m in cfg.model.equities.items()}
        self.bar_ms = int(bar_ms)
        if cfg.day_ms % self.bar_ms:
            raise ValueError("day_ms must be a multiple of bar_ms")
        self.n_bars = cfg.days * cfg.day_ms // self.bar_ms
        self._schedule = {s: np.asarray(p, dtype=float) for s, p in (schedule or {}).items()}
        self.schedule_reads: Counter = Counter()
        super().__init__(cfg, list(teams or []), sink)
        if not record_commands:
            self.exchange.command_log = None

    def _tick(self, t: int) -> None:
        # RoundEngine never advances the exchange clock, so every event would carry ts=0;
        # keep it on the round clock so the trade tape has real timestamps.
        self.exchange.now = max(self.exchange.now, t)
        super()._tick(t)

    def fair(self, symbol: str, step: int | None = None) -> float:
        ex = self.exchange
        m = ex.books[symbol].mid()
        if m is not None:
            return m / 100.0
        last = ex.last_price.get(symbol)
        if last:
            return last / 100.0
        return self._initial[symbol]

    def current_bar(self) -> int:
        """Output bar containing the current instant; bar i covers (i*bar_ms, (i+1)*bar_ms]."""
        u = self.now - self.cfg.pre_open_ms
        return min(max(math.ceil(u / self.bar_ms) - 1, 0), self.n_bars - 1)

    def target_view(self, bot) -> ScheduleView:
        from .bots import TargetTrader

        if not isinstance(bot, TargetTrader):
            raise PermissionError(f"{type(bot).__name__} may not read the target schedule")
        return ScheduleView(self, bot.ctx.account)
