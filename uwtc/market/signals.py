"""Signal generation.

For a target information coefficient ``ic`` a signal score is::

    score = ic * z + sqrt(1 - ic^2) * eps,     eps ~ N(0, 1)

where ``z`` is the standardised realised forward log return. Then
``corr(score, z) = ic`` and, for Gaussian returns, ``E[z | score] = ic * score``,
so the published ``expected_return`` is the calibrated conditional forecast.

Noise draws are keyed by (seed, product, tier/team, day, symbol) so a team's
signal never depends on what other teams did or on message ordering.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from ..config import RoundConfig
from .fair_value import MarketPaths

_PRODUCT_IDS = {"equity": 1, "futures": 2}
_TIER_IDS = {"base": 1, "premium": 2}


def _stable_id(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def decayed_ic(ic0: float, decay_per_day: float, day: int) -> float:
    return ic0 * decay_per_day ** (day - 1)


class SignalEngine:
    def __init__(self, cfg: RoundConfig, paths: MarketPaths) -> None:
        self.cfg = cfg
        self.paths = paths
        self.sc = cfg.signals
        self.seed = cfg.seed
        self._sym_ids = {s: i for i, s in enumerate(sorted(paths.equities))}

    # ---------------------------------------------------------------- #

    def _noise(self, product: str, key: str, day: int, symbol: str) -> float:
        seq = [self.seed & 0xFFFFFFFF, _PRODUCT_IDS[product], _stable_id(key), day, _stable_id(symbol)]
        return float(np.random.default_rng(seq).standard_normal())

    def _noise_key(self, tier: str, team: str) -> str:
        if self.sc.noise_mode == "per_team":
            return f"team:{team}"
        return f"tier:{tier}"

    def forward_return(self, symbol: str, day: int, end_day: int) -> tuple[float, float, float]:
        """(realised log return, mean, std) from open of ``day`` to close of ``end_day``."""
        p = self.paths
        r = math.log(p.equities[symbol][p.close_step(end_day)] / p.equities[symbol][p.open_step(day)])
        h = end_day - day + 1
        return r, p.daily_drift[symbol] * h, math.sqrt(p.daily_var[symbol] * h)

    def tier_ic(self, tier: str) -> float:
        return self.sc.premium_ic if tier == "premium" else self.sc.base_ic

    def effective_ic(self, tier: str, day: int, boost: float = 0.0, cap: float = 0.99) -> float:
        ic = decayed_ic(self.tier_ic(tier), self.sc.decay_per_day, day) + boost
        return float(min(max(ic, 0.0), cap, 0.99))

    @staticmethod
    def _score(ic: float, z: float, eps: float) -> float:
        return ic * z + math.sqrt(max(0.0, 1.0 - ic * ic)) * eps

    def equity_signals(self, day: int, tier: str, team: str, ic: float) -> list[dict]:
        h = self.sc.horizon_days
        end = min(day + h - 1, self.cfg.days)
        out = []
        key = self._noise_key(tier, team)
        for s in self.cfg.instruments.equities:
            r, mu, sd = self.forward_return(s, day, end)
            z = (r - mu) / sd
            score = self._score(ic, z, self._noise("equity", key, day, s))
            sig = {"symbol": s, "kind": "equity", "day": day, "horizon_days": end - day + 1,
                   "score": round(score, 6), "tier": tier}
            if self.sc.disclose_strength:
                sig["strength"] = round(ic, 4)
                sig["expected_return"] = round(mu + ic * score * sd, 6)
            out.append(sig)
        return out

    def futures_signals(self, day: int, team: str, futures: list[tuple[str, str, int]]) -> list[dict]:
        """``futures``: (symbol, underlying, expiry_day) for contracts live on ``day``.

        Forecasts the settlement price of each contract (fair close of expiry day)."""
        ic = self.sc.futures_ic
        key = self._noise_key("futures", team)
        out = []
        for sym, und, exp in futures:
            if exp < day:
                continue
            r, mu, sd = self.forward_return(und, day, exp)
            z = (r - mu) / sd
            score = self._score(ic, z, self._noise("futures", key, day, sym))
            p0 = self.paths.equities[und][self.paths.open_step(day)]
            sig = {"symbol": sym, "kind": "futures", "day": day, "underlying": und, "expiry_day": exp,
                   "score": round(score, 6), "tier": "futures"}
            if self.sc.disclose_strength:
                exp_r = mu + ic * score * sd
                sig["strength"] = round(ic, 4)
                sig["expected_return"] = round(exp_r, 6)
                sig["expected_settlement"] = round(float(p0 * math.exp(exp_r)), 4)
            out.append(sig)
        return out

    def commodity_quote(self, step: int) -> dict[str, float]:
        lead = self.sc.commodity_feed_lead_steps
        return {c: round(self.paths.price(c, step + lead), 4) for c in self.paths.commodities}
