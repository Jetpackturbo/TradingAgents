"""Ground-truth target schedules ``F_t`` (one value per output minute bar).

``F[i]`` is the target in force during bar ``i``. Schedules are generated from the
world seed before any simulation runs and are visible only to TargetTraders (through
:meth:`synthmarket.engine.EndogenousRoundEngine.target_view`) and to the truth files.
"""

from __future__ import annotations

import zlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

Family = Literal["random_walk", "ramp", "regime"]


class ScheduleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Family = "random_walk"
    # Per-minute vol of log F (random_walk / regime; optional extra noise for ramp).
    sigma_f: float = 0.001
    # Per-minute drift of log F. Each symbol draws mu_i ~ N(mu, mu_std).
    mu: float = 0.0
    mu_std: float = 0.0
    # Per-symbol vol multiplier exp(sigma_dispersion * z) for heterogeneous paths.
    sigma_dispersion: float = 0.0
    # regime: drift magnitude |mu_i| ~ regime_mu * exp(0.5 z); sign flips at Poisson times.
    regime_mu: float = 2e-4
    regime_mean_bars: float = 720.0


def stable_seed(*parts) -> list[int]:
    """Process-independent seed sequence (never uses Python's salted ``hash``)."""
    return [zlib.crc32(str(p).encode()) & 0xFFFFFFFF for p in parts]


def generate_schedule(spec: ScheduleSpec, f0: float, n_bars: int, seed: int, symbol: str) -> np.ndarray:
    """Target path of length ``n_bars`` (dollars) with ``F[0] == f0``."""
    rng = np.random.default_rng(stable_seed("schedule", seed, symbol))
    sig = spec.sigma_f * float(np.exp(spec.sigma_dispersion * rng.standard_normal()))
    mu = spec.mu + spec.mu_std * float(rng.standard_normal())
    z = rng.standard_normal(n_bars - 1)
    if spec.family == "random_walk":
        steps = mu + sig * z
    elif spec.family == "ramp":
        steps = np.full(n_bars - 1, mu) + sig * z
    elif spec.family == "regime":
        mag = spec.regime_mu * float(np.exp(0.5 * rng.standard_normal()))
        flips = rng.random(n_bars - 1) < 1.0 / max(spec.regime_mean_bars, 1.0)
        sign0 = 1.0 if rng.random() < 0.5 else -1.0
        sign = sign0 * np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
        steps = mu + mag * sign + sig * z
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ValueError(spec.family)
    logf = np.log(f0) + np.concatenate([[0.0], np.cumsum(steps)])
    return np.exp(logf)


def initial_prices(symbols: list[str], seed: int, lo: float, hi: float) -> dict[str, float]:
    out = {}
    for s in symbols:
        rng = np.random.default_rng(stable_seed("price", seed, s))
        out[s] = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)
    return out
