"""Closed-form reference market: latent fair value, partial-adjustment price.

Per symbol, on a 1-minute grid (log prices)::

    f_t = f_{t-1} + mu + sigma_f * eps_t                    (fair value, the ground truth)
    p_t = p_{t-1} + k * (f_{t-1} - p_{t-1}) + sigma_n * eta_t   (traded price)

so the mispricing ``m_t = f_t - p_t`` is AR(1) with ``phi = 1 - k``::

    m_t = phi * m_{t-1} + mu + sigma_f * eps_t - sigma_n * eta_t,   V_m = (sigma_f^2 + sigma_n^2) / (1 - phi^2)

and one-minute returns ``r_t = k * m_{t-1} + sigma_n * eta_t`` have autocovariance::

    gamma_0 = k^2 V_m + sigma_n^2,     gamma_l = k^2 phi^l V_m - k sigma_n^2 phi^(l-1)   (l >= 1)

Symbols are i.i.d. and start in the stationary law, so at every timestamp the cross-section of
(factor, forward return) is an i.i.d. bivariate Gaussian sample. With Pearson ``rho`` the expected
Spearman correlation of N draws is (Moran 1948)::

    E[r_s] = 6 / (pi (N + 1)) * (arcsin(rho) + (N - 2) arcsin(rho / 2))

``k = 0`` is the placebo world: price is a random walk unrelated to fair value.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CALENDAR_START = pd.Timestamp("2024-01-01 00:00")


@dataclass
class RefWorld:
    name: str = "ref"
    n_symbols: int = 16
    minutes: int = 4 * 1440
    k: float = 0.02
    mu: float = 0.0
    sigma_f: float = 0.0010
    sigma_n: float = 0.0005
    placebo_m0_sd: float = 0.01
    wick_sd: float = 0.0002
    initial_prices: list[float] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RefWorld":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        return cls(**raw)

    @property
    def phi(self) -> float:
        return 1.0 - self.k

    @property
    def var_m(self) -> float:
        if self.k <= 0:
            return float("inf")
        return (self.sigma_f ** 2 + self.sigma_n ** 2) / (1.0 - self.phi ** 2)

    def symbols(self) -> list[str]:
        return [f"R{i:02d}" for i in range(self.n_symbols)]


def simulate(world: RefWorld, seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Returns (bars, truth) dicts keyed by symbol, both on the same continuous 1-minute index."""
    if not 0.0 <= world.k < 1.0:
        raise ValueError("k must be in [0, 1)")
    rng = np.random.default_rng(seed)
    T = world.minutes
    idx = pd.date_range(CALENDAR_START, periods=T, freq="1min")
    bars, truth = {}, {}
    for i, sym in enumerate(world.symbols()):
        p0 = world.initial_prices[i] if i < len(world.initial_prices) else float(rng.uniform(20.0, 200.0))
        eps = rng.standard_normal(T)
        eta = rng.standard_normal(T)
        m0 = (math.sqrt(world.var_m) * rng.standard_normal() + world.mu / world.k) if world.k > 0 \
            else world.placebo_m0_sd * rng.standard_normal()
        f = np.empty(T)
        p = np.empty(T)
        p[0] = math.log(p0)
        f[0] = p[0] + m0
        for t in range(1, T):
            f[t] = f[t - 1] + world.mu + world.sigma_f * eps[t]
            p[t] = p[t - 1] + world.k * (f[t - 1] - p[t - 1]) + world.sigma_n * eta[t]
        close = np.exp(p)
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(world.wick_sd * rng.standard_normal(T)) * close
        bars[sym] = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": rng.integers(50, 500, T).astype(float),
        }, index=idx)
        truth[sym] = pd.DataFrame({"fair": np.exp(f), "mispricing": f - p}, index=idx)
    return bars, truth


def write_world(world: RefWorld, seed: int, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    bars, truth = simulate(world, seed)
    for sym in bars:
        bars[sym].to_parquet(out / "data" / f"{sym}_1m.parquet")
        truth[sym].to_parquet(out / "truth" / f"{sym}_1m.parquet")
    (out / "world.json").write_text(json.dumps({"seed": seed, **asdict(world)}, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Analytics                                                                    #
# --------------------------------------------------------------------------- #


def return_autocov(world: RefWorld, lag: int) -> float:
    k, phi, sn2 = world.k, world.phi, world.sigma_n ** 2
    if k == 0:
        return sn2 if lag == 0 else 0.0
    vm = world.var_m
    if lag == 0:
        return k * k * vm + sn2
    return k * k * phi ** lag * vm - k * sn2 * phi ** (lag - 1)


def _sum_var(world: RefWorld, n: int) -> float:
    """Var of a sum of n consecutive one-minute returns."""
    return n * return_autocov(world, 0) + 2.0 * sum((n - l) * return_autocov(world, l) for l in range(1, n))


def oracle_pearson(world: RefWorld, horizon_min: int) -> float:
    """corr(m_t, p_{t+h} - p_t)."""
    if world.k == 0:
        return 0.0
    cov = (1.0 - world.phi ** horizon_min) * world.var_m
    return cov / math.sqrt(world.var_m * _sum_var(world, horizon_min))


def oracle_pearson_closed_form_var(world: RefWorld, horizon_min: int) -> float:
    """Same as :func:`oracle_pearson` but with Var(R) from the fair/mispricing decomposition (self-check)."""
    k, phi, h, vm, sf2 = world.k, world.phi, horizon_min, world.var_m, world.sigma_f ** 2
    var_r = h * sf2 + 2 * vm * (1 - phi ** h) - 2 * sf2 * (1 - phi ** h) / k
    return (1.0 - phi ** h) * vm / math.sqrt(vm * var_r)


def momentum_pearson(world: RefWorld, lookback_min: int, horizon_min: int) -> float:
    """corr(p_t - p_{t-L}, p_{t+h} - p_t)."""
    if world.k == 0:
        return 0.0
    cov = sum(return_autocov(world, a + b) for a in range(1, horizon_min + 1) for b in range(lookback_min))
    return cov / math.sqrt(_sum_var(world, lookback_min) * _sum_var(world, horizon_min))


def expected_spearman(rho: float, n: int) -> float:
    """Expected sample Spearman correlation of n i.i.d. bivariate-normal draws with Pearson rho."""
    rho = max(-1.0, min(1.0, rho))
    return 6.0 / (math.pi * (n + 1)) * (math.asin(rho) + (n - 2) * math.asin(rho / 2.0))


def mispricing_half_life_min(world: RefWorld) -> float:
    return math.log(2.0) / -math.log(world.phi) if world.k > 0 else float("inf")
