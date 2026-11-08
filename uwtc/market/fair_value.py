"""Hidden fair-value model.

Per step (``steps_per_day`` steps per day) log returns follow a factor model::

    r_i = mu_i/S + beta_i * m + sum_k gamma_ik * c_k + sigma_i/sqrt(S) * e_i + J_i
    m   = mu_m/S + sigma_m/sqrt(S) * z_m          (market factor)
    c_k = mu_k/S + sigma_k/sqrt(S) * z_k          (commodity log returns)

The whole round is generated up front from the round seed, so runs are exactly
reproducible and signals can be derived from the realised future path.
Index ``t`` runs 0..days*S; step ``(d-1)*S`` is the open of day ``d`` and
``d*S`` its close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RoundConfig
from .etf import index_nav_path, letf_nav_path


@dataclass
class MarketPaths:
    steps_per_day: int
    days: int
    equities: dict[str, np.ndarray]  # prices, length days*S + 1
    commodities: dict[str, np.ndarray]
    market_factor: np.ndarray  # cumulative log market factor
    etf: dict[str, np.ndarray] = field(default_factory=dict)
    daily_var: dict[str, float] = field(default_factory=dict)
    daily_drift: dict[str, float] = field(default_factory=dict)
    betas_vs_index: dict[str, float] = field(default_factory=dict)
    jump_steps: dict[str, list[int]] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return self.days * self.steps_per_day

    def open_step(self, day: int) -> int:
        return (day - 1) * self.steps_per_day

    def close_step(self, day: int) -> int:
        return day * self.steps_per_day

    def price(self, symbol: str, step: int) -> float:
        step = min(max(step, 0), self.total_steps)
        if symbol in self.equities:
            return float(self.equities[symbol][step])
        if symbol in self.etf:
            return float(self.etf[symbol][step])
        if symbol in self.commodities:
            return float(self.commodities[symbol][step])
        raise KeyError(symbol)


def _factor_cov(cfg: RoundConfig) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Daily covariance of equity log returns implied by the factor model."""
    m = cfg.model
    syms = list(cfg.instruments.equities)
    comms = list(m.commodities)
    n = len(syms)
    load = np.zeros((n, 1 + len(comms)))
    fvar = np.zeros(1 + len(comms))
    fvar[0] = m.market_daily_vol ** 2
    for k, c in enumerate(comms):
        fvar[1 + k] = m.commodities[c].daily_vol ** 2
    idio = np.zeros(n)
    for i, s in enumerate(syms):
        em = m.equities[s]
        load[i, 0] = em.beta
        for c, g in em.commodity_loadings.items():
            if c not in m.commodities:
                raise ValueError(f"{s} loads on unknown commodity {c}")
            load[i, 1 + comms.index(c)] = g
        jump_var = m.jumps.rate_per_day * m.jumps.size_std ** 2 if m.jumps.enabled else 0.0
        idio[i] = em.idio_vol ** 2 + jump_var
    cov = load @ np.diag(fvar) @ load.T + np.diag(idio)
    return syms, cov, load


def generate_paths(cfg: RoundConfig, seed: int | None = None) -> MarketPaths:
    m = cfg.model
    S = m.steps_per_day
    T = cfg.days * S
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    sqrtS = np.sqrt(S)

    mkt = m.market_drift / S + m.market_daily_vol / sqrtS * rng.standard_normal(T)
    comm_ret: dict[str, np.ndarray] = {}
    commodities: dict[str, np.ndarray] = {}
    for c, cm in m.commodities.items():
        r = cm.drift / S + cm.daily_vol / sqrtS * rng.standard_normal(T)
        comm_ret[c] = r
        commodities[c] = cm.initial_price * np.exp(np.concatenate([[0.0], np.cumsum(r)]))

    equities: dict[str, np.ndarray] = {}
    jump_steps: dict[str, list[int]] = {}
    for s in cfg.instruments.equities:
        em = m.equities[s]
        r = em.drift / S + em.beta * mkt + em.idio_vol / sqrtS * rng.standard_normal(T)
        for c, g in em.commodity_loadings.items():
            r = r + g * comm_ret[c]
        js: list[int] = []
        if m.jumps.enabled:
            hits = rng.random(T) < m.jumps.rate_per_day / S
            sizes = m.jumps.size_std * rng.standard_normal(T)
            r = r + np.where(hits, sizes, 0.0)
            js = [int(i) for i in np.nonzero(hits)[0]]
        jump_steps[s] = js
        equities[s] = em.initial_price * np.exp(np.concatenate([[0.0], np.cumsum(r)]))

    syms, cov, _ = _factor_cov(cfg)
    daily_var = {s: float(cov[i, i]) for i, s in enumerate(syms)}
    daily_drift = {}
    for s in syms:
        em = m.equities[s]
        drift = em.drift + em.beta * m.market_drift
        for c, g in em.commodity_loadings.items():
            drift += g * m.commodities[c].drift
        daily_drift[s] = drift
    w = np.full(len(syms), 1.0 / len(syms))
    idx_var = float(w @ cov @ w)
    betas = {s: float((cov[i] @ w) / idx_var) for i, s in enumerate(syms)}

    paths = MarketPaths(S, cfg.days, equities, commodities, np.concatenate([[0.0], np.cumsum(mkt)]),
                        daily_var=daily_var, daily_drift=daily_drift, betas_vs_index=betas,
                        jump_steps=jump_steps)
    ic = cfg.instruments
    if ic.etf_enabled:
        paths.etf[ic.etf_symbol] = index_nav_path(
            np.vstack([equities[s] for s in ic.equities]), S, ic.etf_weighting)
    if ic.letf_enabled:
        paths.etf[ic.letf_symbol] = letf_nav_path(
            equities[ic.letf_underlying], S, ic.letf_leverage, ic.letf_initial_nav)
    return paths
