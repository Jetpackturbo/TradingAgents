"""ETF NAV mechanics, shared by the fair-value model and the house bots.

* Index ETF (``equal_weight_daily``): each day the NAV earns the mean simple
  return of the constituents since the previous close, i.e. it is rebalanced
  to equal weights at every EOD. ``price_weighted`` NAV is simply the mean
  constituent price.
* Leveraged ETF: each day NAV earns ``L`` times the underlying's simple return
  since the previous close (daily reset), which produces volatility decay.
"""

from __future__ import annotations

import numpy as np


def index_nav(prices_now: np.ndarray | list[float], prices_open: np.ndarray | list[float], nav_open: float,
              weighting: str = "equal_weight_daily") -> float:
    p = np.asarray(prices_now, dtype=float)
    if weighting == "price_weighted":
        return float(p.mean())
    p0 = np.asarray(prices_open, dtype=float)
    return float(nav_open * (1.0 + np.mean(p / p0 - 1.0)))


def letf_nav(price_now: float, price_open: float, nav_open: float, leverage: float) -> float:
    return float(nav_open * (1.0 + leverage * (price_now / price_open - 1.0)))


def index_nav_path(prices: np.ndarray, steps_per_day: int, weighting: str) -> np.ndarray:
    """``prices``: shape (n_constituents, T+1). Returns NAV path of length T+1."""
    n, T1 = prices.shape
    if weighting == "price_weighted":
        return prices.mean(axis=0)
    nav = np.empty(T1)
    nav[0] = prices[:, 0].mean()
    nav_open = nav[0]
    for t0 in range(0, T1 - 1, steps_per_day):
        t1 = min(t0 + steps_per_day, T1 - 1)
        seg = prices[:, t0:t1 + 1] / prices[:, [t0]] - 1.0
        nav[t0:t1 + 1] = nav_open * (1.0 + seg.mean(axis=0))
        nav_open = nav[t1]
    return nav


def letf_nav_path(underlying: np.ndarray, steps_per_day: int, leverage: float, initial_nav: float) -> np.ndarray:
    T1 = underlying.shape[0]
    nav = np.empty(T1)
    nav_open = initial_nav
    nav[0] = initial_nav
    for t0 in range(0, T1 - 1, steps_per_day):
        t1 = min(t0 + steps_per_day, T1 - 1)
        seg = underlying[t0:t1 + 1] / underlying[t0] - 1.0
        nav[t0:t1 + 1] = nav_open * (1.0 + leverage * seg)
        nav_open = nav[t1]
    return nav


def letf_rebalance_shares(aum_open: float, leverage: float, underlying_open: float, underlying_now: float) -> float:
    """Shares of the underlying a daily-reset leveraged fund must trade at the
    close to restore its target exposure (positive = buy).

    Exposure before rebalance: L*AUM0*(1+r). Target: L*AUM0*(1+L*r).
    Required trade notional: AUM0 * L * (L - 1) * r.
    """
    r = underlying_now / underlying_open - 1.0
    return aum_open * leverage * (leverage - 1.0) * r / underlying_now


def index_rebalance_shares(aum_open: float, prices_open: np.ndarray, prices_now: np.ndarray) -> np.ndarray:
    """Shares per constituent an equal-weight fund trades at the close to
    return to equal weights (positive = buy)."""
    p0 = np.asarray(prices_open, dtype=float)
    p1 = np.asarray(prices_now, dtype=float)
    n = len(p0)
    holdings = (aum_open / n) / p0
    aum_now = float(holdings @ p1)
    target = (aum_now / n) / p1
    return target - holdings
