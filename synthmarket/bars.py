"""Trade tape -> 1-minute OHLCV bars on a continuous calendar.

Output bar ``i`` covers sim time ``u in (i*bar_ms, (i+1)*bar_ms]`` (``u`` = round ms minus
``pre_open_ms``; UWTC days run back-to-back with no gap) and is stamped
``calendar_start + i minutes``, the END of its minute. Empty bars carry the previous
close with zero volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def minute_index(calendar_start: str, n_bars: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(calendar_start), periods=n_bars, freq="1min")


def trades_to_bars(ts_ms: np.ndarray, price_cents: np.ndarray, qty: np.ndarray, *, n_bars: int,
                   bar_ms: int, pre_open_ms: int, p0: float) -> dict[str, np.ndarray]:
    """Return float arrays open/high/low/close/volume (dollars) of length ``n_bars``.

    Trades must be in tape order (non-decreasing ts)."""
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    px = np.asarray(price_cents, dtype=np.float64) / 100.0
    q = np.asarray(qty, dtype=np.float64)
    o = np.full(n_bars, np.nan)
    h = np.full(n_bars, np.nan)
    lo = np.full(n_bars, np.nan)
    c = np.full(n_bars, np.nan)
    v = np.zeros(n_bars)
    if len(ts_ms):
        u = ts_ms - pre_open_ms
        b = np.ceil(u / bar_ms).astype(np.int64) - 1
        b = np.clip(b, 0, n_bars - 1)
        np.add.at(v, b, q)
        # first / last trade per bar (tape order preserved)
        starts = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
        ends = np.r_[starts[1:], len(b)] - 1
        bars = b[starts]
        o[bars] = px[starts]
        c[bars] = px[ends]
        h[bars] = np.maximum.reduceat(px, starts)
        lo[bars] = np.minimum.reduceat(px, starts)
    # forward-fill empty bars from the previous close
    close = pd.Series(c).ffill().fillna(p0).to_numpy()
    prev = np.r_[p0, close[:-1]]
    empty = np.isnan(o)
    o[empty] = prev[empty]
    h[empty] = prev[empty]
    lo[empty] = prev[empty]
    c = close
    return {"open": o, "high": h, "low": lo, "close": c, "volume": v}
