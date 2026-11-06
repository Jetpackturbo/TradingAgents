"""Random-walk 1-minute OHLCV in QuantEvolver's format (no signal by construction).

Writes ``{SYM}_1m.parquet`` with a DatetimeIndex and columns open, high, low, close, volume.
The calendar is continuous (24/7 minutes): QuantEvolver's backtrader warmup is
``warmup_days * 1440 // bar_minutes`` bars, so session-only data never leaves warmup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def minute_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1), freq="1min", inclusive="left")


def random_walk_bars(idx: pd.DatetimeIndex, rng: np.random.Generator, p0: float = 100.0,
                     vol_per_min: float = 0.0008) -> pd.DataFrame:
    n = len(idx)
    r = vol_per_min * rng.standard_normal(n)
    close = p0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[p0], close[:-1]])
    wick = np.abs(vol_per_min * rng.standard_normal(n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.integers(50, 500, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/smoke")
    ap.add_argument("--start", default="2023-12-15")
    ap.add_argument("--end", default="2024-02-05")
    ap.add_argument("--n-symbols", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    idx = minute_index(args.start, args.end)
    symbols = ["ASSET"] + [f"S{i:02d}" for i in range(args.n_symbols)]
    for sym in symbols:
        random_walk_bars(idx, rng, p0=float(rng.uniform(20, 200))).to_parquet(out / f"{sym}_1m.parquet")
    print(f"wrote {len(symbols)} symbols x {len(idx)} bars -> {out}")


if __name__ == "__main__":
    main()
