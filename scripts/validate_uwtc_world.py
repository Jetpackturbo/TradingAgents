"""Validation metrics for a generated world-seed directory (data/ + truth/).

* trades per symbol (mean; from summary.json if present)
* fraction of empty 1-minute bars
* mean |mispricing| (bps) and median AR(1) half-life (minutes) of 5-min-sampled mispricing
* oracle IC: mean over 5-minute timestamps of the cross-sectional Spearman correlation
  between mispricing and the forward 30-minute simple return of close
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes, vols, mis = {}, {}, {}
    for f in sorted((d / "data").glob("*_1m.parquet")):
        s = f.name[: -len("_1m.parquet")]
        bars = pd.read_parquet(f)
        truth = pd.read_parquet(d / "truth" / f.name)
        assert bars.index.equals(truth.index)
        closes[s], vols[s], mis[s] = bars["close"], bars["volume"], truth["mispricing"]
    return pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(mis)


def ar1_half_life(x: np.ndarray, dt_min: float) -> tuple[float, float]:
    y, x0 = x[1:], x[:-1]
    A = np.c_[np.ones_like(x0), x0]
    phi = float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    hl = -np.log(2) / np.log(phi) * dt_min if 0 < phi < 1 else float("inf")
    return phi, hl


def cs_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    ra, rb = a.rank(axis=1), b.rank(axis=1)
    ok = ra.notna() & rb.notna()
    ra, rb = ra.where(ok), rb.where(ok)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return (num / den).where(ok.sum(axis=1) >= 3)


def metrics(d: Path, horizon: int = 30, step: int = 5, warmup: int = 0) -> dict:
    close, vol, mis = load(d)
    out: dict = {"dir": str(d), "symbols": close.shape[1], "minutes": close.shape[0]}
    sj = d / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text())
        out["trades_per_symbol_mean"] = s["trades_per_symbol_mean"]
    out["empty_bar_frac"] = float((vol == 0).to_numpy().mean())
    out["mean_abs_mispricing_bps"] = float(mis.abs().to_numpy().mean() * 1e4)
    m5 = mis.iloc[::step]
    phis, hls = [], []
    for c in m5:
        phi, hl = ar1_half_life(m5[c].to_numpy(), step)
        phis.append(phi)
        hls.append(hl)
    out["ar1_phi_5m_median"] = float(np.median(phis))
    out["half_life_min_median"] = float(np.median(hls))
    fwd = close.shift(-horizon) / close - 1.0
    ic = cs_spearman(m5, fwd.iloc[::step]).iloc[warmup // step:].dropna()
    out["oracle_ic_mean"] = float(ic.mean())
    out["oracle_ic_std"] = float(ic.std())
    out["oracle_ic_n"] = int(len(ic))
    # Non-overlapping horizons for an honest t-stat.
    ic_no = ic.iloc[:: horizon // step]
    out["oracle_ic_t_nonoverlap"] = float(ic_no.mean() / ic_no.std() * np.sqrt(len(ic_no)))
    # Diagnostics: bid-ask-bounce-only signal (-1m return) and price/target co-movement.
    rev = -(np.log(close) - np.log(close.shift(1)))
    ric = cs_spearman(rev.iloc[::step], fwd.iloc[::step]).iloc[warmup // step:].dropna()
    out["reversal_1m_ic_mean"] = float(ric.mean())
    past = close / close.shift(horizon) - 1.0
    mic = cs_spearman(past.iloc[::step], fwd.iloc[::step]).iloc[warmup // step:].dropna()
    out["momentum_30m_ic_mean"] = float(mic.mean())
    logf = np.log(close) + mis
    dp = (np.log(close) - np.log(close.shift(60))).iloc[60::60].to_numpy().ravel()
    dfv = (logf - logf.shift(60)).iloc[60::60].to_numpy().ravel()
    out["corr_60m_ret_price_vs_fair"] = float(np.corrcoef(dp, dfv)[0, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--warmup", type=int, default=0, help="minutes to skip before computing IC")
    args = ap.parse_args()
    for d in args.dirs:
        print(json.dumps(metrics(Path(d), warmup=args.warmup), indent=2))


if __name__ == "__main__":
    main()
