"""Ground-truth evaluator for factors on synthetic worlds.

A world is a set of splits that all follow the data contract written by ``synthmarket``::

    <split>/data/{SYM}_1m.parquet    open, high, low, close, volume  (continuous 1-minute DatetimeIndex)
    <split>/truth/{SYM}_1m.parquet   fair, mispricing                  (same index)

``proxy`` is QuantEvolver's cross-sectional RankIC on the training split over the task window (the only
quantity a trainer may see). ``gold`` is the same statistic on held-out re-simulations of the world
(fresh seeds), and ``mispricing_corr`` is the cross-sectional rank correlation between the factor and the
true mispricing ``log(fair) - log(close)`` on those held-out seeds.

Per-timestamp RankIC replicates ``quant_evolver.evaluation.cross_sectional_rankic`` exactly (5-minute
right-closed resampling, ``close.shift(-h) / close - 1`` forward returns, average-rank Spearman, timestamps
with fewer than ``min_symbols_per_time`` joint observations or constant ranks skipped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

FactorFn = Callable[["Split"], pd.DataFrame]


def resample_bars(df_1m: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    rule = f"{int(bar_minutes)}min"
    return pd.DataFrame({
        "open": df_1m["open"].astype(float).resample(rule, label="right", closed="right").first(),
        "high": df_1m["high"].astype(float).resample(rule, label="right", closed="right").max(),
        "low": df_1m["low"].astype(float).resample(rule, label="right", closed="right").min(),
        "close": df_1m["close"].astype(float).resample(rule, label="right", closed="right").last(),
        "volume": df_1m["volume"].astype(float).resample(rule, label="right", closed="right").sum(),
    }).dropna()


@dataclass
class Split:
    path: Path
    bars: dict[str, pd.DataFrame]
    close: pd.DataFrame
    mispricing: pd.DataFrame | None
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @classmethod
    def load(cls, path: str | Path, bar_minutes: int = 5) -> "Split":
        path = Path(path)
        files = sorted((path / "data").glob("*_1m.parquet"))
        if not files:
            raise FileNotFoundError(f"no *_1m.parquet under {path / 'data'}")
        bars, closes, mis = {}, {}, {}
        for f in files:
            sym = f.name[: -len("_1m.parquet")]
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
            b = resample_bars(df.sort_index(), bar_minutes)
            bars[sym] = b
            closes[sym] = b["close"]
            tf = path / "truth" / f.name
            if tf.exists():
                t = pd.read_parquet(tf)
                t.index = pd.to_datetime(t.index)
                mis[sym] = t["mispricing"].astype(float).resample(
                    f"{int(bar_minutes)}min", label="right", closed="right").last().reindex(b.index)
        close = pd.DataFrame(closes).sort_index()
        mispricing = pd.DataFrame(mis).reindex(close.index) if len(mis) == len(bars) else None
        return cls(path, bars, close, mispricing, close.index[0], close.index[-1])


def rowwise_rank_ic(factor: pd.DataFrame, target: pd.DataFrame, min_symbols: int) -> pd.Series:
    """Per-timestamp Spearman between two aligned panels (rows = time, cols = symbols)."""
    factor = factor.replace([np.inf, -np.inf], np.nan)
    target = target.replace([np.inf, -np.inf], np.nan)
    idx = factor.index.intersection(target.index)
    cols = [c for c in factor.columns if c in target.columns]
    f = factor.loc[idx, cols]
    t = target.loc[idx, cols]
    mask = f.notna() & t.notna()
    fr = f.where(mask).rank(axis=1, method="average")
    tr = t.where(mask).rank(axis=1, method="average")
    n = mask.sum(axis=1)
    fd = fr.sub(fr.mean(axis=1), axis=0)
    td = tr.sub(tr.mean(axis=1), axis=0)
    num = (fd * td).sum(axis=1)
    sf = (fd ** 2).sum(axis=1)
    st = (td ** 2).sum(axis=1)
    ok = (n >= min_symbols) & (np.sqrt(sf / n.clip(lower=1)) > 1e-12) & (np.sqrt(st / n.clip(lower=1)) > 1e-12)
    ic = num / np.sqrt(sf * st)
    return ic[ok].astype(float)


def block_bootstrap_se(x: np.ndarray, block: int, n_boot: int = 1000, seed: int = 0) -> float:
    """Moving-block bootstrap standard error of the mean of a serially correlated series."""
    n = len(x)
    if n < 2:
        return float("nan")
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offs = np.arange(block)
    draws = x[(starts[:, :, None] + offs).reshape(n_boot, -1)[:, :n]]
    return float(draws.mean(axis=1).std(ddof=1))


@dataclass
class SynthEvaluator:
    train_dir: str | Path
    gold_dirs: list[str | Path] = field(default_factory=list)
    bar_minutes: int = 5
    horizon_bars: int = 6
    warmup_days: float = 1.0
    min_symbols_per_time: int = 8
    min_times: int = 20
    block_bars: int | None = None
    n_boot: int = 1000
    # Max cached (expr, split) factor panels; the cache is dropped when full (long RL runs see many exprs).
    cache_size: int = 512

    def __post_init__(self) -> None:
        self.train = Split.load(self.train_dir, self.bar_minutes)
        self.gold = [Split.load(d, self.bar_minutes) for d in self.gold_dirs]
        if self.block_bars is None:
            self.block_bars = 4 * self.horizon_bars
        self.dsl_warmup = 240 if int(self.bar_minutes) == 5 else 96
        self._dsl_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------ factors

    def dsl_factor(self, expr: str, split: Split) -> pd.DataFrame:
        """Factor panel from a QuantEvolver DSL expression (QuantEvolver's own interpreter), cached."""
        from quant_evolver.dsl.evaluator import evaluate_expr_series

        key = (expr, str(split.path))
        if key not in self._dsl_cache:
            if len(self._dsl_cache) >= self.cache_size:
                self._dsl_cache.clear()
            cols = {}
            for sym, b in split.bars.items():
                try:
                    cols[sym] = evaluate_expr_series(expr, b, warmup_bars=self.dsl_warmup).astype(float)
                except Exception:
                    continue
            self._dsl_cache[key] = pd.DataFrame(cols).reindex(split.close.index)
        return self._dsl_cache[key]

    @staticmethod
    def validate_expr(expr: str, warmup: int = 240) -> str | None:
        from quant_evolver.dsl.compiler import compile_expr

        try:
            compile_expr(expr, warmup_param=warmup)
        except Exception as exc:
            return str(exc)
        return None

    # ------------------------------------------------------------------ scoring

    def forward_returns(self, split: Split, end: pd.Timestamp | None = None) -> pd.DataFrame:
        close = split.close if end is None else split.close[split.close.index <= end]
        return close.shift(-int(self.horizon_bars)) / close - 1.0

    def default_window(self, split: Split) -> tuple[pd.Timestamp, pd.Timestamp]:
        return split.start + pd.Timedelta(days=self.warmup_days), split.end

    def score_split(self, factor: pd.DataFrame, split: Split, start=None, end=None) -> dict | None:
        s0, e0 = self.default_window(split)
        start = pd.Timestamp(start) if start is not None else s0
        end = pd.Timestamp(end) if end is not None else e0
        fwd = self.forward_returns(split, end)
        fwd = fwd[fwd.index >= start]
        fac = factor[(factor.index >= start) & (factor.index <= end)]
        ic = rowwise_rank_ic(fac, fwd, self.min_symbols_per_time)
        if len(ic) < self.min_times:
            return None
        arr = ic.to_numpy()
        out = {
            "rank_ic": float(arr.mean()),
            "rank_ic_std": float(arr.std()),
            "n_times": int(len(arr)),
            "se": block_bootstrap_se(arr, self.block_bars, self.n_boot),
            "icir": float(arr.mean() / (arr.std() + 1e-8) * np.sqrt(len(arr))),
        }
        if split.mispricing is not None:
            mic = rowwise_rank_ic(fac, split.mispricing, self.min_symbols_per_time)
            out["mispricing_corr"] = float(mic.mean()) if len(mic) else float("nan")
        return out

    def evaluate_panels(self, factor_fn: FactorFn, start=None, end=None) -> dict:
        """Score ``factor_fn(split) -> panel`` on the train window (proxy) and on every gold split (gold)."""
        tr = self.score_split(factor_fn(self.train), self.train, start, end)
        if tr is None:
            return {"valid": False, "reason": "too_few_valid_times"}
        golds = [self.score_split(factor_fn(g), g) for g in self.gold]
        golds = [g for g in golds if g is not None]
        out = {
            "valid": True,
            "proxy": tr["rank_ic"],
            "se": tr["se"],
            "proxy_n_times": tr["n_times"],
            "qe_score": float(np.clip(tr["rank_ic"] * 5.0 + 0.02 * np.tanh(tr["icir"]), -1.0, 1.0)),
            "train_mispricing_corr": tr.get("mispricing_corr", float("nan")),
        }
        if golds:
            g = np.array([x["rank_ic"] for x in golds])
            out.update({
                "gold": float(g.mean()),
                "gold_se": float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan"),
                "gold_n_seeds": int(len(g)),
                "gold_seed_rank_ic": [float(v) for v in g],
                "gold_seed_se": [float(x["se"]) for x in golds],
                "mispricing_corr": float(np.nanmean([x.get("mispricing_corr", np.nan) for x in golds])),
            })
        return out

    def __call__(self, expr: str, start=None, end=None) -> dict | None:
        """GroundTruthReward contract: ``{"proxy", "se", "gold", ...}`` or ``None`` if the expression is invalid."""
        if not expr or self.validate_expr(expr, self.dsl_warmup) is not None:
            return None
        out = self.evaluate_panels(lambda sp: self.dsl_factor(expr, sp), start, end)
        return out if out["valid"] else None


def build_evaluator(train_dir: str, gold_dirs: list[str], **kw) -> SynthEvaluator:
    return SynthEvaluator(train_dir=train_dir, gold_dirs=list(gold_dirs), **kw)
