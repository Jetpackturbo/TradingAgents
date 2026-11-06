"""Run a world: one EndogenousRoundEngine per symbol, optionally in parallel processes."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import bots as _bots  # noqa: F401  (registers bot types)
from .bars import minute_index, trades_to_bars
from .engine import EndogenousRoundEngine
from .schedule import generate_schedule, initial_prices
from .world import WorldConfig


def build_schedules(world: WorldConfig, seed: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    syms = world.symbol_list
    p0 = initial_prices(syms, seed, world.price_min, world.price_max)
    sched = {s: generate_schedule(world.schedule, p0[s], world.n_bars, seed, s) for s in syms}
    return p0, sched


def simulate_symbol(world: WorldConfig, seed: int, symbol: str, p0: float, schedule: np.ndarray,
                    return_engine: bool = False) -> dict:
    """Simulate one symbol. Deterministic in (world, seed, symbol, schedule)."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    cfg = world.round_config(symbol, p0, seed)
    eng = EndogenousRoundEngine(cfg, bar_ms=world.bar_ms, schedule={symbol: schedule})
    eng.start()
    eng.run_to_end()
    invariants_ok = True
    try:
        eng.exchange.check_invariants()
    except AssertionError:
        invariants_ok = False
    trades = [tr for tr in eng.trade_log if tr["symbol"] == symbol]
    ts = np.fromiter((tr["ts"] for tr in trades), dtype=np.int64, count=len(trades))
    px = np.fromiter((tr["price"] for tr in trades), dtype=np.int64, count=len(trades))
    q = np.fromiter((tr["qty"] for tr in trades), dtype=np.int64, count=len(trades))
    bars = trades_to_bars(ts, px, q, n_bars=world.n_bars, bar_ms=world.bar_ms,
                          pre_open_ms=cfg.pre_open_ms, p0=p0)
    kinds = {a: type(b).__name__ for a, b in eng.bots.by_account.items()}
    by_type: dict[str, int] = {}
    for tr in trades:
        for acct in (tr["buyer"], tr["seller"]):
            kind = kinds.get(acct, acct)
            by_type[kind] = by_type.get(kind, 0) + tr["qty"]
    out = {
        "symbol": symbol,
        "bars": bars,
        "n_trades": len(trades),
        "volume_by_bot": by_type,
        "schedule_reads": dict(eng.schedule_reads),
        "bot_types": sorted({type(b).__name__ for b in eng.bots.bots}),
        "invariants_ok": invariants_ok,
        "wall_s": time.perf_counter() - t0,
        "cpu_s": time.process_time() - c0,
    }
    if return_engine:
        out["engine"] = eng
    return out


def _worker(args) -> dict:
    world_json, seed, symbol, p0, schedule = args
    world = WorldConfig.model_validate_json(world_json)
    return simulate_symbol(world, seed, symbol, p0, schedule)


def run_world(world: WorldConfig, seed: int, workers: int | None = None) -> tuple[dict, dict, list[dict]]:
    p0, sched = build_schedules(world, seed)
    syms = world.symbol_list
    workers = workers if workers is not None else (world.workers or min(len(syms), os.cpu_count() or 1))
    tasks = [(world.model_dump_json(), seed, s, p0[s], sched[s]) for s in syms]
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        results: list[dict | None] = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_worker, t) for t in tasks]
            for i, f in enumerate(futs):
                try:
                    results[i] = f.result()
                except Exception as exc:  # e.g. transient DLL-load block in a fresh worker
                    print(f"[synthmarket] worker failed for {syms[i]} ({exc!r}); retrying in-process")
                    results[i] = _worker(tasks[i])
    return p0, sched, results


def simulate_world(world_cfg: WorldConfig | str | Path, seed: int, out_dir: Path | str,
                   workers: int | None = None) -> dict:
    """Simulate ``world_cfg`` with ``seed`` and write ``out_dir/{data,truth}/{SYM}_1m.parquet``."""
    world = world_cfg if isinstance(world_cfg, WorldConfig) else WorldConfig.from_yaml(world_cfg)
    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    p0, sched, results = run_world(world, seed, workers)
    wall = time.perf_counter() - t0
    idx = minute_index(world.calendar_start, world.n_bars)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    per_sym = {}
    for r in results:
        s = r["symbol"]
        df = pd.DataFrame(r["bars"], index=idx)[["open", "high", "low", "close", "volume"]].astype(float)
        fair = sched[s].astype(float)
        truth = pd.DataFrame({"fair": fair, "mispricing": np.log(fair) - np.log(df["close"].to_numpy())},
                             index=idx)
        df.to_parquet(out_dir / "data" / f"{s}_1m.parquet")
        truth.to_parquet(out_dir / "truth" / f"{s}_1m.parquet")
        per_sym[s] = {
            "initial_price": p0[s],
            "trades": r["n_trades"],
            "empty_bar_frac": float((df["volume"] == 0).mean()),
            "volume_by_bot": r["volume_by_bot"],
            "schedule_reads": r["schedule_reads"],
            "bot_types": r["bot_types"],
            "invariants_ok": r["invariants_ok"],
            "cpu_s": round(r["cpu_s"], 3),
        }
    cpu = sum(r["cpu_s"] for r in results)
    n_sym_minutes = world.n_bars * len(results)
    summary = {
        "world": world.name,
        "seed": seed,
        "symbols": len(results),
        "bars": world.n_bars,
        "bar_ms": world.bar_ms,
        "tick_ms": world.tick_ms,
        "calendar": [str(idx[0]), str(idx[-1])],
        "trades_total": int(sum(r["n_trades"] for r in results)),
        "trades_per_symbol_mean": float(np.mean([r["n_trades"] for r in results])),
        "invariants_ok": all(r["invariants_ok"] for r in results),
        "wall_s": round(wall, 2),
        "cpu_s_total": round(cpu, 2),
        "sim_minutes_per_cpu_s": round(n_sym_minutes / cpu, 1) if cpu > 0 else None,
        "sim_minutes_per_wall_s": round(n_sym_minutes / wall, 1) if wall > 0 else None,
        "per_symbol": per_sym,
    }
    return summary
