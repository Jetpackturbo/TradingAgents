"""Throughput probe: simulate one symbol of a tiny world in-process and time it."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmarket.simulate import build_schedules, simulate_symbol  # noqa: E402
from synthmarket.world import WorldConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--bars-per-day", type=int, default=240)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--bar-ms", type=int, default=None)
    ap.add_argument("--tick-ms", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    w = WorldConfig.from_yaml(args.world)
    upd = {"bars_per_day": args.bars_per_day, "days": args.days, "n_symbols": 1, "symbols": None}
    if args.bar_ms:
        upd["bar_ms"] = args.bar_ms
    if args.tick_ms:
        upd["tick_ms"] = args.tick_ms
    w = WorldConfig.model_validate({**w.model_dump(), **upd})
    p0, sched = build_schedules(w, args.seed)
    s = w.symbol_list[0]
    t0 = time.perf_counter()
    r = simulate_symbol(w, args.seed, s, p0[s], sched[s])
    dt = time.perf_counter() - t0
    print(f"bars={w.n_bars} bar_ms={w.bar_ms} tick_ms={w.tick_ms} trades={r['n_trades']} "
          f"wall={dt:.2f}s cpu={r['cpu_s']:.2f}s minutes/cpu_s={w.n_bars / r['cpu_s']:.1f} "
          f"invariants_ok={r['invariants_ok']}")
    print("volume_by_bot", r["volume_by_bot"])


if __name__ == "__main__":
    main()
