"""Generate a UWTC world in the evaluator layout: ``out/train`` (seed 0) + ``out/gold/seed_{1000+s}``.

    python scripts/gen_uwtc_splits.py --world configs/worlds/uwtc_informed.yaml \
        --out runs/worlds/uwtc_informed --gold-seeds 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmarket import simulate_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--gold-seeds", type=int, default=64)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    jobs = [(args.train_seed, out / "train")]
    jobs += [(1000 + s, out / "gold" / f"seed_{1000 + s}") for s in range(args.gold_seeds)]
    t0 = time.time()
    for seed, path in jobs:
        summary = simulate_world(args.world, seed, path, workers=args.workers)
        (path / "summary.json").write_text(json.dumps(summary, indent=2))
        if not summary.get("invariants_ok", False):
            raise SystemExit(f"invariant check failed for seed {seed} -> {path}")
    print(f"{Path(args.world).stem}: train seed {args.train_seed} + {args.gold_seeds} gold seeds "
          f"-> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
