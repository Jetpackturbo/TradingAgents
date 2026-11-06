"""Generate one endogenous-price UWTC world-seed.

    python scripts/gen_uwtc_world.py --world configs/worlds/uwtc_informed.yaml --seed 1 \
        --out runs/worlds/uwtc_informed/seed_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmarket import simulate_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    summary = simulate_world(args.world, args.seed, out, workers=args.workers)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    brief = {k: v for k, v in summary.items() if k != "per_symbol"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
