"""Generate a reference world: one training split plus held-out gold seeds.

    python scripts/gen_ref_world.py --world configs/worlds/ref_slow.yaml --out runs/worlds/ref_slow --gold-seeds 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthmarket.reference import RefWorld, write_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--gold-seeds", type=int, default=24)
    args = ap.parse_args()
    world = RefWorld.from_yaml(args.world)
    out = Path(args.out)
    write_world(world, args.train_seed, out / "train")
    for s in range(args.gold_seeds):
        write_world(world, 1000 + s, out / "gold" / f"seed_{1000 + s}")
    print(f"{world.name}: train seed {args.train_seed} + {args.gold_seeds} gold seeds -> {out}")


if __name__ == "__main__":
    main()
