"""Diagnostic sweep of FlowMarketMaker learning settings, applied identically to the informed
and placebo worlds (3-day worlds under runs/worlds/_probe/sweep/)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmarket import WorldConfig, simulate_world  # noqa: E402
from validate_uwtc_world import metrics  # noqa: E402

VARIANTS = {
    "current": {},
    "no_ewma": {"ewma_alpha": 0.0},
    "no_skew": {"skew_bps_per_unit": 0.0},
    "no_ewma_no_skew": {"ewma_alpha": 0.0, "skew_bps_per_unit": 0.0},
}
KEYS = ("oracle_ic_mean", "oracle_ic_t_nonoverlap", "momentum_30m_ic_mean", "reversal_1m_ic_mean",
        "mean_abs_mispricing_bps", "half_life_min_median", "empty_bar_frac", "corr_60m_ret_price_vs_fair")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    for v in args.variants:
        for wname in ("uwtc_informed", "uwtc_placebo"):
            d = WorldConfig.from_yaml(root / "configs" / "worlds" / f"{wname}.yaml").model_dump()
            d["days"] = args.days
            d["market_makers"] = copy.deepcopy(d["market_makers"])
            for mm in d["market_makers"]:
                mm["params"].update(VARIANTS[v])
            out = root / "runs" / "worlds" / "_probe" / "sweep" / f"{v}_{wname}_s{args.seed}"
            simulate_world(WorldConfig.model_validate(d), args.seed, out)
            m = metrics(out)
            print(f"{v:16s} {wname:14s} " + " ".join(f"{k}={m[k]:.3f}" for k in KEYS), flush=True)


if __name__ == "__main__":
    main()
