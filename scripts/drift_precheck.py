"""CPU pre-check for the resting-order drift sanity job (runs on the Hyak node before training).

Confirms that the drift alpha is detectable before any GPU time is spent:

* drift oracle (rank by each symbol's true drift, from ``truth/fair``) has large positive gold,
* QuantEvolver-DSL momentum has large positive gold and its negation the exact opposite,
* QuantEvolver's own cross-sectional RankIC evaluator agrees with our proxy on the same expression,
* a random factor scores 0 and a constant factor is rejected.

Placebo worlds have no drift (the oracle is constant and therefore invalid); their checks only
cover the null factors, and momentum is reported (bounce around a flat target makes it negative).

    python scripts/drift_precheck.py --world-dir runs/worlds/uwtc_resting_drift --kind drift --out precheck.json

Exit code 1 if any drift-world check fails, so the job stops before training.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth_eval.drift import drift_panel  # noqa: E402
from synth_eval.evaluator import Split, SynthEvaluator  # noqa: E402

MOM_EXPR = "div(last(close(49)), first(close(49)))"  # 48 x 5-min bars = 4-hour return
Z_TOL = 3.0


def random_factor(sp: Split) -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32(str(sp.path).encode()))
    return pd.DataFrame(rng.standard_normal(sp.close.shape), index=sp.close.index, columns=sp.close.columns)


def qe_native_rankic(expr: str, train_dir: Path, start: str, end: str) -> float:
    from quant_evolver.evaluation.cross_sectional_rankic import CrossSectionalConfig, evaluate_cross_sectional_expr
    from quant_evolver.evaluation.data import LocalDataConfig

    syms = sorted(f.name[: -len("_1m.parquet")] for f in (train_dir / "data").glob("*_1m.parquet"))
    cfg = CrossSectionalConfig(symbols=syms, start_date=start, end_date=end, bar_minutes=5, horizon_bars=6,
                               warmup_days=1, data=LocalDataConfig(data_dir=str(train_dir / "data")))
    return float(evaluate_cross_sectional_expr(expr, cfg).metrics.get("cross_sectional/rank_ic_mean", np.nan))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-dir", required=True)
    ap.add_argument("--kind", choices=["drift", "placebo"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.world_dir)
    golds = sorted((root / "gold").glob("seed_*"))
    ev = SynthEvaluator(root / "train", golds)
    res = {
        "drift_oracle": ev.evaluate_panels(drift_panel),
        "momentum": ev.evaluate_panels(lambda sp: ev.dsl_factor(MOM_EXPR, sp)),
        "neg_momentum": ev.evaluate_panels(lambda sp: ev.dsl_factor(f"neg({MOM_EXPR})", sp)),
        "random": ev.evaluate_panels(random_factor),
        "constant": ev.evaluate_panels(lambda sp: pd.DataFrame(1.0, index=sp.close.index, columns=sp.close.columns)),
    }
    # QuantEvolver's own evaluator on the same window as our proxy: day 2 through the end of the train
    # split (end date = the day after the last bar, so both include the final day).
    s, e = ev.default_window(ev.train)
    start, end = str(s.date()), str((e + pd.Timedelta(days=1)).date())
    qe = qe_native_rankic(MOM_EXPR, root / "train", start, end)
    ours = ev.evaluate_panels(lambda sp: ev.dsl_factor(MOM_EXPR, sp), start, end)["proxy"]

    def z(r, target=0.0):
        return (r["gold"] - target) / r["gold_se"]

    checks = []

    def check(name, measured, ok, fatal=True):
        checks.append({"check": name, "measured": measured, "pass": bool(ok), "fatal": fatal})

    m, nm, rnd, o = res["momentum"], res["neg_momentum"], res["random"], res["drift_oracle"]
    check("constant factor rejected", f"valid={res['constant']['valid']}", res["constant"]["valid"] is False)
    check("random factor gold = 0", f"{rnd['gold']:+.4f} +/- {rnd['gold_se']:.4f}", abs(z(rnd)) <= Z_TOL)
    check("neg(momentum) = -momentum", f"{nm['gold']:+.6f} vs {-m['gold']:+.6f}", abs(nm["gold"] + m["gold"]) < 1e-9)
    check("QuantEvolver native RankIC = our proxy (momentum, same window)", f"{qe:+.6f} vs {ours:+.6f}",
          abs(qe - ours) < 1e-6, fatal=False)
    if args.kind == "drift":
        check("drift oracle gold > 0.5", f"{o['gold']:+.4f} +/- {o['gold_se']:.4f}", o["valid"] and o["gold"] > 0.5)
        check("momentum gold > 0.5 and significant", f"{m['gold']:+.4f} +/- {m['gold_se']:.4f}",
              m["gold"] > 0.5 and z(m) > 10)
        check("QuantEvolver native momentum RankIC > 0.5", f"{qe:+.4f}", qe > 0.5)
    else:
        check("drift oracle undefined (flat targets -> constant factor rejected)", f"valid={o['valid']}",
              o["valid"] is False)
        check("momentum gold (report only; bounce around a flat target)",
              f"{m['gold']:+.4f} +/- {m['gold_se']:.4f}", True, fatal=False)

    slim = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in res.items()}
    out = {"world_dir": str(root), "kind": args.kind, "n_gold": len(golds), "mom_expr": MOM_EXPR,
           "qe_native_momentum_rank_ic": qe, "factors": slim, "checks": checks,
           "all_pass": all(c["pass"] for c in checks),
           "fatal_pass": all(c["pass"] for c in checks if c["fatal"])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}]{'' if c['fatal'] else ' (non-fatal)'} {c['check']}: {c['measured']}")
    print(f"precheck {args.kind}: {'ALL PASS' if out['all_pass'] else 'FAILED'} -> {args.out}")
    sys.exit(0 if out["fatal_pass"] else 1)


if __name__ == "__main__":
    main()
