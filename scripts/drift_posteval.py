"""Post-training evaluation for the resting-order drift sanity job.

Reads verl's per-step rollout dumps (``<rollout_dir>/<step>.jsonl``, with GroundTruthReward's ``expr`` /
``proxy`` / ``valid`` fields), then re-scores every distinct expression from the first and the last
``--window`` steps on ALL held-out gold seeds (training only logged a few):

* ``gold``: mean held-out RankIC against 30-minute forward returns,
* ``drift_corr``: cross-sectional rank correlation of the factor with each symbol's true drift.

Rollouts are weighted by how often the policy produced each expression. The drift world passes if the
late policy detects the drift: rollout-weighted gold >= 0.5 x the drift oracle's gold and rollout-weighted
drift_corr >= 0.5. Learning (late vs early proxy and gold) is reported with bootstrap CIs. Placebo runs are
reported only (their prices mean-revert around flat targets, so price factors can have real gold there).

    python scripts/drift_posteval.py --rollout-dir <run>/pure_verl/experiments/<name>/rollout_log \
        --world-dir runs/worlds/uwtc_resting_drift --kind drift --precheck <run>/precheck.json --out <run>/posteval
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth_eval.drift import drift_panel  # noqa: E402
from synth_eval.evaluator import SynthEvaluator, rowwise_rank_ic  # noqa: E402

_EV: SynthEvaluator | None = None
_DRIFT: list | None = None


def _score(expr: str) -> dict | None:
    out = _EV(expr)
    if out is None:
        return None
    dc = []
    for g, d in zip(_EV.gold, _DRIFT):
        if d is None:
            continue
        s, e = _EV.default_window(g)
        fac = _EV.dsl_factor(expr, g)
        ic = rowwise_rank_ic(fac[(fac.index >= s) & (fac.index <= e)], d[(d.index >= s) & (d.index <= e)],
                             _EV.min_symbols_per_time)
        if len(ic):
            dc.append(float(ic.mean()))
    return {"proxy": out["proxy"], "gold": out["gold"], "gold_se": out["gold_se"],
            "drift_corr": float(np.mean(dc)) if dc else float("nan")}


def load_rollouts(rollout_dir: Path) -> dict[int, list[dict]]:
    steps = {}
    for f in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        steps[int(f.stem)] = rows
    return steps


def weighted(rows: list[dict], scores: dict, key: str) -> np.ndarray:
    """Per-rollout values (invalid rollouts excluded) for bootstrap."""
    vals = [scores[r["expr"]][key] for r in rows if r.get("expr") and scores.get(r["expr"]) is not None]
    return np.array([v for v in vals if np.isfinite(v)], dtype=float)


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    global _EV, _DRIFT
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--world-dir", required=True)
    ap.add_argument("--kind", choices=["drift", "placebo"], required=True)
    ap.add_argument("--precheck", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=5, help="steps in the early and late windows")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    steps = load_rollouts(Path(args.rollout_dir))
    if not steps:
        raise SystemExit(f"no rollout dumps under {args.rollout_dir}")
    order = sorted(steps)
    early_steps, late_steps = order[: args.window], order[-args.window:]
    early = [r for s in early_steps for r in steps[s]]
    late = [r for s in late_steps for r in steps[s]]

    root = Path(args.world_dir)
    _EV = SynthEvaluator(root / "train", sorted((root / "gold").glob("seed_*")), n_boot=200)
    _DRIFT = [drift_panel(g) for g in _EV.gold] if args.kind == "drift" else [None] * len(_EV.gold)
    exprs = sorted({r["expr"] for r in early + late if r.get("expr")})
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    if ctx is not None and args.workers > 1:
        with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
            scores = dict(zip(exprs, pool.map(_score, exprs, chunksize=4)))
    else:
        scores = {e: _score(e) for e in exprs}

    pre = json.loads(Path(args.precheck).read_text())
    oracle_gold = pre["factors"]["drift_oracle"].get("gold", float("nan"))
    per_step = [{"step": s, "n": len(steps[s]),
                 "valid_frac": float(np.mean([float(r.get("valid", 0)) for r in steps[s]])),
                 "mean_score": float(np.mean([r["score"] for r in steps[s]]))} for s in order]

    summary = {}
    for name, rows in (("early", early), ("late", late)):
        g, p, d = weighted(rows, scores, "gold"), weighted(rows, scores, "proxy"), weighted(rows, scores, "drift_corr")
        top = Counter(r["expr"] for r in rows if r.get("expr") and scores.get(r["expr"]) is not None).most_common(5)
        summary[name] = {
            "steps": early_steps if name == "early" else late_steps,
            "rollouts": len(rows),
            "valid_frac": float(np.mean([float(r.get("valid", 0)) for r in rows])),
            "gold_mean": float(g.mean()) if len(g) else float("nan"), "gold_ci95": boot_ci(g),
            "proxy_mean": float(p.mean()) if len(p) else float("nan"), "proxy_ci95": boot_ci(p),
            "drift_corr_mean": float(d.mean()) if len(d) else float("nan"), "drift_corr_ci95": boot_ci(d),
            "distinct_valid_exprs": len({r["expr"] for r in rows if r.get("expr") and scores.get(r["expr"])}),
            "top_exprs": [{"expr": e, "count": c, **scores[e]} for e, c in top],
        }
    gl, ge = weighted(late, scores, "gold"), weighted(early, scores, "gold")
    pl, pe = weighted(late, scores, "proxy"), weighted(early, scores, "proxy")
    learning = {"gold_late_minus_early_ci95": boot_diff_ci(gl, ge), "proxy_late_minus_early_ci95": boot_diff_ci(pl, pe)}

    checks = []
    L = summary["late"]
    if args.kind == "drift":
        checks.append({"check": "late policy gold >= 0.5 x drift-oracle gold",
                       "measured": f"{L['gold_mean']:+.4f} vs oracle {oracle_gold:+.4f}",
                       "pass": bool(L["gold_mean"] >= 0.5 * oracle_gold)})
        checks.append({"check": "late policy factors rank symbols by true drift (drift_corr >= 0.5)",
                       "measured": f"{L['drift_corr_mean']:+.4f} CI {L['drift_corr_ci95']}",
                       "pass": bool(L["drift_corr_mean"] >= 0.5)})
        checks.append({"check": "late valid-expression rate >= 0.8", "measured": f"{L['valid_frac']:.3f}",
                       "pass": bool(L["valid_frac"] >= 0.8)})
    report = {"kind": args.kind, "world_dir": str(root), "n_gold": len(_EV.gold), "oracle_gold": oracle_gold,
              "summary": summary, "learning": learning, "per_step": per_step, "checks": checks,
              "verdict": ("PASS" if all(c["pass"] for c in checks) else "FAIL") if checks else "REPORT_ONLY"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "posteval.json").write_text(json.dumps(report, indent=2))

    lines = [f"# Drift sanity post-eval ({args.kind}): {report['verdict']}", "",
             f"World `{root}`; {len(_EV.gold)} held-out seeds; drift-oracle gold {oracle_gold:+.4f}.", "",
             "| window | steps | rollouts | valid | proxy | gold (95% CI) | drift_corr (95% CI) | distinct exprs |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("early", "late"):
        S = summary[name]
        lines.append(f"| {name} | {S['steps'][0]}-{S['steps'][-1]} | {S['rollouts']} | {S['valid_frac']:.2f} | "
                     f"{S['proxy_mean']:+.4f} | {S['gold_mean']:+.4f} [{S['gold_ci95'][0]:+.4f}, "
                     f"{S['gold_ci95'][1]:+.4f}] | {S['drift_corr_mean']:+.4f} [{S['drift_corr_ci95'][0]:+.4f}, "
                     f"{S['drift_corr_ci95'][1]:+.4f}] | {S['distinct_valid_exprs']} |")
    lines += ["", f"Late minus early: gold CI {learning['gold_late_minus_early_ci95']}, "
              f"proxy CI {learning['proxy_late_minus_early_ci95']}", ""]
    if checks:
        lines += ["| check | measured | pass |", "|---|---|---|"]
        lines += [f"| {c['check']} | {c['measured']} | {'PASS' if c['pass'] else 'FAIL'} |" for c in checks]
    lines += ["", "Most frequent late expressions:", ""]
    lines += [f"- `{t['expr']}` x{t['count']}: gold {t['gold']:+.4f}, drift_corr {t['drift_corr']:+.4f}"
              for t in L["top_exprs"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
