"""Step (d): sanity-check the ground-truth evaluator with hard-coded factors of known performance.

Writes ``runs/sanity/results.csv`` and ``runs/sanity/report.md``. Every check is computed here; failures are
reported as failures.

    python scripts/sanity.py                       # reference worlds (closed-form expectations)
    python scripts/sanity.py --uwtc                # also the endogenous UWTC worlds, if generated
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth_eval.evaluator import Split, SynthEvaluator  # noqa: E402
from synthmarket.reference import RefWorld, expected_spearman, momentum_pearson, oracle_pearson  # noqa: E402

N_SYMBOLS_REF = 16
HORIZON_BARS, BAR_MIN = 6, 5
H_MIN = HORIZON_BARS * BAR_MIN
MOM_N = 13
MOM_EXPR = f"div(last(close({MOM_N})), first(close({MOM_N})))"
MOM_L_MIN = (MOM_N - 1) * BAR_MIN
Z_TOL = 3.0
TRUTH_FACTORS = ("oracle", "neg_oracle")


def oracle(sp: Split) -> pd.DataFrame:
    return sp.mispricing


def neg_oracle(sp: Split) -> pd.DataFrame:
    return -sp.mispricing


def random_factor(sp: Split) -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32(str(sp.path).encode()))
    return pd.DataFrame(rng.standard_normal(sp.close.shape), index=sp.close.index, columns=sp.close.columns)


def constant(sp: Split) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=sp.close.index, columns=sp.close.columns)


@dataclass
class WorldSpec:
    name: str
    root: Path
    ref: RefWorld | None = None

    def dirs(self) -> tuple[Path, list[Path]]:
        return self.root / "train", sorted((self.root / "gold").glob("seed_*"))


def check_row(world: str, check: str, expected: str, measured: str, passed: bool, **extra) -> dict:
    return {"world": world, "check": check, "expected": expected, "measured": measured,
            "pass": bool(passed), **extra}


def run_world(spec: WorldSpec) -> tuple[list[dict], list[dict]]:
    train, golds = spec.dirs()
    ev = SynthEvaluator(train, golds, bar_minutes=BAR_MIN, horizon_bars=HORIZON_BARS, warmup_days=1.0)
    rows, factors = [], []
    w = spec.name

    def record(label, out):
        factors.append({"world": w, "factor": label, **{k: v for k, v in out.items() if not isinstance(v, list)}})
        return out

    o = record("oracle", ev.evaluate_panels(oracle))
    no = record("neg_oracle", ev.evaluate_panels(neg_oracle))
    rnd = record("random", ev.evaluate_panels(random_factor))
    const = record("constant", ev.evaluate_panels(constant))
    mom = ev.evaluate_panels(lambda sp: ev.dsl_factor(MOM_EXPR, sp))
    record(f"dsl:{MOM_EXPR}", mom)
    rev = ev.evaluate_panels(lambda sp: ev.dsl_factor(f"neg({MOM_EXPR})", sp))
    record(f"dsl:neg({MOM_EXPR})", rev)
    bad = ev("unknown_op(close(60))")

    def z(out, target):
        return (out["gold"] - target) / out["gold_se"]

    if spec.ref is not None:
        exp_o = expected_spearman(oracle_pearson(spec.ref, H_MIN), N_SYMBOLS_REF)
        exp_m = expected_spearman(momentum_pearson(spec.ref, MOM_L_MIN, H_MIN), N_SYMBOLS_REF)
        rows.append(check_row(w, "oracle gold = analytic E[RankIC]", f"{exp_o:+.4f}",
                              f"{o['gold']:+.4f} +/- {o['gold_se']:.4f} (z={z(o, exp_o):+.2f})",
                              abs(z(o, exp_o)) <= Z_TOL))
        rows.append(check_row(w, f"momentum {MOM_EXPR} gold = analytic", f"{exp_m:+.4f}",
                              f"{mom['gold']:+.4f} +/- {mom['gold_se']:.4f} (z={z(mom, exp_m):+.2f})",
                              abs(z(mom, exp_m)) <= Z_TOL))
    else:
        rows.append(check_row(w, "oracle gold > 0 (informed) / = 0 (placebo)",
                              "> 0" if "placebo" not in w else "0 within 3 se",
                              f"{o['gold']:+.4f} +/- {o['gold_se']:.4f}",
                              (o["gold"] > Z_TOL * o["gold_se"]) if "placebo" not in w
                              else abs(o["gold"]) <= Z_TOL * o["gold_se"]))
        rows.append(check_row(w, f"momentum {MOM_EXPR} gold (no closed form)", "report only",
                              f"{mom['gold']:+.4f} +/- {mom['gold_se']:.4f}", True))
    rows.append(check_row(w, "oracle mispricing_corr = 1", "1.0000", f"{o['mispricing_corr']:.6f}",
                          o["mispricing_corr"] > 0.9999))
    rows.append(check_row(w, "neg_oracle = -oracle (gold and proxy)", f"{-o['gold']:+.6f} / {-o['proxy']:+.6f}",
                          f"{no['gold']:+.6f} / {no['proxy']:+.6f}",
                          abs(no["gold"] + o["gold"]) < 1e-9 and abs(no["proxy"] + o["proxy"]) < 1e-9))
    rows.append(check_row(w, "neg(momentum) = -momentum (DSL path)", f"{-mom['gold']:+.6f}", f"{rev['gold']:+.6f}",
                          abs(rev["gold"] + mom["gold"]) < 1e-9))
    rows.append(check_row(w, "random factor gold = 0", "0 within 3 se",
                          f"{rnd['gold']:+.4f} +/- {rnd['gold_se']:.4f} (z={z(rnd, 0.0):+.2f})",
                          abs(z(rnd, 0.0)) <= Z_TOL))
    rows.append(check_row(w, "constant factor rejected", "valid=False", f"valid={const['valid']}",
                          const["valid"] is False))
    rows.append(check_row(w, "invalid DSL op rejected", "None", repr(bad), bad is None))
    mom_fn = lambda sp: ev.dsl_factor(MOM_EXPR, sp)  # noqa: E731
    for label, out, fn in (("oracle", o, oracle), ("momentum", mom, mom_fn), ("random", rnd, random_factor)):
        emp = float(np.std(out["gold_seed_rank_ic"], ddof=1))
        boot = float(np.mean(out["gold_seed_se"]))
        half = half_split_sd(ev, fn)
        rows.append(check_row(w, f"block-bootstrap se calibrated to within-seed time sampling ({label})",
                              "ratio in [0.67, 1.5]",
                              f"mean boot se {boot:.4f} vs half-split sd {half:.4f} (ratio {boot / half:.2f}); "
                              f"seed-to-seed sd {emp:.4f} (ratio {boot / emp:.2f})",
                              0.67 <= boot / half <= 1.5))
    return rows, factors


def half_split_sd(ev: SynthEvaluator, factor_fn) -> float:
    """Time-sampling noise of one seed's mean RankIC, estimated without the bootstrap: across gold seeds,
    sd(IC_first_half - IC_second_half) / 2. Unlike the seed-to-seed sd, this excludes variation from
    per-seed world parameters (e.g. drawn per-symbol vols), which a within-seed bootstrap cannot see."""
    diffs = []
    for g in ev.gold:
        s, e = ev.default_window(g)
        mid = s + (e - s) / 2
        fac = factor_fn(g)
        a, b = ev.score_split(fac, g, s, mid), ev.score_split(fac, g, mid, e)
        if a is not None and b is not None:
            diffs.append(a["rank_ic"] - b["rank_ic"])
    return float(np.std(diffs, ddof=1) / 2.0)


def proxy_gold_rows(world: str, factors: list[dict]) -> list[dict]:
    valid = [f for f in factors if f.get("valid") and f["factor"] != "constant"]
    sig = [f for f in valid if abs(f["gold"]) > Z_TOL * f["gold_se"]]
    agree = sum(np.sign(f["proxy"]) == np.sign(f["gold"]) for f in sig)
    order_p = pd.Series([f["proxy"] for f in valid]).rank().to_numpy()
    order_g = pd.Series([f["gold"] for f in valid]).rank().to_numpy()
    rho = float(np.corrcoef(order_p, order_g)[0, 1]) if len(valid) > 2 else float("nan")
    gap = float(np.mean([abs(f["proxy"] - f["gold"]) for f in valid]))
    rows = [check_row(world, "proxy sign = gold sign (factors with significant gold)", f"{len(sig)}/{len(sig)}",
                      f"{agree}/{len(sig)}", agree == len(sig))]
    if "placebo" in world:
        truth_sig = [f["factor"] for f in sig if f["factor"] in TRUTH_FACTORS]
        price_sig = [f"{f['factor']} {f['gold']:+.4f}" for f in sig if f["factor"] not in TRUTH_FACTORS]
        rows.append(check_row(world, "no truth-derived factor (oracle, neg_oracle) has significant gold (placebo)",
                              "0 significant", f"{len(truth_sig)} significant", len(truth_sig) == 0))
        rows.append(check_row(world, "price-only factors with significant gold (placebo; microstructure, not F)",
                              "report only", "; ".join(price_sig) or "none", True))
    elif len(sig) >= 3:
        rows.append(check_row(world, "proxy vs gold rank agreement (Spearman over factor set)", "> 0.8",
                              f"{rho:.3f}", rho > 0.8))
    rows.append(check_row(world, "mean |proxy - gold| (single train split vs held-out seeds)", "report only",
                          f"{gap:.4f}", True))
    return rows


def end_to_end_reward() -> list[dict]:
    """Fake verl batch through build_reward_fn(ground_truth) built from the RFT YAML."""
    from quant_evolver.rft.config import RFTConfig
    from quant_evolver.rft.verl_main import build_reward_fn, build_verl_config
    from quant_evolver.utils.config import load_config

    from synth_eval.fake_batch import CharTokenizer, make_batch

    import os
    os.chdir(ROOT)
    rft = RFTConfig.from_dict(load_config(ROOT / "configs" / "rft_ground_truth_ref_slow.yaml")["rft"])
    cfg = build_verl_config(rft)
    train = pd.read_parquet(cfg.data.train_files[0])
    tasks = [train.iloc[i]["extra_info"] for i in range(3)]
    tok = CharTokenizer()
    reward_fn = build_reward_fn(rft, tok)
    responses = [f"Reasoning... <expr>{MOM_EXPR}</expr>", "<expr>unknown_op(close(60))</expr>", "no tags at all"]
    batch, lens = make_batch(responses, tasks, tokenizer=tok)
    out = reward_fn(batch, return_dict=True)
    R, extra = out["reward_tensor"], out["reward_extra_info"]
    direct = reward_fn.ev(MOM_EXPR, tasks[0]["start_date"], tasks[0]["end_date"])
    rows = []
    nz_ok = all(R[i].nonzero().flatten().tolist() == [lens[i] - 1] for i in range(len(responses)))
    rows.append(check_row("e2e", "reward placed only on last response token", "True", str(nz_ok), nz_ok))
    rows.append(check_row("e2e", "reward = evaluator proxy for the row's task window",
                          f"{direct['proxy']:+.6f} (window {tasks[0]['start_date']}..{tasks[0]['end_date']})",
                          f"{float(R[0].sum()):+.6f}", abs(float(R[0].sum()) - direct["proxy"]) < 1e-6))
    rows.append(check_row("e2e", "invalid / missing expr -> -1", "-1.0, -1.0",
                          f"{float(R[1].sum()):+.1f}, {float(R[2].sum()):+.1f}",
                          float(R[1].sum()) == -1.0 and float(R[2].sum()) == -1.0))
    keys_ok = all(len(extra[k]) == len(responses) for k in ("gold", "proxy_se", "proxy", "mispricing_corr"))
    rows.append(check_row("e2e", "reward_extra_info logs gold/proxy_se per row", "lists of len 3",
                          f"gold={[round(x, 4) if x == x else x for x in extra['gold']]}, "
                          f"proxy_se={[round(x, 4) for x in extra['proxy_se']]}", keys_ok))
    rows.append(check_row("e2e", "prompt rows carry task windows", "start/end in extra_info",
                          f"{sorted({(t['start_date'], t['end_date']) for t in tasks})}",
                          all("start_date" in t and "end_date" in t for t in tasks)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-root", default=str(ROOT / "runs" / "worlds"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "sanity"))
    ap.add_argument("--uwtc", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    root = Path(args.worlds_root)
    specs = [WorldSpec(n, root / n, RefWorld.from_yaml(ROOT / "configs" / "worlds" / f"{n}.yaml"))
             for n in ("ref_slow", "ref_fast", "ref_noisy", "ref_placebo")]
    if args.uwtc:
        specs += [WorldSpec(n, root / n) for n in ("uwtc_informed", "uwtc_placebo") if (root / n / "train").exists()]
    rows, factors = [], []
    for spec in specs:
        r, f = run_world(spec)
        rows += r + proxy_gold_rows(spec.name, f)
        factors += f
        print(f"[{spec.name}] {sum(x['pass'] for x in r)}/{len(r)} checks passed", flush=True)
    if not args.skip_e2e:
        rows += end_to_end_reward()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    pd.DataFrame(factors).to_csv(out / "factors.csv", index=False)
    n_pass = sum(r["pass"] for r in rows)
    lines = ["# Sanity report", "", f"{n_pass}/{len(rows)} checks passed.", "",
             f"Horizon {H_MIN} min ({HORIZON_BARS} x {BAR_MIN}-min bars); tolerance |z| <= {Z_TOL} using the "
             "across-seed standard error of gold.", "",
             "Check-specification revisions (each made after a failure; original numbers are still shown):", "",
             "- Placebo worlds: proxy-vs-gold rank agreement is not tested (all gold ~0, so it compares noise "
             "with noise). The placebo claim is that F carries no information, so the test is that truth-derived "
             "factors have no significant gold; price-only signal (e.g. microstructure reversal) is reported.",
             "- Bootstrap calibration is tested against within-seed time-sampling noise (half-split sd), which is "
             "what a within-seed bootstrap estimates. Seed-to-seed sd is also shown: in UWTC worlds it is larger "
             "because per-seed parameters (per-symbol vol/drift) vary between seeds.", "",
             "| world | check | expected | measured | pass |", "|---|---|---|---|---|"]
    lines += [f"| {r['world']} | {r['check']} | {r['expected']} | {r['measured']} | {'PASS' if r['pass'] else 'FAIL'} |"
              for r in rows]
    lines += ["", "## Factor scores", "", "| world | factor | proxy | se | gold | gold_se | mispricing_corr |",
              "|---|---|---|---|---|---|---|"]
    for f in factors:
        if f.get("valid"):
            lines.append(f"| {f['world']} | `{f['factor']}` | {f['proxy']:+.4f} | {f['se']:.4f} | {f['gold']:+.4f} | "
                         f"{f['gold_se']:.4f} | {f['mispricing_corr']:+.4f} |")
        else:
            lines.append(f"| {f['world']} | `{f['factor']}` | invalid | | | | |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{n_pass}/{len(rows)} checks passed -> {out / 'report.md'}")
    for r in rows:
        if not r["pass"]:
            print("FAIL:", json.dumps(r))


if __name__ == "__main__":
    main()
