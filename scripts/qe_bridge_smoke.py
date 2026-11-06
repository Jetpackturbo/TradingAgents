"""Step (b): exercise QuantEvolver's unmodified RFT plumbing without a GPU.

1. ``build_verl_config`` on the example config (writes train/val prompt parquet, merges verl base config).
2. ``QuantRewardBridge`` on a fake verl batch, in direction and cross-sectional modes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synth_eval.fake_batch import CharTokenizer, make_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "qe_smoke"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    os.chdir(run_dir)

    from quant_evolver.rft.config import RFTConfig
    from quant_evolver.rft.reward_bridge import QuantRewardBridge
    from quant_evolver.rft.verl_main import build_verl_config
    from quant_evolver.utils.config import load_config

    raw = load_config("configs/example_rft_pure_verl.yaml")
    rft = RFTConfig.from_dict(raw["rft"])
    cfg = build_verl_config(rft)
    train = pd.read_parquet(cfg.data.train_files[0])
    print(f"[build_verl_config] adv_estimator={cfg.algorithm.adv_estimator} rollout.n={cfg.actor_rollout_ref.rollout.n} "
          f"train_rows={len(train)} columns={list(train.columns)}")
    print(f"[build_verl_config] first prompt (truncated): {str(train.iloc[0]['prompt'])[:400]!r}")
    print(f"[build_verl_config] first extra_info: {train.iloc[0]['extra_info']}")

    responses = [
        "<expr>div(ts_mean(returns(60)), ts_std(returns(60)))</expr>",
        "<expr>neg(zscore(last(close(120)), close(120)))</expr>",
        "<expr>unknown_op(close(60))</expr>",
        "no expression here",
    ]
    extra = [{"task_id": f"t{i}", "start_date": "2024-01-01", "end_date": "2024-02-01", "bar_minutes": 5}
             for i in range(len(responses))]
    tok = CharTokenizer()

    def run(label: str, **kw) -> None:
        bridge = QuantRewardBridge(tok, max_workers=2, **kw)
        batch, lens = make_batch(responses, extra, tokenizer=tok)
        out = bridge(batch, return_dict=True)
        R = out["reward_tensor"]
        for i, r in enumerate(responses):
            nz = R[i].nonzero().flatten().tolist()
            print(f"[{label}] reward={float(R[i].sum()):+.4f} nonzero_at={nz} last_token={lens[i] - 1} :: {r[:60]}")
        print(f"[{label}] reward_extra_info keys: {sorted(out['reward_extra_info'])}")

    run("direction", reward_kind="direction", horizon_bars=1, data_dir="data")
    run("xsec data_dir=data", reward_kind="cross_sectional_rankic", horizon_bars=6, data_dir="data")
    alt = run_dir / "data_alt_location"
    alt.mkdir(exist_ok=True)
    for p in (run_dir / "data").glob("*.parquet"):
        target = alt / p.name
        if not target.exists():
            target.write_bytes(p.read_bytes())
    os.chdir(alt)
    run("xsec data_dir=<abs path>, cwd without ./data", reward_kind="cross_sectional_rankic", horizon_bars=6,
        data_dir=str(alt))


if __name__ == "__main__":
    main()
