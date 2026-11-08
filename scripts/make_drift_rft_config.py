"""Write the task bank and QuantEvolver RFT config for one resting-order drift sanity run.

Training is from scratch (QuantEvolver ``mode: ocd``): prompts carry no seed expression, so a
positive result cannot come from copying a momentum seed. Rewards are the ground-truth evaluator's
proxy (train-split RankIC over a one-day task window); gold on held-out seeds is only logged.

    python scripts/make_drift_rft_config.py --world-dir runs/worlds/uwtc_resting_drift \
        --run-dir runs/hyak/drift_123 --model-path /gscratch/.../Qwen3-1.7B --n-gpus 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def task_bank(train_dir: Path, scenario: str) -> dict:
    """One task per full day after the 1-day DSL warmup."""
    idx = pd.read_parquet(sorted((train_dir / "data").glob("*_1m.parquet"))[0]).index
    days = pd.date_range(pd.Timestamp(idx[0]).normalize() + pd.Timedelta(days=1), pd.Timestamp(idx[-1]).normalize(),
                         freq="D")
    tasks = []
    for i, d in enumerate(days):
        s, e = str(d.date()), str((d + pd.Timedelta(days=1)).date())
        tasks.append({"id": f"no_seed__day_{i + 1}", "seed_id": "", "seed_name": "", "family": "no_seed",
                      "seed_expr": "", "time_split": f"day_{i + 1}", "start_date": s, "end_date": e,
                      "bar_minutes": 5, "scenario": scenario, "mutation_hints": []})
    return {"metadata": {"num_tasks": len(tasks)}, "tasks": tasks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--train-batch-size", type=int, default=32, help="prompts per step")
    ap.add_argument("--rollout-n", type=int, default=8, help="samples per prompt (GRPO group size)")
    ap.add_argument("--train-gold-seeds", type=int, default=8, help="gold seeds logged during training")
    ap.add_argument("--reward-workers", type=int, default=16)
    ap.add_argument("--ray-cpus", type=int, default=None)
    ap.add_argument("--lr", default="1e-6")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = Path(args.world_dir).resolve()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    golds = sorted((world / "gold").glob("seed_*"))[: args.train_gold_seeds]
    if not golds:
        raise SystemExit(f"no gold seeds under {world / 'gold'}")

    bank_path = run / "task_bank.yaml"
    bank_path.write_text(yaml.safe_dump(task_bank(world / "train", "cross-sectional 30-minute return ranking on "
                                                  "16 synthetic symbols"), sort_keys=False))

    overrides = [f"data.seed={args.seed}"]
    if args.ray_cpus:
        overrides.append(f"ray_init.num_cpus={args.ray_cpus}")
    rft = {
        "backend": "pure_verl",
        "mode": "ocd",
        "reward_kind": "cross_sectional_rankic",
        "bar_minutes": 5,
        "horizon_bars": 6,
        "data_dir": str(world / "train" / "data"),
        "reward_impl": "ground_truth",
        "reward_evaluator": "synth_eval.evaluator:build_evaluator",
        "reward_kwargs": {"train_dir": str(world / "train"), "gold_dirs": [str(g) for g in golds],
                          "bar_minutes": 5, "horizon_bars": 6, "warmup_days": 1.0, "n_boot": 200},
        "reward_workers": args.reward_workers,
        "experiment_name": run.name,
        "project_name": "trading_agents_drift_sanity",
        "output_dir": str(run / "generated_factors"),
        "run_root": str(run),
        "task_bank_path": str(bank_path),
        "task_sampler_seed": args.seed,
        "model_path": args.model_path,
        "n_gpus": args.n_gpus,
        "total_epochs": 1,
        "train_batch_size": args.train_batch_size,
        "rollout_n": args.rollout_n,
        "train_prompt_rows": args.train_batch_size * args.steps,
        "val_prompt_rows": args.train_batch_size,
        "val_n": 1,
        "save_freq": -1,
        "test_freq": -1,
        "max_prompt_length": 3000,
        "max_response_length": 256,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.6,
        "temperature": 1.0,
        "lr": args.lr,
        "relative_reward": False,
        "diversity_reward": False,
        "residual_response_reward": False,
        "xsec_behavior_diversity": False,
        "extra_overrides": overrides,
    }
    cfg_path = run / "rft_config.yaml"
    cfg_path.write_text(yaml.safe_dump({"rft": rft}, sort_keys=False))
    print(cfg_path)


if __name__ == "__main__":
    main()
