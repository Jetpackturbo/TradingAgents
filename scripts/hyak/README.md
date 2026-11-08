# Hyak: drift sanity check

Tests whether QuantEvolver's GRPO training detects the simplest alpha: a market made only of bots that
post never-cancelled resting limit orders around a target with a constant per-symbol drift.

## One-time setup (klone login node)

1. Put this repo on scratch at `$HYAK_ROOT/TradingAgents` with `HYAK_ROOT=/gscratch/socialrl/$USER` (e.g. `git
   clone` after pushing; defaults in `common.sh`). The UWTC exchange code it needs is included (`uwtc/`).
2. `cd $HYAK_ROOT/TradingAgents && bash scripts/hyak/submit_drift_sanity.sh --setup`
   builds the env and downloads `Qwen/Qwen3-1.7B` (1 L40, up to 2 h), then starts the sanity jobs once
   setup succeeds. Later runs: drop `--setup`.

## What runs

`drift_sanity.sbatch` is a 2-task array, each task on 4 L40s / 32 CPUs / 256 GB for up to 8 h:

| task | world | role |
|---|---|---|
| 0 | `uwtc_resting_drift` | the test |
| 1 | `uwtc_resting_placebo` | control: flat targets, no drift |

Each task generates its world (train + 64 gold seeds), runs `scripts/drift_precheck.py` (the drift task
stops here if the drift oracle, DSL momentum or QuantEvolver's native RankIC fail to detect the drift),
trains with `quant_evolver.rft.verl_main` (`reward_impl: ground_truth`, from-scratch `ocd` prompts, 60
steps x 32 prompts x 8 samples), and runs `scripts/drift_posteval.py`.

## Results

`runs/hyak/<world>_<jobid>/`: `precheck.json`, `rft_config.yaml`, `train.log`,
`pure_verl/experiments/<name>/rollout_log/<step>.jsonl`, and `posteval/report.md` with the verdict.
The drift task passes if late-training factors reach >= 0.5 x the drift oracle's held-out RankIC and rank
symbols by their true drift (drift_corr >= 0.5) with >= 80% valid expressions. The placebo is reported
only: around a flat target, price bounce makes short-term reversal a real (non-drift) alpha.
