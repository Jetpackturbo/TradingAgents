# third_party patches

## QuantEvolver

- Source: https://github.com/QuantLLM/QuantEvolver at commit `4eb0e78842138ada5334349585b114ad923564e8`
  (2026-05-14), downloaded as a zip archive (no git on this machine's PATH).
- Installed with `pip install -e "third_party/QuantEvolver[backtest,dev]"` and `[rft]` (torch CPU wheel from
  the PyTorch CPU index), plus `verl==0.4.1`.
- Source changes needed to build/import as-is: **none**.

### Environment notes (Windows)

- Windows Application Control blocks each newly installed native DLL on its first load
  (`ImportError: DLL load failed ... An Application Control policy has blocked this file`).
  Re-running the same import succeeds. Expect one or two such failures after any fresh `pip install`.
- verl/vLLM training needs Linux + GPU; on this machine verl only imports (no rollout engine).

### Data assumption (not a patch)

- `evaluation/backtrader_signal.py` sets `warmup_bars = warmup_days * (1440 // bar_minutes)`, i.e. it
  assumes a 24/7 minute calendar. Session-only (e.g. 09:30-16:00) data never leaves warmup and every seed
  fails with `No valid factor scores collected. bars=0`. All data we generate uses a continuous minute calendar.

### Bug fixes applied (`quant_evolver/rft/reward_bridge.py`)

Found by `scripts/qe_bridge_smoke.py` (fake verl batch through the unmodified bridge):

1. **Cross-sectional reward always -1 in the default process executor.** `QuantRewardBridge.__call__`
   submits `_score_cross_sectional_expr_worker` with 9 positional args (an extra `free_code_eval_timeout_s`)
   but the worker accepted 8. The `TypeError` was swallowed and every rollout scored -1.0, so GRPO would see
   zero within-group variance in `reward_kind: cross_sectional_rankic` DSL mode.
   Fix: the worker takes an ignored `_timeout_s=0` positional arg.
2. **Cross-sectional path ignored `data_dir` / `data_file_template`.** The worker built `CrossSectionalConfig`
   without `data=`, so it always read `./data/{symbol_safe}_1m.parquet` relative to the CWD.
   Fix: the worker takes `data_dir`, `data_file_template` and both call sites pass `self.data_dir`,
   `self.data_file_template` (DSL mode only; the free-code worker signature is unchanged).

Verified: bridge cross-sectional reward for `div(ts_mean(returns(60)), ts_std(returns(60)))` = -0.0628, equal to
`quantevolver rft-eval-expr ... --reward-kind cross_sectional_rankic` (-0.06277) on the same data, including from a
CWD without `./data`. QuantEvolver's own tests still pass (3/3).

## Changes made for the ground-truth evaluation

All additive; the default `reward_impl: quant_evolver` path is unchanged.

- `quant_evolver/rft/gt_reward.py` (new): `GroundTruthReward`, a verl `reward_fn(data, return_dict)` with the
  same contract as `QuantRewardBridge`. It decodes each response (attention-mask aware), extracts the DSL
  expression with QuantEvolver's own `extract_expr`, and calls an external evaluator
  `ev(expr, start_date, end_date) -> dict | None`. The reward placed on the last response token is the evaluator's
  **proxy** (train-split RankIC); invalid expressions get `invalid_score` (-1). `reward_extra_info` logs
  `proxy, proxy_se, gold, gold_se, mispricing_corr, valid` per rollout so the proxy/gold gap can be tracked
  during training without ever being rewarded on gold. `load_evaluator("module:factory", kwargs)` builds it.
- `quant_evolver/rft/config.py`: three new `RFTConfig` fields, `reward_impl` (`quant_evolver` | `ground_truth`),
  `reward_evaluator` (default `synth_eval.evaluator:build_evaluator`), `reward_kwargs`.
- `quant_evolver/rft/verl_main.py`: the reward construction moved into `build_reward_fn(rft, tokenizer)`, which
  returns `GroundTruthReward` for `reward_impl: ground_truth`, raises on unknown values, and otherwise builds the
  original `QuantRewardBridge` with the original arguments.

Verified: `scripts/sanity.py` (end-to-end section) loads `configs/rft_ground_truth_ref_slow.yaml`, builds the
reward via `build_reward_fn`, and scores a fake verl batch; the last-token reward equals the evaluator proxy
exactly. `quantevolver rft-launch configs/rft_ground_truth_ref_slow.yaml --dry-run` writes the launch script and a
resolved config that keeps `reward_impl: ground_truth` and the evaluator kwargs.
