# TradingAgents

Synthetic markets with known ground truth for evaluating LLM "fair price" factor discovery, built on
[QuantEvolver](https://github.com/QuantLLM/QuantEvolver).

- `synthmarket/`: two simulators that write QuantEvolver-format 1-min OHLCV parquet plus `truth/` (fair value,
  mispricing).
  - `reference.py`: closed-form partial-adjustment model with analytic RankIC for known factors.
  - `engine.py`, `bots.py`, `world.py`, ...: endogenous prices on the UWTC exchange; informed traders trade toward a
    hidden target F, market makers learn only from the tape.
- `synth_eval/evaluator.py`: `SynthEvaluator(expr, start, end)` returns proxy (train RankIC), gold (held-out seeds),
  standard errors and correlation with true mispricing.
- `third_party/QuantEvolver/`: vendored QuantEvolver plus a `reward_impl: ground_truth` hook; see
  `third_party/PATCHES.md` for every change and the bugs fixed.
- `runs/sanity/`: latest sanity report (hard-coded factors of known performance). Other `runs/` output is not
  committed.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -e "third_party/QuantEvolver[backtest,dev,rft]" -e . -e "<path-to>/UWTC_Backend[dev]"
```

## Reproduce

```powershell
foreach ($w in "ref_slow","ref_fast","ref_noisy","ref_placebo") {
  .\.venv\Scripts\python scripts\gen_ref_world.py --world configs\worlds\$w.yaml --out runs\worlds\$w --gold-seeds 64 }
foreach ($w in "uwtc_informed","uwtc_placebo") {
  .\.venv\Scripts\python scripts\gen_uwtc_splits.py --world configs\worlds\$w.yaml --out runs\worlds\$w --gold-seeds 64 }
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts\sanity.py --uwtc
```
