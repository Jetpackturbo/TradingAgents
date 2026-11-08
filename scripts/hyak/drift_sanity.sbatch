#!/bin/bash
# Sanity check: can QuantEvolver (GRPO, from-scratch DSL generation) detect the SIMPLEST alpha?
#
# World: only RestingDriftTraders, posting never-cancelled limit orders around a per-symbol target
# that drifts at a constant rate (configs/worlds/uwtc_resting_drift.yaml). Array task 1 is the
# placebo control with flat targets (uwtc_resting_placebo.yaml).
#
# Each array task, on one node with 4 L40s:
#   1. generate the world (train seed + 64 held-out gold seeds) if not already on disk
#   2. CPU precheck: drift oracle / DSL momentum / QuantEvolver-native RankIC detect the drift
#      (drift world aborts here, before any GPU work, if they do not)
#   3. GRPO training: QuantEvolver verl_main with the ground-truth reward (train-split RankIC)
#   4. post-eval: re-score early vs late rollouts on all 64 gold seeds + correlation with true drift
#
#   cd $REPO_DIR && bash scripts/hyak/submit_drift_sanity.sh          # both array tasks
#
#SBATCH --job-name=qe-drift-sanity
#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hyak/common.sh
activate_env

WORLDS=(uwtc_resting_drift uwtc_resting_placebo)
KINDS=(drift placebo)
i="${SLURM_ARRAY_TASK_ID:-0}"
WORLD="${WORLDS[$i]}"
KIND="${KINDS[$i]}"
STEPS="${STEPS:-60}"
NGPU=4
CPUS="${SLURM_CPUS_PER_TASK:-32}"
WORLD_DIR="$REPO_DIR/runs/worlds/$WORLD"
RUN_NAME="${WORLD}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
RUN_DIR="$REPO_DIR/runs/hyak/$RUN_NAME"
mkdir -p "$RUN_DIR"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'ray stop --force >/dev/null 2>&1 || true; rm -rf "$RAY_TMPDIR"' EXIT

echo "== $(date -Is) task=$i world=$WORLD kind=$KIND run=$RUN_DIR host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "== [1/4] world"
if [ ! -f "$WORLD_DIR/gold/seed_1063/summary.json" ]; then
    python scripts/gen_uwtc_splits.py --world "configs/worlds/$WORLD.yaml" --out "$WORLD_DIR" \
        --gold-seeds 64 --workers 16
fi

echo "== [2/4] precheck"
python scripts/drift_precheck.py --world-dir "$WORLD_DIR" --kind "$KIND" --out "$RUN_DIR/precheck.json"

echo "== [3/4] GRPO training ($STEPS steps)"
python scripts/make_drift_rft_config.py --world-dir "$WORLD_DIR" --run-dir "$RUN_DIR" --model-path "$MODEL_DIR" \
    --n-gpus "$NGPU" --steps "$STEPS" --reward-workers 16 --ray-cpus $((CPUS - 16))
python -m quant_evolver.rft.verl_main --config "$RUN_DIR/rft_config.yaml" 2>&1 | tee "$RUN_DIR/train.log"

echo "== [4/4] post-eval"
python scripts/drift_posteval.py --rollout-dir "$RUN_DIR/pure_verl/experiments/$RUN_NAME/rollout_log" \
    --world-dir "$WORLD_DIR" --kind "$KIND" --precheck "$RUN_DIR/precheck.json" --out "$RUN_DIR/posteval" \
    --workers 16
echo "== $(date -Is) done: $RUN_DIR/posteval/report.md"
