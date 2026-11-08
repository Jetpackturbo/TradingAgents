#!/bin/bash
# Submit the drift sanity jobs from the repo root on a klone login node.
#
#   bash scripts/hyak/submit_drift_sanity.sh                 # drift + placebo (2 x 4 L40s)
#   bash scripts/hyak/submit_drift_sanity.sh --only-drift    # drift world only (1 x 4 L40s)
#   bash scripts/hyak/submit_drift_sanity.sh --setup         # first run: build the env, then train
#
# STEPS=<n> (default 60) sets the number of GRPO steps.

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

array="0-1"
setup=0
for arg in "$@"; do
    case "$arg" in
        --only-drift) array="0" ;;
        --setup) setup=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

dep=()
if [ "$setup" = 1 ]; then
    sid=$(sbatch --parsable scripts/hyak/setup_env.sbatch)
    echo "setup job $sid"
    dep=(--dependency="afterok:$sid")
fi
jid=$(sbatch --parsable ${dep[@]+"${dep[@]}"} --array="$array" --export=ALL,STEPS="${STEPS:-60}" scripts/hyak/drift_sanity.sbatch)
echo "drift sanity array job $jid (tasks $array); logs in logs/qe-drift-sanity_${jid}_*.out"
echo "watch: squeue -u $USER"
