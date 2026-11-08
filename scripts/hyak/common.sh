# Shared settings for the Hyak (klone) jobs. Sourced by every script in scripts/hyak/.
# Override any of these by exporting it before calling sbatch / the submit script.

: "${HYAK_ROOT:=/gscratch/socialrl/$USER}"                  # big, fast storage (home quota is small)
: "${REPO_DIR:=$(pwd)}"                                      # this repository (callers cd here first)
: "${CONDA_DIR:=$HYAK_ROOT/miniforge3}"
: "${ENV_PREFIX:=$HYAK_ROOT/envs/trading-agents-rft}"
: "${MODEL_ID:=Qwen/Qwen3-1.7B}"
: "${MODEL_DIR:=$HYAK_ROOT/models/Qwen3-1.7B}"
: "${HF_HOME:=$HYAK_ROOT/hf_cache}"
export HYAK_ROOT REPO_DIR CONDA_DIR ENV_PREFIX MODEL_ID MODEL_DIR HF_HOME

activate_env() {
    # conda's activate scripts read unset variables, which `set -u` would turn into errors.
    set +u
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
    set -u
    export PYTHONPATH="$REPO_DIR/third_party/QuantEvolver:$REPO_DIR:${PYTHONPATH:-}"
}
