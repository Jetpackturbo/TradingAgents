#!/bin/bash
# One-time environment setup on a socialrl L40 node: Miniforge, a Python 3.11 env with
# torch 2.6 (CUDA 12.4) + vLLM 0.8.5 + verl 0.4.1 + flash-attn, this repo (including its uwtc package),
# QuantEvolver, and the base model. Ends with a GPU smoke test. Safe to re-run (skips finished steps).
#
#   cd $REPO_DIR && sbatch scripts/hyak/setup_env.sbatch
#
#SBATCH --job-name=qe-setup
#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hyak/common.sh

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    installer="$HYAK_ROOT/Miniforge3-Linux-x86_64.sh"
    curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$installer" -b -p "$CONDA_DIR"
    rm -f "$installer"
fi
set +u
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" python=3.11
conda activate "$ENV_PREFIX"
set -u

export PIP_CACHE_DIR="$HYAK_ROOT/pip_cache"
constraints="$(mktemp)"
# setuptools>=81 dropped pkg_resources, which parts of this stack still import.
# vllm 0.8.5 / verl 0.4.1 do not cap transformers, but they predate transformers 5.x.
printf 'torch==2.6.0\nvllm==0.8.5\nverl==0.4.1\nsetuptools<81\ntransformers==4.51.3\n' > "$constraints"
python -m pip install --upgrade pip
python -m pip install "setuptools<81"
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c "$constraints" vllm==0.8.5
python -m pip install -c "$constraints" verl==0.4.1
python -m pip install -c "$constraints" transformers==4.51.3 pytest
# Prebuilt wheel: compute nodes have no nvcc, so building flash-attn from source fails.
if ! python -c "import flash_attn" 2>/dev/null; then
    abi="$(python -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')"
    python -m pip install -c "$constraints" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"
fi
python -m pip install -c "$constraints" -e "third_party/QuantEvolver[backtest,rft]" -e .
rm -f "$constraints"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" \
        "$MODEL_ID" "$MODEL_DIR"
fi

activate_env
python - <<'PY'
import torch, vllm, verl, flash_attn, transformers
import quant_evolver.rft.verl_main  # noqa: F401
import synthmarket, synth_eval  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("vllm", vllm.__version__, "verl", verl.__version__, "flash_attn", flash_attn.__version__,
      "transformers", transformers.__version__)
assert torch.cuda.is_available(), "no CUDA device visible"
PY
python -m pytest -q tests
echo "setup OK: env=$ENV_PREFIX model=$MODEL_DIR"
