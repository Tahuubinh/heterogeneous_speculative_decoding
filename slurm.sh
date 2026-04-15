#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=draft
#SBATCH --mem=20GB
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --cpus-per-task=4

set -e

header="===== GPU allocation info ====="
echo "$header"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPUs (index, name, memory.total):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "nvidia-smi not found; cannot list GPUs."
fi

printf '%*s\n' "${#header}" '' | tr ' ' '='

SCRIPT_USER="${SUDO_USER:-$USER}"
VENV_PATH="/bigtemp/${SCRIPT_USER}/nlp_venv"
source "$VENV_PATH/bin/activate"

HF_CACHE_ROOT="/bigtemp/${SCRIPT_USER}/hf_cache"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

python -m evaluation.inference_sps \
    --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
    --drafter-path EleutherAI/pythia-160m \
    --model-id test_sps_open_eleuther \
    --question-file data/spec_bench/split_categories/rag.jsonl \
    --bench-name spec_bench \
    --temperature 0.0 \
    --dtype float16 \
    --draft-tokens 5 \
    --max-new-tokens 512

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
