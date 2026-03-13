#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=analysisreg
#SBATCH --mem=64GB
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --cpus-per-task=8

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

python -m evaluation.inference_sps \
    --model-path EleutherAI/pythia-6.9b \
    --drafter-path EleutherAI/pythia-160m \
    --model-id pythia-6.9b-sps-pythia-160m \
    --question-file data/spec_bench/split_categories/summarization.jsonl \
    --bench-name spec_bench \
    --temperature 0.0 \
    --dtype float16