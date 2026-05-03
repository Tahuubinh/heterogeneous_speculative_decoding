#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=draft
#SBATCH --mem=30GB
#SBATCH --output=./slurm/slurm_ngram%j.out
#SBATCH --error=./slurm/slurm_ngram%j.err
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

# n-gram inference configuration (override at submit time, e.g. NGRAM_MODEL_TEMPLATE=... sbatch slurm.sh).
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5}"
NGRAM_MODEL_TEMPLATE="${NGRAM_MODEL_TEMPLATE:-artifacts/ngram/oasst_sft4_pythia12b_{order}gram_ultrachat.pkl.gz}"
MODEL_ID_PREFIX="${MODEL_ID_PREFIX:-${MODEL_ID:-test_ngram_open_ultrachat}}"
NGRAM_ORDERS="${NGRAM_ORDERS:-1 2 3 4}"
# Directory containing per-category question files (jsonl). Will iterate all *.jsonl inside.
QUESTION_DIR="${QUESTION_DIR:-data/spec_bench/split_categories}"
# Bench name base (used when question-file is not provided by the script).
BENCH_NAME="${BENCH_NAME:-spec_bench}"
D_TYPE="${D_TYPE:-float16}"
# Draft candidates to iterate (space-separated list)
DRAFT_CANDIDATES="${DRAFT_CANDIDATES:-3 5 7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

resolve_ngram_model_path() {
    local template="$1"
    local order="$2"
    local candidate="$template"

    if [[ "$template" == *"{order}"* ]]; then
        candidate="${template//\{order\}/$order}"
    fi

    if [[ -f "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    # Backward-compatible fallback for old artifact names that were produced incorrectly.
    local legacy_candidate="artifacts/ngram/oasst_sft4_pythia12b_{ordergram_ultrachat.pkl_${order}gram.gz}"
    if [[ -f "$legacy_candidate" ]]; then
        echo "$legacy_candidate"
        return 0
    fi

    if [[ -d "artifacts/ngram" ]]; then
        local discovered
        discovered="$(find artifacts/ngram -maxdepth 1 -type f -name "*${order}gram*" ! -name "*.stats.json" | head -n 1)"
        if [[ -n "$discovered" ]]; then
            echo "$discovered"
            return 0
        fi
    fi

    return 1
}

if [[ ! -d "$QUESTION_DIR" ]]; then
    echo "Question directory not found: $QUESTION_DIR"
    exit 1
fi

for QUESTION_FILE in "$QUESTION_DIR"/*.jsonl; do
    if [[ ! -f "$QUESTION_FILE" ]]; then
        echo "No question files found in $QUESTION_DIR"
        exit 1
    fi

    TASK_NAME="$(basename "$QUESTION_FILE" .jsonl)"

    for ORDER in $NGRAM_ORDERS; do
        if ! [[ "$ORDER" =~ ^[0-9]+$ ]] || [[ "$ORDER" -lt 1 ]]; then
            echo "Invalid n-gram order in NGRAM_ORDERS: $ORDER"
            echo "Expected a space-separated list of positive integers, e.g. '1 2 3 4'."
            exit 1
        fi

        CURRENT_NGRAM_MODEL_PATH="$(resolve_ngram_model_path "$NGRAM_MODEL_TEMPLATE" "$ORDER")" || {
            echo "Missing n-gram model file for ${ORDER}-gram."
            echo "Tried template path: ${NGRAM_MODEL_TEMPLATE//\{order\}/$ORDER}"
            echo "Train the model with train_ngram_slurm.sh or set NGRAM_MODEL_TEMPLATE accordingly."
            exit 1
        }

        for DRAFT in $DRAFT_CANDIDATES; do
            if ! [[ "$DRAFT" =~ ^[0-9]+$ ]] || [[ "$DRAFT" -lt 1 ]]; then
                echo "Invalid draft candidate value: $DRAFT"
                exit 1
            fi

            CURRENT_MODEL_ID="${MODEL_ID_PREFIX}_${ORDER}gram_draft${DRAFT}_${TASK_NAME}"
            echo "Running ${ORDER}-gram inference for task=${TASK_NAME} (draft=${DRAFT})"
            echo "  model path: ${CURRENT_NGRAM_MODEL_PATH}"
            echo "  model id:   ${CURRENT_MODEL_ID}"
            echo "  question file: ${QUESTION_FILE}"

            python -m evaluation.inference_ngram \
                --model-path "$TARGET_MODEL_PATH" \
                --ngram-model-path "$CURRENT_NGRAM_MODEL_PATH" \
                --model-id "$CURRENT_MODEL_ID" \
                --question-file "$QUESTION_FILE" \
                --bench-name "$BENCH_NAME" \
                --dtype "$D_TYPE" \
                --draft-num-candidate-tokens "$DRAFT" \
                --max-new-tokens "$MAX_NEW_TOKENS"
        done
    done
done
