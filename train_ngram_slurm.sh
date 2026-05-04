#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=draft
#SBATCH --mem=30GB
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

# UltraChat and n-gram training configuration (override via env vars at submit time).
export ULTRACHAT_DATASET="${ULTRACHAT_DATASET:-HuggingFaceH4/ultrachat_200k}"
export ULTRACHAT_SPLIT="${ULTRACHAT_SPLIT:-train_sft}"
export ULTRACHAT_MAX_ROWS="${ULTRACHAT_MAX_ROWS:-0}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5}"
export CORPUS_PATH="${CORPUS_PATH:-data/corpora/ultrachat_train_sft.jsonl}"
export NGRAM_OUTPUT_TEMPLATE="${NGRAM_OUTPUT_TEMPLATE:-${NGRAM_OUTPUT_PATH:-artifacts/ngram/oasst_sft4_pythia12b_{order}gram_ultrachat.pkl.gz}}"

OUTPUT_PATH_EXAMPLE="${NGRAM_OUTPUT_TEMPLATE//\{order\}/1}"
mkdir -p "$(dirname "$CORPUS_PATH")" "$(dirname "$OUTPUT_PATH_EXAMPLE")"

python - <<'PY'
import json
import os

from datasets import get_dataset_split_names, load_dataset

dataset_name = os.environ["ULTRACHAT_DATASET"]
requested_split = os.environ["ULTRACHAT_SPLIT"]
max_rows = int(os.environ["ULTRACHAT_MAX_ROWS"])
corpus_path = os.environ["CORPUS_PATH"]

available_splits = get_dataset_split_names(dataset_name)
if requested_split in available_splits:
    split_name = requested_split
else:
    for fallback in ("train_sft", "train", "train_gen"):
        if fallback in available_splits:
            split_name = fallback
            break
    else:
        split_name = available_splits[0]
    print(
        f"Requested split '{requested_split}' not found for {dataset_name}. "
        f"Using '{split_name}' instead. Available: {available_splits}"
    )

print(f"Loading dataset: {dataset_name} [{split_name}]")
dataset = load_dataset(dataset_name, split=split_name)

rows_seen = 0
texts_written = 0

with open(corpus_path, "w", encoding="utf-8") as fout:
    for row in dataset:
        rows_seen += 1

        messages = row.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    content = content.strip()
                    if content:
                        fout.write(json.dumps({"text": content}, ensure_ascii=True) + "\n")
                        texts_written += 1
        else:
            for key in ("text", "prompt", "response", "completion"):
                value = row.get(key)
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        fout.write(json.dumps({"text": value}, ensure_ascii=True) + "\n")
                        texts_written += 1

        if max_rows > 0 and rows_seen >= max_rows:
            break

print(
    "Prepared UltraChat corpus:",
    f"rows={rows_seen}",
    f"texts={texts_written}",
    f"output={corpus_path}",
)
PY

resolve_output_path() {
    local template="$1"
    local order="$2"
    if [[ "$template" == *"{order}"* ]]; then
        echo "${template//\{order\}/$order}"
    else
        local base="$template"
        local ext=""
        if [[ "$template" == *.pkl.gz ]]; then
            base="${template%.pkl.gz}"
            ext=".pkl.gz"
        elif [[ "$template" == *.* ]]; then
            base="${template%.*}"
            ext=".${template##*.}"
        fi
        echo "${base}_${order}gram${ext}"
    fi
}

for ORDER in 1 2 3 4; do
    CURRENT_OUTPUT_PATH="$(resolve_output_path "$NGRAM_OUTPUT_TEMPLATE" "$ORDER")"
    mkdir -p "$(dirname "$CURRENT_OUTPUT_PATH")"

    echo "Training ${ORDER}-gram model -> ${CURRENT_OUTPUT_PATH}"
    python -m evaluation.train_ngram \
      --tokenizer-path "$TOKENIZER_PATH" \
      --corpus-path "$CORPUS_PATH" \
      --output-path "$CURRENT_OUTPUT_PATH" \
      --order "$ORDER" \
      --max-texts 0 \
      --jsonl-text-field text
done