#!/bin/bash
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00

set -e
source /u/nzj6jt/miniconda3/etc/profile.d/conda.sh
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.8.1
conda activate mamba_llm

PYTHONPATH=. python evaluation/inference_sps_mamba.py \
  --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
  --drafter-path state-spaces/mamba-130m \
  --model-id mamba-replay \
  --question-file data/spec_bench/split_categories/translation.jsonl \
  --dtype float16 \
  --max-new-tokens 512 \
  --mamba-state-strategy reset_replay \
  --mamba-use-cuda-graphs \
  --mamba-cg-warmups 2

PYTHONPATH=. python evaluation/inference_sps.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --drafter-path EleutherAI/pythia-160m     --model-id transformer     --question-file data/spec_bench/split_categories/translation.jsonl     --dtype float16     --max-new-tokens 512
PYTHONPATH=. python evaluation/inference_baseline.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --model-id baseline     --question-file data/spec_bench/split_categories/translation.jsonl     --dtype float16     --max-new-tokens 512


PYTHONPATH=. python evaluation/inference_sps_mamba.py \
  --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
  --drafter-path state-spaces/mamba-130m \
  --model-id mamba-replay \
  --question-file data/spec_bench/split_categories/math_reasoning.jsonl \
  --dtype float16 \
  --max-new-tokens 512 \
  --mamba-state-strategy reset_replay \
  --mamba-use-cuda-graphs \
  --mamba-cg-warmups 2

PYTHONPATH=. python evaluation/inference_sps.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --drafter-path EleutherAI/pythia-160m     --model-id transformer     --question-file data/spec_bench/split_categories/math_reasoning.jsonl     --dtype float16     --max-new-tokens 512
PYTHONPATH=. python evaluation/inference_baseline.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --model-id baseline     --question-file data/spec_bench/split_categories/math_reasoning.jsonl     --dtype float16     --max-new-tokens 512


PYTHONPATH=. python evaluation/inference_sps_mamba.py \
  --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
  --drafter-path state-spaces/mamba-130m \
  --model-id mamba-replay \
  --question-file data/spec_bench/split_categories/rag.jsonl \
  --dtype float16 \
  --max-new-tokens 512 \
  --mamba-state-strategy reset_replay \
  --mamba-use-cuda-graphs \
  --mamba-cg-warmups 2

PYTHONPATH=. python evaluation/inference_sps.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --drafter-path EleutherAI/pythia-160m     --model-id transformer     --question-file data/spec_bench/split_categories/rag.jsonl     --dtype float16     --max-new-tokens 512
PYTHONPATH=. python evaluation/inference_baseline.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --model-id baseline     --question-file data/spec_bench/split_categories/rag.jsonl     --dtype float16     --max-new-tokens 512


PYTHONPATH=. python evaluation/inference_sps_mamba.py \
  --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
  --drafter-path state-spaces/mamba-130m \
  --model-id mamba-replay \
  --question-file data/spec_bench/split_categories/coding.jsonl \
  --dtype float16 \
  --max-new-tokens 512 \
  --mamba-state-strategy reset_replay \
  --mamba-use-cuda-graphs \
  --mamba-cg-warmups 2

PYTHONPATH=. python evaluation/inference_sps.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --drafter-path EleutherAI/pythia-160m     --model-id transformer     --question-file data/spec_bench/split_categories/coding.jsonl     --dtype float16     --max-new-tokens 512
PYTHONPATH=. python evaluation/inference_baseline.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --model-id baseline     --question-file data/spec_bench/split_categories/coding.jsonl     --dtype float16     --max-new-tokens 512


PYTHONPATH=. python evaluation/inference_sps_mamba.py \
  --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
  --drafter-path state-spaces/mamba-130m \
  --model-id mamba-replay \
  --question-file data/spec_bench/split_categories/summarization.jsonl \
  --dtype float16 \
  --max-new-tokens 512 \
  --mamba-state-strategy reset_replay \
  --mamba-use-cuda-graphs \
  --mamba-cg-warmups 2


PYTHONPATH=. python evaluation/inference_sps.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --drafter-path EleutherAI/pythia-160m     --model-id transformer     --question-file data/spec_bench/split_categories/summarization.jsonl     --dtype float16     --max-new-tokens 512
PYTHONPATH=. python evaluation/inference_baseline.py     --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5     --model-id baseline     --question-file data/spec_bench/split_categories/summarization.jsonl     --dtype float16     --max-new-tokens 512
