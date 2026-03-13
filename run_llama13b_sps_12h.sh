#!/bin/bash
#SBATCH --job-name=llama13b_sps
#SBATCH -A cs6770_sp26
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

set -e

cd /sfs/weka/scratch/mft6zc/heterogeneous_speculative_decoding

module load gcc/11.4.0 openmpi/4.1.4
module load python/3.11.4
source specbench-env/bin/activate

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

python -m evaluation.inference_sps \
  --model-path /sfs/weka/scratch/mft6zc/models/Llama-2-13b-hf \
  --drafter-path /sfs/weka/scratch/mft6zc/models/open_llama_3b_v2 \
  --model-id Llama-2-13b-hf-sps-openllama3b-float16-temp-0.0-12h \
  --bench-name spec_bench \
  --temperature 0.0 \
  --dtype float16