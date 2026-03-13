#!/bin/bash
#SBATCH --job-name=specbench_base
#SBATCH -A cs6770_sp26grep "#SBATCH -A" run_llama13b_baseline.sh
#SBATCH --output=./slurm/slurm%j.out
#SBATCH --error=./slurm/slurm%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

set -e

cd /scratch/mft6zc/heterogeneous_speculative_decoding

module load gcc/11.4.0 openmpi/4.1.4
module load python/3.11.4
source specbench-env/bin/activate

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

python -m evaluation.inference_baseline \
  --model-path /scratch/mft6zc/models/vicuna-7b-v1.3 \
  --model-id vicuna-7b-v1.3-vanilla-float16-temp-0.0 \
  --bench-name spec_bench \
  --temperature 0.0 \
  --dtype float16