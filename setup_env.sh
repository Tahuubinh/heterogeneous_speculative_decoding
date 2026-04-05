conda create -n llm python=3.10 -y
conda activate llm
module load cuda/12.8.1

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.37.1
pip install accelerate==0.21.0
pip install fschat==0.2.31
pip install gradio==3.50.2
pip install openai==0.28.0
pip install anthropic==0.5.0
pip install sentencepiece==0.2.0
pip install protobuf==3.19.0
pip install maturin==0.12
pip install datasets==3.4.1


# This part is for mamba, you may not need to install this if you do not need to use mamba. Otherwise, if need a GPU to build mamba.
rm -rf causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.0
export MAX_JOBS=2
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install . --no-build-isolation --no-cache-dir
cd ../
rm -rf causal-conv1d

rm -rf mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.2.0.post1
export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS=2
export MAMBA_FORCE_BUILD=TRUE
pip install . --no-build-isolation --no-cache-dir
cd ..
rm -rf mamba
