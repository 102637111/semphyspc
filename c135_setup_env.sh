#!/bin/bash
# c135 训练环境搭建（Ubuntu 18.04 / 3x2080Ti sm_75 / glibc2.27 -> torch<=2.3.1+cu118）
# 运行：ssh c135 'bash -s' < c135_setup_env.sh  （或 scp 后 bash 运行）
set -uo pipefail
LOG="$HOME/setup_c135_env.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) ==="

export TORCH_CUDA_ARCH_LIST="7.5"

# 1. Miniconda (py310)
if [ ! -d "$HOME/miniconda3" ]; then
  echo "--- [1/7] install miniconda ---"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
else
  echo "--- [1/7] miniconda already present ---"
fi
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "conda: $(conda --version)"

# 2. env semphyspc (py3.10)
if conda env list | grep -qw semphyspc; then
  echo "--- [2/7] env semphyspc already present ---"
else
  echo "--- [2/7] create env semphyspc ---"
  conda create -y -n semphyspc python=3.10
fi
conda activate semphyspc
echo "python: $(python --version)"

# 3. torch 2.3.1 + cu118
echo "--- [3/7] install torch 2.3.1+cu118 ---"
pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# 4. cuda-toolkit 11.8 (提供 nvcc)
echo "--- [4/7] install cuda-toolkit 11.8 ---"
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
echo "nvcc: $(which nvcc)"; nvcc --version | tail -1

# 5. mamba_ssm（先试与本机一致的 2.3.2，失败回退 torch2.3 兼容的 2.2.4）
echo "--- [5/7] install mamba-ssm ---"
if ! pip install --no-cache-dir mamba-ssm==2.3.2.post1; then
  echo "mamba-ssm 2.3.2 失败，回退 2.2.4"
  pip install --no-cache-dir mamba-ssm==2.2.4
fi
python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"

# 6. pointnet2_ops（针对 sm_75 编译）
echo "--- [6/7] install pointnet2-ops ---"
pip install --no-cache-dir pointnet2-ops==3.0.0
python -c "import pointnet2_ops; print('pointnet2_ops ok')"

# 7. requirements（在 repo 根下运行，需先 rsync 代码）
echo "--- [7/7] install requirements ---"
if [ -f "$HOME/semphyspc/requirements.txt" ]; then
  cd "$HOME/semphyspc"
  pip install --no-cache-dir -r requirements.txt
else
  echo "WARN: $HOME/semphyspc/requirements.txt 不存在，跳过（先 rsync 代码）"
fi

echo "=== ENV_DONE $(date) ==="
python -c "import torch, mamba_ssm, pointnet2_ops; print('FINAL torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
