#!/bin/bash
# c135 修正：mamba_ssm / pointnet2_ops 用 --no-build-isolation 重装（构建隔离里看不到已装的 torch）
set -uo pipefail
LOG="$HOME/setup_c135_fix.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== FIX START $(date) ==="

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semphyspc
export TORCH_CUDA_ARCH_LIST="7.5"
export CUDA_HOME="$CONDA_PREFIX"

echo "--- 预装构建工具链 ---"
pip install --no-cache-dir --quiet packaging setuptools wheel

echo "--- [A] mamba-ssm 2.2.4 (torch2.3 兼容, no-build-isolation) ---"
pip install --no-cache-dir --no-build-isolation mamba-ssm==2.2.4
python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"

echo "--- [B] pointnet2-ops 3.0.0 (no-build-isolation) ---"
pip install --no-cache-dir --no-build-isolation pointnet2-ops==3.0.0
python -c "import pointnet2_ops; print('pointnet2_ops ok')"

echo "--- [C] requirements ---"
cd "$HOME/semphyspc"
pip install --no-cache-dir -r requirements.txt

echo "=== FIX_DONE $(date) ==="
python -c "import torch, mamba_ssm, pointnet2_ops; print('FINAL torch', torch.__version__, 'mamba_ssm', mamba_ssm.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
