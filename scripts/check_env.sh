#!/bin/bash
# ============================================================
# 环境检查脚本
# ============================================================

echo "=========================================="
echo "环境检查"
echo "=========================================="

# 检查当前conda环境
echo -e "\n[1] Conda环境:"
echo "当前环境: $CONDA_DEFAULT_ENV"
conda env list | grep "*"

# 检查Python版本
echo -e "\n[2] Python版本:"
python --version
which python

# 检查PyTorch版本
echo -e "\n[3] PyTorch版本:"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

# 检查MMDetection
echo -e "\n[4] MMDetection:"
python -c "import mmdet; print('MMDet:', mmdet.__version__)"

# 检查NumPy
echo -e "\n[5] NumPy版本:"
python -c "import numpy; print('NumPy:', numpy.__version__)"

echo -e "\n=========================================="
echo "期望的环境配置:"
echo "=========================================="
echo "Conda环境: fusionkd_lxy"
echo "Python: 3.8.x"
echo "PyTorch: 1.12.1"
echo "CUDA: 11.6"
echo "NumPy: <2.0"
