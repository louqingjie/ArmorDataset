#!/usr/bin/env bash
# 一键运行: 在 yolo conda 环境中完成 BaseLine 推理并生成 Test/ 下的结果表格图
# 用法:  bash run_grid.sh [传给 run_inference_grid.py 的额外参数]
set -euo pipefail

CONDA_SH="${CONDA_SH:-/home/wpie/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-yolo}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$(dirname "$0")"

python - <<'PY'
import importlib, subprocess, sys
for pkg, mod in (("onnxruntime", "onnxruntime"), ("pillow", "PIL"), ("opencv-python-headless", "cv2")):
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"[i] 安装缺失依赖: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

python run_inference_grid.py "$@"
