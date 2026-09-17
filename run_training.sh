#!/usr/bin/env bash
# 一键训练：准备数据集 → 训练姿态学生 → 训练颜色/编号头 → 评估
#
#   ./run_training.sh                          # 用 training/config.yaml
#   ./run_training.sh training/config.yaml     # 指定配置
#   ./run_training.sh "" --set data.limit=300 --set pose.epochs=2   # 冒烟（空配置=默认）
#
# 参数会透传给每个子步骤（--set key=value 可覆盖配置项）。
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-training/config.yaml}"
shift || true
ARGS=("$@")
CONFIG_ARG=()
[ -n "$CONFIG" ] && CONFIG_ARG=(--config "$CONFIG")

source activate yolo

echo "=============== 1/4 准备数据 ==============="
python -m training.prepare "${CONFIG_ARG[@]}" "${ARGS[@]}"

echo "=============== 2/4 训练姿态模型 ==============="
python -m training.train_pose "${CONFIG_ARG[@]}" "${ARGS[@]}"

echo "=============== 3/4 训练颜色/编号分类头 ==============="
python -m training.train_cls "${CONFIG_ARG[@]}" "${ARGS[@]}"

echo "=============== 4/4 评估 ==============="
BEST=$(ls -t TrainSet/*/runs/pose/*/weights/best.pt 2>/dev/null | head -1 || true)
if [ -n "$BEST" ]; then
  python -m training.eval_kpts "${CONFIG_ARG[@]}" "${ARGS[@]}" --weights "$BEST"
else
  echo "未找到 best.pt，跳过评估"
fi
