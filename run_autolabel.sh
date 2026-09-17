#!/usr/bin/env bash
# 自动标注流水线入口（conda 环境 yolo）
# 用法:
#   ./run_autolabel.sh                      # 默认试点 1000 张 -> AutoLabel/
#   ./run_autolabel.sh --limit 0 --workers 4 # 全量（自动断点续跑）
#   ./run_autolabel.sh --images BaseLine --limit 0 --no-val --out AutoLabel_smoke  # 冒烟
#   ./run_autolabel.sh --stats-only          # 只重算统计/报告/面板
set -euo pipefail
cd "$(dirname "$0")"
source activate yolo
exec python -m autolabel.run "$@"
