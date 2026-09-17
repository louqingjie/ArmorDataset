#!/usr/bin/env bash
# 自动标注 Web 工作台启动脚本（conda 环境 yolo，零新增依赖）
#   用法: ./run_webui.sh                  # 默认 http://127.0.0.1:8765/
#         ./run_webui.sh --port 9000      # 指定端口（被占用时自动向后回退）
#         ./run_webui.sh --allow-root /data/datasets
set -euo pipefail
cd "$(dirname "$0")"
source activate yolo
exec python -m webui.server "$@"
