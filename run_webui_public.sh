#!/usr/bin/env bash
# 公网数据集核对：标注工作台 + Cloudflare 隧道（https://yolo.pieblock.asia）
#   用法: ./run_webui_public.sh          # 启动 webui（若未运行）并挂起隧道（后台）
#         ./run_webui_public.sh --stop   # 停止隧道（webui 不受影响）
#         ./run_webui_public.sh --status # 查看隧道/公网状态
set -euo pipefail
cd "$(dirname "$0")"

TUNNEL="armor-webui"
CFG="$HOME/.cloudflared/armor-webui.yml"
DOMAIN="yolo.pieblock.asia"
PORT=8765
STATE="webui/state"
TUNNEL_LOG="$STATE/cloudflared.log"
TUNNEL_PID="$STATE/cloudflared.pid"
WEB_LOG="$STATE/webui_public.out"

healthy() { curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/api/state"; }

tunnel_alive() {
  [[ -f "$TUNNEL_PID" ]] && kill -0 "$(cat "$TUNNEL_PID")" 2>/dev/null
}

case "${1:-}" in
  --stop)
    if tunnel_alive; then kill "$(cat "$TUNNEL_PID")" && echo "隧道已停止 (pid $(cat "$TUNNEL_PID"))"; else
      pkill -f "cloudflared tunnel --config $CFG" 2>/dev/null && echo "隧道已停止" || echo "隧道未在运行"
    fi
    rm -f "$TUNNEL_PID"; exit 0 ;;
  --status)
    echo "webui : $(healthy && echo "运行中  http://127.0.0.1:$PORT" || echo '未运行')"
    echo "隧道  : $(tunnel_alive && echo "运行中  pid $(cat "$TUNNEL_PID")" || echo '未运行')"
    echo "公网  : https://$DOMAIN"
    exit 0 ;;
esac

# 1) webui：严格绑定 8765（隧道配置固定指向该端口，禁止回退）
if healthy; then
  echo "[1/2] webui 已在运行: http://127.0.0.1:$PORT"
else
  echo "[1/2] 启动 webui ..."
  source activate yolo
  nohup python -m webui.server --port "$PORT" --port-tries 1 >>"$WEB_LOG" 2>&1 &
  for _ in $(seq 1 20); do sleep 0.5; healthy && break; done
  if ! healthy; then
    echo "!! webui 启动失败：端口 $PORT 可能被占用，详见 $WEB_LOG"; exit 1
  fi
  echo "      webui 已启动: http://127.0.0.1:$PORT"
fi

# 2) 隧道：后台常驻，自动重连
if tunnel_alive; then
  echo "[2/2] 隧道已在运行 (pid $(cat "$TUNNEL_PID"))"
else
  echo "[2/2] 启动 Cloudflare 隧道 ..."
  nohup cloudflared tunnel --config "$CFG" run "$TUNNEL" >>"$TUNNEL_LOG" 2>&1 &
  echo $! >"$TUNNEL_PID"
  sleep 4
  tunnel_alive || { echo "!! 隧道启动失败，详见 $TUNNEL_LOG"; exit 1; }
  echo "      隧道已启动 (pid $(cat "$TUNNEL_PID"))"
fi

echo
echo "  公网地址: https://$DOMAIN"
echo "  日志    : $TUNNEL_LOG"
echo "  停止隧道: ./run_webui_public.sh --stop"
