# -*- coding: utf-8 -*-
"""零依赖 Web 服务：ThreadingHTTPServer + 正则路由表 + 静态资源 + 端口回退。

启动: python -m webui.server --port 8765        (或 ./run_webui.sh)
仅监听 127.0.0.1；默认端口 8765，被占用时在 8765..8795 内自动回退。
"""
from __future__ import annotations

import argparse
import atexit
import errno
import json
import logging
import logging.handlers
import os
import re
import signal
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from webui import api

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG = logging.getLogger("webui")

ROUTES = []

MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".ico": "image/x-icon", ".woff2": "font/woff2", ".map": "application/json; charset=utf-8"}

MAX_BODY = 2 * 1024 * 1024


def route(method, pattern, local_only=False):
    """注册路由：pattern 为正则（严格匹配），处理器接收 Ctx 并返回 dict 或响应元组。

    local_only=True 的接口仅允许本机直连调用（经公网隧道访问返回 403）。
    """
    rx = re.compile("^" + pattern + "$")

    def deco(fn):
        ROUTES.append((method.upper(), rx, fn, local_only))
        return fn

    return deco


# --------------------------------------------------------------------------- #
# 路由表
# --------------------------------------------------------------------------- #
route("GET", r"/api/state")(api.api_state)
route("GET", r"/api/images")(api.api_images)
route("GET", r"/api/meta")(api.api_meta)
route("GET", r"/api/image")(api.api_image)
route("GET", r"/api/asset")(api.api_asset)
route("GET", r"/api/stats")(api.api_stats)
route("POST", r"/api/stats/refresh")(api.api_stats_refresh)
route("GET", r"/api/task")(api.api_task)
route("POST", r"/api/save")(api.api_save)
route("POST", r"/api/restore")(api.api_restore)
route("GET", r"/api/backups")(api.api_backups)
route("POST", r"/api/job/start", local_only=True)(api.api_job_start)   # GPU 流水线：仅本机可启动
route("POST", r"/api/job/stop", local_only=True)(api.api_job_stop)
route("GET", r"/api/job/status")(api.api_job_status)
route("GET", r"/api/job/log")(api.api_job_log)
route("POST", r"/api/deprecate")(api.api_deprecate)
route("GET", r"/api/deprecated")(api.api_deprecated_list)
route("POST", r"/api/deprecate/sync")(api.api_deprecate_sync)
route("POST", r"/api/export/preflight")(api.api_export_preflight)
route("POST", r"/api/export/run")(api.api_export_run)
route("GET", r"/api/export/status")(api.api_export_status)


class Ctx:
    """请求上下文：查询参数、JSON body 与便捷取值方法。"""

    def __init__(self, query, body, method, path, server_port, local=True):
        self.query = {k: v[0] for k, v in query.items()}
        self.body = body or {}
        self.method = method
        self.path = path
        self.server_port = server_port
        self.local = local              # False = 请求经公网隧道转发而来

    def q(self, name, default=None):
        v = self.query.get(name)
        return default if v is None or v == "" else v

    def qi(self, name, default=0):
        try:
            return int(self.q(name, default))
        except (TypeError, ValueError):
            return int(default)

    def qf(self, name, default=0.0):
        try:
            return float(self.q(name, default))
        except (TypeError, ValueError):
            return float(default)

    def qb(self, name, default=False):
        v = self.q(name)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "yes", "on")


class Handler(BaseHTTPRequestHandler):
    server_version = "ArmorAutoLabel"
    protocol_version = "HTTP/1.1"

    # ---- 基础输出 ----
    def log_message(self, fmt, *args):
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, status, ctype, body, headers=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            LOG.debug("客户端断开: %s", self.path)

    def _json(self, data, status=200, headers=None):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body, headers)

    def _respond(self, out):
        if isinstance(out, tuple):
            if len(out) == 3:
                status, ctype, body = out
                headers = None
            else:
                status, ctype, body, headers = out
            if isinstance(body, str):
                body = body.encode("utf-8")
            self._send(int(status), ctype, body, headers)
        else:
            self._json(out)

    # ---- 方法入口 ----
    def do_GET(self):
        self._handle("GET")

    def do_HEAD(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _is_local_request(self):
        """本机直连判定：Cloudflare 隧道转发时会强制附加下列代理头，公网请求无法去除；
        本机浏览器直连 127.0.0.1:8765 则不带这些头。"""
        for h in ("CF-Connecting-IP", "CF-Ray", "X-Forwarded-For", "X-Forwarded-Proto"):
            if self.headers.get(h):
                return False
        return True

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise api.ApiError("Content-Length 非法", 400)
        if n > MAX_BODY:
            raise api.ApiError("请求体过大", 413)
        if n == 0:
            return {}
        raw = self.rfile.read(n)
        ctype = (self.headers.get("Content-Type") or "").lower()
        if "json" not in ctype:
            raise api.ApiError("仅支持 application/json", 415)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise api.ApiError("JSON 解析失败: %s" % exc, 400)
        if not isinstance(data, dict):
            raise api.ApiError("请求体必须是 JSON 对象", 400)
        return data

    def _handle(self, method):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                body = self._read_body() if method == "POST" else {}
                local = self._is_local_request()
                ctx = Ctx(query, body, method, path, self.server.server_port, local)
                for m, rx, fn, local_only in ROUTES:
                    if m != method:
                        continue
                    if rx.match(path):
                        if local_only and not local:
                            LOG.warning("拒绝公网受限调用 %s %s (CF-Connecting-IP=%s)",
                                        method, path, self.headers.get("CF-Connecting-IP"))
                            raise api.ApiError("该操作仅限本机执行；公网访问已禁用任务启动/停止", 403)
                        return self._respond(fn(ctx))
                raise api.ApiError("未知接口: %s" % path, 404)
            return self._respond(self._static(path))        # 静态资源同样要写出响应
        except api.ApiError as exc:
            self._json({"ok": False, "error": str(exc)}, status=exc.status)
        except Exception as exc:                          # 兜底：不让服务因单个请求挂掉
            tb = traceback.format_exc(limit=6)
            LOG.error("请求失败 %s %s\n%s", method, path, tb)
            self._json({"ok": False, "error": "内部错误: %s: %s" % (type(exc).__name__, exc),
                        "trace": tb.splitlines()[-1][:300]}, status=500)

    # ---- 静态资源（离线单页） ----
    def _static(self, path):
        if path in ("/", "/index.html"):
            rel = "index.html"
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
        else:
            rel = path.lstrip("/")
        root = STATIC_DIR.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise api.ApiError("非法静态资源路径", 403)
        if not target.is_file():
            if rel != "index.html":                       # SPA 回退
                return self._static("/index.html")
            raise api.ApiError("前端资源缺失：请确认 webui/static/index.html 存在", 404)
        ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
        headers = {} if target.suffix.lower() in (".html",) else {}
        return (200, ctype, target.read_bytes(), headers)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def setup_logging(level="INFO"):
    LOG.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if LOG.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(STATE_DIR / "webui.log", maxBytes=5 * 1024 * 1024,
                                              backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(fh)


def bind(host, port, tries=30):
    for i in range(max(1, tries)):
        p = port + i
        try:
            httpd = Server((host, p), Handler)
            return httpd, p
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                LOG.warning("端口 %d 不可用(%s)，尝试下一个", p, exc.strerror or exc)
                continue
            raise
    raise SystemExit("端口 %d..%d 全部不可用" % (port, port + tries - 1))


def write_runtime_info(port, host):
    info = {"pid": os.getpid(), "host": host, "port": port,
            "url": "http://%s:%d/" % (host, port), "started": __import__("time").time(),
            "root": str(api.ROOT)}
    try:
        (STATE_DIR / "server.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    except OSError:
        pass
    return info


def shutdown_all():
    try:
        import webui.jobs as jobs
        jobs.stop(silent=True)
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m webui.server",
                                 description="装甲板自动标注 Web 工作台（零依赖）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--port-tries", type=int, default=30, help="端口占用时向后尝试的次数")
    ap.add_argument("--allow-root", action="append", default=[],
                    help="额外允许访问的根目录（可多次指定）")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    setup_logging(args.log_level)
    for r in args.allow_root:
        api.ALLOWED_ROOTS.append(Path(r).expanduser().resolve())

    httpd, port = bind(args.host, args.port, args.port_tries)
    info = write_runtime_info(port, args.host)
    atexit.register(shutdown_all)

    def _sig(_signum, _frame):
        LOG.info("收到退出信号，正在关闭…")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except (ValueError, OSError):
            pass

    print("\n  装甲板自动标注工作台已启动")
    print("  URL   : %s" % info["url"])
    print("  根目录: %s" % info["root"])
    print("  日志  : %s\n" % (STATE_DIR / "webui.log"))
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        shutdown_all()
        LOG.info("服务已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
