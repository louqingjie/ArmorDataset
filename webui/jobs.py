# -*- coding: utf-8 -*-
"""autolabel.run 子进程管理：启动 / 停止 / 续跑、进度聚合、日志环形缓冲与增量拉取。

* 子进程以 `start_new_session=True` 独立进程组启动，停止时对整组发信号，
  保证 spawn 出来的 worker 一并退出；
* 进度 = 本次运行新增的 `progress.w*.jsonl` 行数 + `progress.jsonl` 增量（合并后归入）；
* 日志：stdout/stderr 合并 → 环形缓冲（3000 行）+ `out/webui_job.log` 落盘，
  前端按 `since` 偏移增量拉取。
"""
from __future__ import annotations

import collections
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from webui import api

LOG = logging.getLogger("webui.jobs")
ROOT = api.ROOT
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")     # 去掉 onnxruntime 等输出的终端着色码

# 表单字段 -> CLI 参数（值与类型）
ARG_SPEC = {
    "images": ("--images", str), "out": ("--out", str),
    "limit": ("--limit", int), "sample": ("--sample", str), "seed": ("--seed", int),
    "workers": ("--workers", int), "class_mode": ("--class-mode", str),
    "expand": ("--expand", float), "min_roi_side": ("--min-roi-side", int),
    "conf": ("--conf", float), "max_det": ("--max-det", int),
    "min_refine_plate_w": ("--min-refine-plate-w", float),
    "refine_extra_thresholds": ("--refine-extra-thresholds", str),
    "agree_iou": ("--agree-iou", float), "agree_kpt_pct": ("--agree-kpt-pct", float),
    "consensus": ("--consensus", str), "panels": ("--panels", int),
    "sheet_cols": ("--sheet-cols", int),
}
FLAG_SPEC = {"no_val": "--no-val", "no_resume": "--no-resume", "stats_only": "--stats-only"}


class Job:
    def __init__(self):
        self.reset()

    def reset(self):
        self.id = None
        self.out = None
        self.cmd = None
        self.proc = None
        self.status = "idle"          # idle|running|done|stopped|error
        self.returncode = None
        self.started = None
        self.finished = None
        self.total = 0
        self.baseline = 0             # progress.jsonl 起始行数
        self.done = 0
        self.error = None
        self.log_path = None
        self.lines = collections.deque(maxlen=3000)
        self.line_count = 0           # 累计写入行数（含被挤出缓冲的）
        self._thread = None
        self._last_progress = None
        self._last_key = None

    # ------------------------------------------------------------------ 启动
    def start(self, payload: dict):
        if self.proc is not None and self.proc.poll() is None:
            raise api.ApiError("已有任务在运行，请先停止", 409)
        self.reset()

        out = api.safe_path(payload.get("out") or api.rel_to_root(api.C.DEFAULT_OUT),
                            must_exist=False)
        images = api.safe_path(payload.get("images") or api.rel_to_root(api.C.DEFAULT_IMAGES),
                               must_exist=True)
        if not images.is_dir():
            raise api.ApiError("图片目录不存在: %s" % images)

        args = [sys.executable, "-m", "autolabel.run", "--images", str(images), "--out", str(out)]
        for k, (flag, cast) in ARG_SPEC.items():
            if k in ("images", "out"):
                continue
            if payload.get(k) is None or payload.get(k) == "":
                continue
            try:
                val = cast(payload[k])
            except (TypeError, ValueError):
                raise api.ApiError("参数 %s 取值非法: %r" % (k, payload.get(k)))
            args += [flag, str(val)]
        for k, flag in FLAG_SPEC.items():
            if payload.get(k):
                args.append(flag)

        out.mkdir(parents=True, exist_ok=True)
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.out = out
        self.cmd = args
        self.baseline = _count_lines(out / "progress.jsonl")
        self.log_path = out / "webui_job.log"
        self.started = time.time()
        self.status = "running"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(args, cwd=str(ROOT), env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, errors="replace",
                                     start_new_session=True)
        self._thread = threading.Thread(target=self._pump, name="job-log", daemon=True)
        self._thread.start()
        LOG.info("任务启动 pid=%s out=%s: %s", self.proc.pid, api.rel_to_root(out),
                 " ".join(a if " " not in a else "'%s'" % a for a in args))
        self._append("[webui] 已启动: %s" % " ".join(a if " " not in a else "'%s'" % a for a in args))
        return {"ok": True, "job": self.snapshot()}

    # ------------------------------------------------------------------ 日志泵
    def _append(self, line):
        line = ANSI_RE.sub("", line).rstrip("\n")
        if not line:
            return
        self.lines.append(line)
        self.line_count += 1
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        # 解析控制台进度关键字
        if "处理 " in line and line.startswith("[input]"):
            self._parse_total(line)
        elif line.startswith("[resume]"):
            self._parse_total(line)

    def _parse_total(self, line):
        """[resume] 与 [input] 两行都会报数量，取最大值避免重复累加。"""
        m = re.search(r"处理 (\d+) 张", line)
        if m:
            self.total = max(self.total, int(m.group(1)))

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self._append(line)
        except Exception as exc:                       # 管道异常不应影响服务
            LOG.warning("日志读取中断: %s", exc)
        finally:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            self.returncode = self.proc.wait()
            self.finished = time.time()
            if self.status == "running":
                self.status = "done" if self.returncode == 0 else "error"
            self._append("[webui] 进程结束 rc=%s status=%s" % (self.returncode, self.status))
            api.invalidate_index(self.out)
            api._STATS_CACHE.pop(str(self.out.resolve()), None)
            LOG.info("任务结束 rc=%s status=%s", self.returncode, self.status)

    # ------------------------------------------------------------------ 进度
    def progress(self):
        if not self.out:
            return {"done": 0, "total": 0, "percent": 0.0, "ms_per_image": None, "current": None}
        merged = max(0, _count_lines(self.out / "progress.jsonl") - self.baseline)
        workers = sum(_count_lines(p) for p in self.out.glob("progress.w*.jsonl"))
        done = merged + workers
        self.done = done
        elapsed = (self.finished or time.time()) - (self.started or time.time())
        return {
            "done": done,
            "total": self.total or None,
            "percent": round(100.0 * done / self.total, 1) if self.total else None,
            "ms_per_image": round(1000.0 * elapsed / done, 1) if done else None,
            "elapsed_s": round(elapsed, 1),
        }

    def snapshot(self):
        running = bool(self.proc is not None and self.proc.poll() is None)
        return {
            "id": self.id, "out": api.rel_to_root(self.out) if self.out else None,
            "status": "running" if running else self.status,
            "running": running, "returncode": self.returncode,
            "started": self.started, "finished": self.finished,
            "log_path": api.rel_to_root(self.log_path) if self.log_path else None,
            "progress": self.progress(),
            "cmd": " ".join(self.cmd or []),
        }

    # ------------------------------------------------------------------ 停止
    def stop(self, silent=False):
        if self.proc is None or self.proc.poll() is not None:
            if not silent:
                return {"ok": True, "job": self.snapshot(), "note": "当前没有运行中的任务"}
            return {"ok": True}
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self._append("[webui] 已发送停止信号(SIGINT)")
        except ProcessLookupError:
            pass
        except Exception as exc:
            LOG.warning("停止失败: %s", exc)
        for _ in range(100):                       # 最多等 10s
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self._append("[webui] 强制终止(SIGKILL)")
            except Exception:
                pass
        self.status = "stopped"
        if not silent:
            return {"ok": True, "job": self.snapshot()}
        return {"ok": True}


def _count_lines(path: Path):
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


_JOB = Job()


def start(payload):
    return _JOB.start(payload or {})


def stop(silent=False):
    return _JOB.stop(silent=silent)


def status():
    return {"ok": True, "job": _JOB.snapshot()}


def log(since=0, limit=400):
    """增量返回日志：since 为上次已取到的行序号（从 1 开始）。"""
    since = max(0, int(since))
    total = _JOB.line_count
    buffered_start = total - len(_JOB.lines) + 1
    if since + 1 < buffered_start:                 # 请求过旧，已被环形缓冲挤出
        return {"ok": True, "lines": list(_JOB.lines), "next": total,
                "dropped": buffered_start - since - 1, "status": _JOB.snapshot()["status"]}
    offset = max(0, since + 1 - buffered_start)
    lines = list(_JOB.lines)[offset:offset + int(limit)]
    return {"ok": True, "lines": lines, "next": since + len(lines),
            "dropped": 0, "status": _JOB.snapshot()["status"]}
