# -*- coding: utf-8 -*-
"""训练脚本共用的小工具：配置加载、路径解析、日志。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=None):
    """读取训练配置，返回 (cfg, cfg_path)。支持 key=value 覆盖（--set）。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise SystemExit("配置文件不存在: %s" % cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return cfg, cfg_path


def apply_overrides(cfg, pairs):
    """命令行 --set a.b=1 形式的覆盖（值是 yaml 字面量）。"""
    for item in pairs or []:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def paths(cfg):
    """返回 (workdir, dataset_dir, models_dir, runs_dir, cls_dir, eval_dir)，均已绝对化。"""
    work = Path(cfg["data"]["workdir"])
    if not work.is_absolute():
        work = ROOT / work
    ds = work / "dataset"
    return (work, ds, work / "models", work / "runs", work / "cls", work / "eval")


def resolve(p):
    p = Path(str(p))
    return p if p.is_absolute() else (ROOT / p)


def parse_imgsz(v, default=640):
    """imgsz 归一化：整数=正方形；"512,640" / [512,640] / (512,640) = 高×宽（ultralytics 顺序）。

    注意 ultralytics 的矩形输入是 [height, width]，和常见书写 "640x512"（宽×高）相反。
    """
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return int(v[0])
        if len(v) >= 2:
            return [int(v[0]), int(v[1])]
    s = str(v).replace("x", ",").replace("×", ",").replace(" ", "")
    parts = [p for p in s.split(",") if p]
    if len(parts) == 1:
        return int(float(parts[0]))
    return [int(float(parts[0])), int(float(parts[1]))]


def imgsz_str(v):
    """给日志用的可读形式：640 或 512x640(高x宽)。"""
    v = parse_imgsz(v)
    return "%d" % v if isinstance(v, int) else "%dx%d" % (v[0], v[1])


def check_imgsz_orientation(imgsz, landscape=True):
    """防呆：本库图片 97% 是横图（宽>高），输入若写成竖图（高>宽）多半是顺序写反。

    ultralytics 的 imgsz 是 [高, 宽]；"640x512"（宽×高）应写成 [512, 640]。
    """
    if isinstance(imgsz, list) and len(imgsz) == 2 and landscape and imgsz[0] > imgsz[1]:
        warn("imgsz=[%d, %d] 是竖图（高>宽），而本库图片几乎全是横图："
             "顺序应为 [高, 宽]，想表达 宽640×高512 请写 [512, 640]" % (imgsz[0], imgsz[1]))
    return imgsz


def report_cb(prefix=""):
    """传给 webui.dataset._export_task 的同步进度回调。"""
    state = {"pct": -10}

    def cb(p, msg=""):
        pct = int(round(float(p) * 100))
        if pct >= state["pct"] + 10 or pct >= 100:
            state["pct"] = pct
            print("  %s%3d%% %s" % (prefix, pct, msg), flush=True)

    return cb


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def kv_table(title, rows, widths=None):
    """打印对齐的小表格。rows = [[c1, c2, ...], ...]，第一行作表头。"""
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    rows = [list(map(str, r)) + [""] * (ncol - len(r)) for r in rows]
    if widths is None:
        widths = [max(_w(r[i], i) for r in rows) for i in range(ncol)]
    line = "  ".join("%-*s" % (widths[i], rows[0][i]) for i in range(ncol))
    print(title)
    print("  " + line)
    print("  " + "-" * len(line))
    for r in rows[1:]:
        print("  " + "  ".join("%-*s" % (widths[i], r[i]) for i in range(ncol)))


def _w(s, i):
    # 中文字符按 2 列宽估算，避免表格错位
    n = 0
    for ch in str(s):
        n += 2 if ord(ch) > 0x2E80 else 1
    return n


def warn(msg):
    print("[warn] %s" % msg, file=sys.stderr, flush=True)
