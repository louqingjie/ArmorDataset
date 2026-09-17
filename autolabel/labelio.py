# -*- coding: utf-8 -*-
"""标签输出：YOLO-pose txt（4 点）+ 同名 meta json（原子写）。

txt 行格式（Ultralytics 姿态任务，kpt_shape=[4,2]，无可见性）:
    cls cx cy w h x1 y1 x2 y2 x3 y3 x4 y4      # 全部归一化到 [0,1]
关键点顺序固定 [左上, 左下, 右下, 右上]（config.KPT_ORDER）。
cls 取 0（--class-mode single，颜色/编号在 json 中）或 color*9+num（merged）。
无目标的图片不写 txt（Ultralytics 视作纯背景负样本），但始终写 meta json 便于断点续跑与统计。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from . import config as C

REVIEW_FLAG_PREFIXES = ("conflict", "no_match", "refine_failed", "tiny")


def class_id(obj, mode=C.CLASS_MODE):
    if mode == "merged":
        return int(obj["color"]) * 9 + int(obj["num"])
    return 0


def yolo_line(obj, img_w, img_h, mode=C.CLASS_MODE):
    """单个目标 -> 一行 YOLO-pose 标签（归一化、clamp 到 [0,1]）。"""
    q = np.asarray(obj["quad_final"], np.float32)
    xs = np.clip(q[:, 0] / float(img_w), 0.0, 1.0)
    ys = np.clip(q[:, 1] / float(img_h), 0.0, 1.0)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    kpts = " ".join("%.6f %.6f" % (x, y) for x, y in zip(xs, ys))
    return "%d %.6f %.6f %.6f %.6f %s" % (class_id(obj, mode), cx, cy, bw, bh, kpts)


def needs_review(obj):
    """是否需要人工复核；被废弃的目标（如大装甲）不再进复核队列。"""
    if obj.get("deprecated"):
        return False
    return _needs_review_flags(obj)


def _needs_review_flags(obj):
    """是否需要人工复核（冲突/单侧检出/精修失败/极小目标）。"""
    return any(any(f.startswith(p) for p in REVIEW_FLAG_PREFIXES) for f in obj["flags"])


def object_meta(obj, img_w, img_h):
    """输出到 meta json 的目标结构（像素坐标，去掉内部中间量）。"""
    # 剔除内部中间量(下划线开头)与 ndarray 字段(仅保留 *_px 的 list 版本)
    drop = {"quad_coarse", "quad_final"}
    out = {k: v for k, v in obj.items()
           if not k.startswith("_") and k not in drop and not isinstance(v, np.ndarray)}
    out["quad_final_px"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in obj["quad_final"]]
    out["quad_coarse_px"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in obj["quad_coarse"]]
    out["norm_size"] = [round(float(img_w), 1), round(float(img_h), 1)]
    out["review"] = needs_review(obj)
    # 同时保留两教师一致性中间量，便于统计报告与二次筛选
    for k in ("_iou_two_teachers", "_kpt_diff_px", "_kpt_diff_pct", "_d_secondary_px"):
        if k in obj and obj[k] is not None:
            out[k.lstrip("_")] = obj[k]
    return out


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_image_labels(out_dir, key, img_w, img_h, objects, image_name=None,
                       image_path=None, rel=None, teachers=(), run_config=None,
                       mode=C.CLASS_MODE):
    """写出 labels/<key>.txt 与 meta/<key>.json（key 为相对路径，可含子目录）。

    key 用"相对输入目录的路径去扩展名"保证唯一（SUM/13.jpg 与 SUM/val/13.png 不再互相覆盖），
    同时保持 Ultralytics 的 images/labels 目录镜像布局（labels/val/13.txt ↔ images/val/13.jpg）。
    """
    out_dir = Path(out_dir)
    labels_dir, meta_dir = out_dir / "labels", out_dir / "meta"
    key = str(key).replace("\\", "/")
    name = image_name or (key.split("/")[-1] + ".jpg")

    lines = [yolo_line(o, img_w, img_h, mode) for o in objects]
    txt_path = labels_dir / (key + ".txt")
    if lines:
        _atomic_write(txt_path, "\n".join(lines) + "\n")
    elif txt_path.exists():
        txt_path.unlink()          # 重跑后变背景图，避免残留旧标签

    meta = {
        "key": key, "image": name, "rel": rel or key, "path": image_path,
        "size": [int(img_h), int(img_w)],
        "teachers": list(teachers), "config": dict(run_config or {}),
        "class_mode": mode, "kpt_order": list(C.KPT_ORDER),
        "n_obj": len(objects), "n_review": sum(1 for o in objects if needs_review(o)),
        "objects": [object_meta(o, img_w, img_h) for o in objects],
    }
    json_path = meta_dir / (key + ".json")
    _atomic_write(json_path, json.dumps(meta, ensure_ascii=False, indent=1))
    return {"key": key, "n_obj": len(objects), "n_review": meta["n_review"],
            "txt": str(txt_path) if lines else None, "json": str(json_path),
            "flags": [f for o in objects for f in o["flags"]]}


def patch_meta(out_dir, key, extra: dict):
    """在已写出的 meta json 上原子合并额外字段（Web 端人工修正审计等）。"""
    p = Path(out_dir) / "meta" / (str(key) + ".json")
    if not p.exists():
        raise FileNotFoundError(str(p))
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta.update(extra or {})
    _atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def write_edited_labels(out_dir, key, img_w, img_h, objects, extra_meta=None, **kwargs):
    """人工修正后回写：复用 write_image_labels 保证格式一致，再补审计字段。"""
    res = write_image_labels(out_dir, key, img_w, img_h, objects, **kwargs)
    if extra_meta:
        patch_meta(out_dir, key, extra_meta)
    return res


def review_rows(out_dir):
    """汇总全部 meta json -> 待复核清单行（供 CSV/面板使用）。"""
    meta_dir = Path(out_dir) / "meta"
    rows = []
    for p in sorted(meta_dir.rglob("*.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        for i, o in enumerate(meta["objects"]):
            if not o.get("review") or o.get("deprecated"):
                continue
            rows.append({
                "image": meta.get("key", meta["image"]), "obj": i, "source": o.get("source"),
                "flags": "|".join(o.get("flags", [])), "reason": o.get("reason") or "",
                "color": "%s(%d)" % (o.get("color_name", "?"), o.get("color", -1)),
                "num": "%s(%d)" % (o.get("num_name", "?"), o.get("num", -1)),
                "score_p": o.get("score_primary"), "score_s": o.get("score_secondary"),
                "plate_w": o.get("plate_w"), "iou2": o.get("iou_two_teachers"),
                "kpt_diff_pct": o.get("kpt_diff_pct"),
                "d_refine_px": (o.get("refine") or {}).get("d_coarse_to_refine_px"),
            })
    return rows
