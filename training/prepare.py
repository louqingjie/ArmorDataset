# -*- coding: utf-8 -*-
"""训练集准备：① 导出 YOLO-pose 数据集 ② 生成模型结构 yaml ③ 构建颜色/编号裁块数据集。

用法:
  python -m training.prepare                       # 用 training/config.yaml
  python -m training.prepare --set data.limit=300   # 冒烟：只取均衡子集前 300 张
  python -m training.prepare --skip-crops           # 只导姿态数据集
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

from training import model_yaml
from training.common import (ROOT, apply_overrides, kv_table, load_config, paths,
                             report_cb, resolve, warn, write_json)
from webui import api as W
from webui import dataset as DS

CLASS_ORDER = ["B", "R", "G", "N"]
NUM_ORDER = ["1", "2", "3", "4", "5", "7", "O", "B", "LB"]


# --------------------------------------------------------------------------- #
# ① 姿态数据集
# --------------------------------------------------------------------------- #
def export_dataset(cfg):
    data = cfg["data"]
    out_dir = W.safe_path(str(data["source_out"]), must_exist=True)
    _, dest, _, _, _, _ = paths(cfg)
    if dest.exists():
        for sub in ("images", "labels"):
            shutil.rmtree(dest / sub, ignore_errors=True)
    payload = {
        "out": W.rel_to_root(out_dir), "dest": str(dest),
        "filter": data.get("filter", "all"),
        "object_filter": data.get("object_filter", "keep_all"),
        "order": data.get("order", "balanced"),
        "split_mode": data.get("split_mode", "ratio"),
        "val_ratio": float(data.get("val_ratio", 0.1)),
        "class_mode": data.get("class_mode", "single"),
        "image_mode": data.get("image_mode", "symlink"),
        "limit": int(data.get("limit", 0)),
        "seed": int(data.get("seed", 0)),
        "include_background": bool(data.get("include_background")),
        "include_deprecated": False,
        "overwrite": True,
    }
    print("[1/3] 导出姿态数据集 -> %s" % dest.relative_to(ROOT))
    res = DS.preflight(payload)
    print(DS.describe_order(out_dir, res))
    print("  开始导出（filter=%s，image_mode=%s）…" % (payload["filter"], payload["image_mode"]))
    summary = DS._export_task(report_cb("  "), out_dir, dest, payload)
    c = summary["counts"]
    print("  train: %d 图 / %d 目标    val: %d 图 / %d 目标   跳过 %d   失败 %d"
          % (c["train"]["images"], c["train"]["objects"], c["val"]["images"],
             c["val"]["objects"], summary["n_skipped"], summary["n_errors"]))
    if summary.get("split_mode") == "group":
        print("  划分: 按来源分组隔离（holdout=%s，不抽样）" % "/".join(summary.get("holdout_groups") or []))
    else:
        print("  划分: 按比例抽样 val_ratio=%.2f" % float(data.get("val_ratio") or 0))
    print("  来源分组构成: %s" % json.dumps(summary.get("groups") or {}, ensure_ascii=False))
    print("  排除废弃目标 %d 个；data.yaml: %s"
          % (summary["n_deprecated_objects_excluded"], Path(summary["data_yaml"]).name))
    return out_dir, dest, summary


# --------------------------------------------------------------------------- #
# ② 模型结构 yaml
# --------------------------------------------------------------------------- #
def build_yamls(cfg):
    _, _, models, _, _, _ = paths(cfg)
    pose = cfg["pose"]
    kpt = tuple(pose.get("kpt_shape", (4, 2)))
    flip = tuple(pose.get("flip_idx", (3, 2, 1, 0)))
    print("\n[2/3] 生成模型结构 yaml -> %s" % models.relative_to(ROOT))
    targets = [pose["student"]]
    for extra in cfg.get("pose_control", []) or []:        # 可选：对照组模型
        if extra not in targets:
            targets.append(extra)
    out = {}
    for fam in targets:
        p = models / ("%s-armor.yaml" % fam)
        out[fam] = model_yaml.build(fam, p, nc=1, kpt_shape=kpt, flip_idx=flip)
    return out


# --------------------------------------------------------------------------- #
# ③ 颜色 / 编号裁块数据集
# --------------------------------------------------------------------------- #
def _split_map(dataset_dir: Path):
    """导出的图片文件名 -> split（train/val）。key 里的 '/' 在导出时被换成 '__'。"""
    m = {}
    for split in ("train", "val"):
        d = dataset_dir / "images" / split
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                m[f.name.rsplit(".", 1)[0].replace("__", "/")] = split
    return m


def build_crops(cfg, out_dir: Path, dataset_dir: Path):
    cls_cfg = cfg["cls"]
    if not cls_cfg.get("enabled", True):
        print("\n[3/3] 分类头数据集：已禁用（cls.enabled=false）")
        return None
    work, _, _, _, cls_dir, _ = paths(cfg)
    size = int(cls_cfg.get("size", 64))
    pad = float(cls_cfg.get("pad", 1.25))
    min_w = float(cls_cfg.get("min_plate_w", 16))
    splits = _split_map(dataset_dir)
    print("\n[3/3] 构建颜色/编号裁块数据集 -> %s（%d 张图）" % (cls_dir.relative_to(ROOT), len(splits)))
    cnt_color, cnt_num = Counter(), Counter()
    n_obj = n_skip_small = n_skip_img = 0
    cache_key, cache = None, None
    for key, split in sorted(splits.items()):
        try:
            meta = W.read_meta(out_dir, key)
            img_path = W.resolve_image_path(meta, out_dir)
        except Exception:
            n_skip_img += 1
            continue
        if key != cache_key:
            cache_key = key
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cache = img
        img = cache
        if img is None:
            n_skip_img += 1
            continue
        h, w = img.shape[:2]
        for i, o in enumerate(meta.get("objects") or []):
            if o.get("deprecated"):
                continue
            if float(o.get("plate_w") or 0) < min_w:
                n_skip_small += 1
                continue
            q = o.get("quad_final_px")
            if not q:
                continue
            xs = [p[0] for p in q]
            ys = [p[1] for p in q]
            cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            bw, bh = max(4.0, (max(xs) - min(xs)) * pad), max(4.0, (max(ys) - min(ys)) * pad)
            x1 = max(0, int(round(cx - bw / 2)))
            y1 = max(0, int(round(cy - bh / 2)))
            x2 = min(w, int(round(cx + bw / 2)))
            y2 = min(h, int(round(cy + bh / 2)))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tile = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            flat = "%s_%d.jpg" % (str(key).replace("/", "__"), i)
            cname, nname = o.get("color_name") or "?", o.get("num_name") or "?"
            for task, name, counter in (("color", cname, cnt_color), ("num", nname, cnt_num)):
                d = cls_dir / "crops" / task / split / name
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / flat), tile, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                counter[(split, name)] += 1
            n_obj += 1

    def table(task, order, counter):
        rows = [["类别", "train", "val", "合计"]]
        for name in order:
            tr, va = counter[("train", name)], counter[("val", name)]
            if tr or va:
                rows.append([name, tr, va, tr + va])
        rows.append(["合计", sum(v for (s, _), v in counter.items() if s == "train"),
                     sum(v for (s, _), v in counter.items() if s == "val"),
                     sum(counter.values())])
        kv_table("  %s 类别分布：" % ("颜色" if task == "color" else "编号"), rows)

    print("  裁块 %d 个（跳过 板宽<%gpx %d 个；读图失败 %d 张）"
          % (n_obj, min_w, n_skip_small, n_skip_img))
    table("color", CLASS_ORDER, cnt_color)
    table("num", NUM_ORDER, cnt_num)
    info = write_json(work / "cls" / "counts.json",
                      {"size": size, "pad": pad, "min_plate_w": min_w, "n_crop": n_obj,
                       "color": {"%s/%s" % k: v for k, v in sorted(cnt_color.items())},
                       "num": {"%s/%s" % k: v for k, v in sorted(cnt_num.items())}})
    print("  统计 -> %s" % info.relative_to(ROOT))
    return info


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.prepare",
                                 description="导出训练集并生成模型/分类头数据集")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="覆盖配置，如 --set data.limit=300")
    ap.add_argument("--skip-crops", action="store_true")
    ap.add_argument("--skip-yaml", action="store_true")
    args = ap.parse_args(argv)

    cfg, cfg_path = load_config(args.config)
    apply_overrides(cfg, args.set)
    print("配置: %s" % cfg_path.relative_to(ROOT))
    out_dir, dest, summary = export_dataset(cfg)
    if not args.skip_yaml:
        build_yamls(cfg)
    if not args.skip_crops:
        build_crops(cfg, out_dir, dest)
    print("\n下一步：\n  python -m training.train_pose --config %s\n  python -m training.train_cls  --config %s"
          % (args.config or "training/config.yaml", args.config or "training/config.yaml"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
