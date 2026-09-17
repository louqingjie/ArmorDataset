# -*- coding: utf-8 -*-
"""训练集导出：过滤 → train/val 划分 → 拷贝/软链 → 标签重写 → data.yaml。

产出目录（可直接交给 Ultralytics 训练）:
    <dest>/images/{train,val}/*.jpg|png
    <dest>/labels/{train,val}/*.txt          (YOLO-pose，kpt_shape=[4,2])
    <dest>/data.yaml
    <dest>/export_report.json

过滤条件:
    all            全部有目标的图
    high           仅高可信（flags 含 agree）
    exclude_review 排除仍有待复核目标的图
    reviewed       仅人工确认过的图（flags 含 reviewed）
对象级过滤 `object_filter=drop_review` 时，仍待复核的目标会被剔除而保留同图其他目标。
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

import make_table as MT
from autolabel import config as C
from autolabel import labelio

from webui import api, edits

FILTERS = ("all", "high", "exclude_review", "reviewed")
OBJ_FILTERS = ("keep_all", "drop_review")
IMAGE_MODES = ("copy", "symlink", "none")
ORDER_MODES = ("balanced", "random", "key")

# 交错周期：按编号递增轮一遍颜色 —— R1, B1, P1, N1, R2, B2, … 然后回到 R1 循环往复。
# 编号顺序 1~5 → 哨兵(索引0) → 前哨站(6) → 基地(7) → 大基地(8)
CYCLE_NUM = (1, 2, 3, 4, 5, 0, 6, 7, 8)
CYCLE_COLOR = ("R", "B", "G", "N")          # 红 → 蓝 → 灰白(未点亮) → 其他（索引 2 实测非紫，见 color_audit.py）
CAND_CLASSES = 8        # 均衡取图：每步在最缺的若干个类别里挑图
PEEK = 6                # 每类最多试探 6 张候选图


def cycle_pairs():
    """交错周期里的类标签序列，如 ['R1','B1','P1','N1','R2',...]。"""
    return ["%s%s" % (c, MT.NUM_NAMES.get(n, str(n))) for n in CYCLE_NUM for c in CYCLE_COLOR]


def _flatten(key: str, suffix: str):
    return str(key).replace("/", "__") + suffix


def _class_names(mode):
    if mode == "merged":
        return {i: "%s%s" % (MT.COLOR_NAMES.get(i // 9, "?"), MT.NUM_NAMES.get(i % 9, "?"))
                for i in range(36)}
    return {0: "armor"}


def _yaml_text(dest: Path, counts, mode):
    names = _class_names(mode)
    lines = ["# 由 webui 自动生成 —— 装甲板四点关键点数据集",
             "path: %s" % dest,
             "train: images/train",
             "val: images/val",
             "kpt_shape: [4, 2]         # 4 个关键点 (左上/左下/右下/右上)，无可见性",
             "flip_idx: [3, 2, 1, 0]    # 水平翻转时的关键点对应关系",
             "names:"]
    lines += ["  %d: %s" % (k, v) for k, v in names.items()]
    lines.append("# 图片: train=%d val=%d 目标: train=%d val=%d" %
                 (counts["train"]["images"], counts["val"]["images"],
                  counts["train"]["objects"], counts["val"]["objects"]))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 选取与统计
# --------------------------------------------------------------------------- #
def _balanced_order(entries):
    """类别均衡排序：每步都从“当前样本数最少的类别”取一张未用过的图，并列时按交错周期顺序。

    * 单类图片的情形下，它正好表现为 R1 → B1 → P1 → N1 → R2 → B2 → … 的周期循环；
    * 一张图常含多个类别，被取走后会同时为这些类别计数（“顺带喂饱”）；
      下一轮就会优先补给其它类别，使**每个类别在导出集里的出现次数尽量相等**；
    * 无目标的背景图排在最后（仅当 include_background 时才会出现）。
    """
    from collections import Counter, deque
    from itertools import islice
    buckets = {}
    for e in entries:
        for pair in (e.get("pairs") or {}):
            buckets.setdefault(pair, []).append(e)
    for k in list(buckets):
        buckets[k] = deque(sorted(buckets[k], key=lambda x: x["key"]))
    cyc = cycle_pairs()
    cyc_set = set(cyc)
    active = [k for k in cyc if buckets.get(k)] + [k for k in sorted(buckets) if k not in cyc_set]
    remaining = {k: len(v) for k, v in buckets.items()}
    cnt, order, seen = Counter(), [], set()

    def pairs_of(e):
        return [p for p in (e.get("pairs") or {}) if p in remaining]

    while True:
        # 1) 找出当前最缺的若干类别（计数最小，并列按交错周期顺序）
        candidates = [k for k in active if remaining.get(k, 0) > 0]
        if not candidates:
            break
        min_cnt = min(cnt[k] for k in candidates)
        pool = [k for k in candidates if cnt[k] == min_cnt][:CAND_CLASSES]

        # 2) 在这些类别的候选图里，选“顺带引入的已计数最少”的一张：
        #    稀有类别（基地/大基地等）所在的图常同时含热门类别，这一步避免顺手把热门类喂饱
        best = None                                   # ((代价, 周期序, key), 类, 图)
        for ci, k in enumerate(pool):
            for cand in islice(buckets[k], PEEK):
                if cand["key"] in seen:
                    continue
                cost = sum(cnt[p] for p in pairs_of(cand))
                tie = (cost, ci, cand["key"])
                if best is None or tie < best[0]:
                    best = (tie, k, cand)
        if best is None:
            # 桶里只剩已被取用的图 → 清理并继续
            for k in pool:
                buckets[k] = deque([c for c in buckets[k] if c["key"] not in seen])
                remaining[k] = len(buckets[k])
            if not any(remaining.get(k, 0) > 0 for k in active):
                break
            continue

        _tie, k, picked = best
        seen.add(picked["key"])
        order.append(picked)
        for p in pairs_of(picked):
            cnt[p] += 1
            remaining[p] = remaining.get(p, 0) - 1

    order += [e for e in sorted(entries, key=lambda x: x["key"]) if e["key"] not in seen]
    return order


def _select(out_dir: Path, opts):
    idx = api.get_index(out_dir)
    flt = opts.get("filter", "all")
    if flt not in FILTERS:
        raise api.ApiError("未知过滤条件: %s" % flt)
    order = opts.get("order", "balanced")
    if order not in ORDER_MODES:
        raise api.ApiError("未知排序方式: %s" % order)
    include_bg = bool(opts.get("include_background"))
    include_dep = bool(opts.get("include_deprecated"))
    cand = []
    for e in idx.entries.values():
        if e["n_obj"] == 0 or (e.get("n_usable") or 0) == 0:
            # 无目标、或目标全部被废弃（如大装甲）→ 按背景图处理，默认不导出
            if include_bg and flt in ("all", "high"):
                cand.append(e)
            continue
        if flt == "high" and "agree" not in e["flags"]:
            continue
        if flt == "exclude_review" and e["n_review"]:
            continue
        if flt == "reviewed" and not e["reviewed"]:
            continue
        cand.append(e)

    # 废弃数据不参与训练：默认整体排除（include_deprecated=True 时才放行）
    n_dep = sum(1 for e in cand if e.get("deprecated"))
    kept = [e for e in cand if include_dep or not e.get("deprecated")]
    info = {"n_deprecated_excluded": n_dep, "include_deprecated": include_dep}
    kept.sort(key=lambda e: e["key"])
    if order == "balanced":
        kept = _balanced_order(kept)
    elif order == "random":
        rng = random.Random(int(opts.get("seed") or 0))
        kept = list(kept)
        rng.shuffle(kept)
    limit = int(opts.get("limit") or 0)
    if limit > 0:
        kept = kept[:limit]          # 均衡模式下直接取前 N 张 = 按周期轮出来的均衡子集
    return idx, kept, info


def _balance_brief(cls_img):
    """图片级类别计数的均衡度摘要：各类别出现次数的 min/中位/max 与离散度。"""
    vals = sorted(cls_img.values())
    if not vals:
        return {}
    import statistics
    return {"classes": len(vals), "min": vals[0], "median": int(statistics.median(vals)),
            "max": vals[-1],
            "spread": round(vals[-1] / max(1, vals[0]), 2)}


def describe_order(out_dir, res, preview=12):
    """把预检结果打印成可读文本（含前 N 张的类别，用于核对交错顺序）。"""
    lines = ["排序方式: %s  |  交错周期: %s …" % (res["order"], " → ".join(res.get("cycle") or [])),
             "命中图片 %d 张 / 目标 %d 个（覆盖 %d 类）" %
             (res["n_images"], res["n_objects"], res.get("n_classes_hit", 0)),
             "已废弃被排除: %d 张%s" % (res.get("n_deprecated_excluded", 0),
                                  "（未排除）" if res.get("include_deprecated") else ""),
             "类别分布: %s" % json.dumps(res.get("class_dist") or {}, ensure_ascii=False),
             "train/val 预计: %d / %d" % (res["n_train"], res["n_val"]),
             "前 %d 张导出顺序:" % preview]
    for k in (res.get("order_preview") or [])[:preview]:
        try:
            meta = api.read_meta(out_dir, k)
            pairs = ["%s%s" % (o.get("color_name") or "?", o.get("num_name") or "?")
                     for o in (meta.get("objects") or [])]
        except Exception:
            pairs = ["?"]
        lines.append("   %-18s %s" % (k, ", ".join(pairs)))
    return "\n".join(lines)


def preflight(payload):
    out = api.safe_path(payload.get("out") or api.rel_to_root(C.DEFAULT_OUT), must_exist=True)
    opts = dict(payload)
    idx, kept, sel_info = _select(out, opts)
    obj_filter = opts.get("object_filter", "keep_all")
    if obj_filter not in OBJ_FILTERS:
        raise api.ApiError("未知对象过滤: %s" % obj_filter)

    n_obj, n_review_obj, n_bg, n_dep_obj = 0, 0, 0, 0
    cls_cnt = {}
    cls_img = {}                      # 图片级计数（一张图对每个类别最多计 1 次）
    for e in kept:
        for p in (e.get("pairs") or {}):
            cls_img[p] = cls_img.get(p, 0) + 1
    size_bytes = 0
    mode = opts.get("class_mode") or C.CLASS_MODE
    for e in kept:
        meta = api.read_meta(out, e["key"])
        h, w = meta["size"]
        objs = meta.get("objects") or []
        usable = []
        for o in objs:
            if o.get("deprecated"):                 # 目标级废弃：标注保留但不参与训练
                n_dep_obj += 1
                continue
            if obj_filter == "drop_review" and o.get("review"):
                n_review_obj += 1
                continue
            usable.append(o)
        if not usable:
            n_bg += 1
        n_obj += len(usable)
        for o in usable:
            cid = labelio.class_id(edits.to_internal(o, w, h), mode)
            cls_cnt[str(cid)] = cls_cnt.get(str(cid), 0) + 1
        try:
            p = api.resolve_image_path(meta, out)
            size_bytes += p.stat().st_size
        except api.ApiError:
            pass

    val_ratio = float(opts.get("val_ratio", 0.1) or 0.0)
    n_val = int(round(len(kept) * max(0.0, min(0.9, val_ratio))))
    class_names = _class_names(mode)
    return {"ok": True, "out": api.rel_to_root(out), "dest": payload.get("dest"),
            "filter": opts.get("filter", "all"), "object_filter": obj_filter,
            "order": opts.get("order", "balanced"),
            "order_preview": [e["key"] for e in kept[:12]],
            "cycle": cycle_pairs()[:12],
            "n_classes_hit": len({p for e in kept for p in (e.get("pairs") or {})}),
            "class_images": dict(sorted(cls_img.items(), key=lambda t: -t[1])),
            "balance": _balance_brief(cls_img),
            "class_mode": mode, "n_images": len(kept), "n_objects": n_obj,
            "n_deprecated_excluded": sel_info.get("n_deprecated_excluded", 0),
            "include_deprecated": sel_info.get("include_deprecated", False),
            "n_dropped_review_objects": n_review_obj,
            "n_deprecated_objects_excluded": n_dep_obj,
            "n_images_without_usable_obj": n_bg,
            "n_train": len(kept) - n_val, "n_val": n_val,
            "class_dist": {class_names.get(int(k), k): v for k, v in
                           sorted(cls_cnt.items(), key=lambda t: -t[1])},
            "est_copy_mb": round(size_bytes / 1e6, 1),
            "image_mode": opts.get("image_mode", "copy"),
            "sample_keys": [e["key"] for e in kept[:12]]}


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def run_export(payload):
    out = api.safe_path(payload.get("out") or api.rel_to_root(C.DEFAULT_OUT), must_exist=True)
    dest_raw = payload.get("dest")
    if not dest_raw:
        raise api.ApiError("请指定导出目录 dest")
    dest = api.safe_path(dest_raw, must_exist=False)
    if dest.exists() and any(dest.iterdir()) and not payload.get("overwrite"):
        raise api.ApiError("导出目录非空，需勾选覆盖: %s" % api.rel_to_root(dest), 409)
    task = api.submit("export", _export_task, out, dest, dict(payload))
    return {"ok": True, "task": dict(task), "dest": api.rel_to_root(dest)}


def _export_task(report, out_dir: Path, dest: Path, opts):
    idx, kept, sel_info = _select(out_dir, opts)
    mode = opts.get("class_mode") or C.CLASS_MODE
    obj_filter = opts.get("object_filter", "keep_all")
    image_mode = opts.get("image_mode", "copy")
    if image_mode not in IMAGE_MODES:
        raise api.ApiError("未知图片方式: %s" % image_mode)
    val_ratio = max(0.0, min(0.9, float(opts.get("val_ratio", 0.1) or 0.0)))
    order = opts.get("order", "balanced")
    if order == "balanced" and val_ratio > 0:
        # 在交错序列上等距抽 val（而不是随机抽样）：train/val 两侧的类别分布同样均衡
        step = max(2, int(round(1.0 / val_ratio)))
        val_keys = {e["key"] for i, e in enumerate(kept) if i % step == 0}
    else:
        rng = random.Random(int(opts.get("seed") or 0))
        shuffled = list(kept)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_ratio))
        val_keys = {e["key"] for e in shuffled[:n_val]}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": {"images": 0, "objects": 0}, "val": {"images": 0, "objects": 0}}
    n_dep_obj = 0
    skipped, errors = [], []
    total = max(1, len(kept))
    for i, e in enumerate(kept, 1):
        split = "val" if e["key"] in val_keys else "train"
        try:
            meta = api.read_meta(out_dir, e["key"])
            h, w = meta["size"]
            lines = []
            for o in meta.get("objects") or []:
                if o.get("deprecated"):                  # 目标级废弃：不进标签
                    n_dep_obj += 1
                    continue
                if obj_filter == "drop_review" and o.get("review"):
                    continue
                lines.append(labelio.yolo_line(edits.to_internal(o, w, h), w, h, mode))
            if not lines and not opts.get("include_background"):
                skipped.append({"key": e["key"], "reason": "无可用目标"})
                continue
            img_src = api.resolve_image_path(meta, out_dir)
            name = _flatten(e["key"], img_src.suffix.lower())
            img_dst = dest / "images" / split / name
            lab_dst = dest / "labels" / split / (Path(name).stem + ".txt")
            if image_mode == "copy":
                shutil.copy2(img_src, img_dst)
            elif image_mode == "symlink":
                if img_dst.exists():
                    img_dst.unlink()
                img_dst.symlink_to(img_src)
            if lines:
                labelio._atomic_write(lab_dst, "\n".join(lines) + "\n")
            counts[split]["images"] += 1
            counts[split]["objects"] += len(lines)
        except Exception as exc:
            errors.append({"key": e["key"], "error": "%s: %s" % (type(exc).__name__, exc)})
        if i % 50 == 0 or i == total:
            report(i / total, "%s/%s 张" % (i, total))

    yaml_text = _yaml_text(dest, counts, mode)
    labelio._atomic_write(dest / "data.yaml", yaml_text)
    summary = {"ok": True, "dest": api.rel_to_root(dest), "class_mode": mode,
               "filter": opts.get("filter", "all"), "object_filter": obj_filter,
               "order": order, "cycle": cycle_pairs()[:12],
               "n_deprecated_excluded": sel_info.get("n_deprecated_excluded", 0),
               "n_deprecated_objects_excluded": n_dep_obj,
               "image_mode": image_mode, "counts": counts,
               "skipped": skipped[:200], "n_skipped": len(skipped),
               "errors": errors[:200], "n_errors": len(errors),
               "data_yaml": str(dest / "data.yaml"), "finished": round(time.time(), 3),
               "kpt_shape": [4, 2], "class_names": _class_names(mode)}
    labelio._atomic_write(dest / "export_report.json", json.dumps(summary, ensure_ascii=False, indent=1))
    report(1.0, "导出完成")
    return summary


# --------------------------------------------------------------------------- #
# 命令行入口（无需网页：可用于预检/批量导出/CI）
# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m webui.dataset",
                                 description="训练集导出 CLI（默认只预检，加 --run 才真正导出）")
    ap.add_argument("--out", default=api.rel_to_root(C.DEFAULT_OUT), help="源输出目录（含 meta/）")
    ap.add_argument("--dest", default="TrainSet/armor_pose", help="导出目录")
    ap.add_argument("--filter", default="all", choices=list(FILTERS))
    ap.add_argument("--object-filter", default="keep_all", choices=list(OBJ_FILTERS))
    ap.add_argument("--order", default="balanced", choices=list(ORDER_MODES),
                    help="balanced=按 R1→B1→R2… 交错均衡；random=随机；key=按文件名")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--class-mode", default=C.CLASS_MODE, choices=("single", "merged"))
    ap.add_argument("--image-mode", default="symlink", choices=list(IMAGE_MODES))
    ap.add_argument("--limit", type=int, default=0, help="0=不限（均衡模式下取前 N 张）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-background", action="store_true")
    ap.add_argument("--include-deprecated", action="store_true",
                    help="把已废弃的图片也纳入导出（默认排除，废弃数据不参与训练）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--run", action="store_true", help="真正执行导出（默认只预检）")
    ap.add_argument("--sync-deprecated", action="store_true",
                    help="把 out/deprecated.json 的废弃标记重新镜像进 meta 后退出")
    args = ap.parse_args(argv)

    if args.sync_deprecated:
        from webui import deprecate as dep_mod
        print("同步废弃标记:", dep_mod.sync_mirror(api.safe_path(args.out)))
        return 0

    payload = {"out": args.out, "dest": args.dest, "filter": args.filter,
               "object_filter": args.object_filter, "order": args.order,
               "val_ratio": args.val_ratio, "class_mode": args.class_mode,
               "image_mode": args.image_mode, "limit": args.limit, "seed": args.seed,
               "include_background": args.include_background,
               "include_deprecated": args.include_deprecated, "overwrite": args.overwrite}
    out_dir = api.safe_path(args.out)
    res = preflight(payload)
    print(describe_order(out_dir, res))
    if not args.run:
        print("\n（预检完成；加 --run 执行导出）")
        return 0
    dest = api.safe_path(args.dest, must_exist=False)
    print("\n开始导出 → %s" % api.rel_to_root(dest))
    summary = _export_task(lambda p, m="": None, out_dir, dest, payload)
    print("完成：train=%s val=%s 跳过=%d 失败=%d\ndata.yaml: %s" %
          (summary["counts"]["train"], summary["counts"]["val"],
           summary["n_skipped"], summary["n_errors"], summary["data_yaml"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
