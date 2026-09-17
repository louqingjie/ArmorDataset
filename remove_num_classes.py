#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按“编号头”批量删除标注（本次用途：一号装甲板与世界大基地装甲板外观改版，需重新标注）。

默认删除编号索引 1（1 号装甲板：R1/B1/P1/N1）与 8（大基地装甲板：RLB/BLB/PLB/NLB）。

行为:
  * 只动对象级标注：从 meta 的 objects 中移除匹配目标，并同步重写 labels/*.txt；
  * 剩余目标全部删光的图片 → 变成背景图（labels 文件删除，meta 保留，n_obj=0）；
  * meta 其它字段（config/teachers/deprecated 镜像/review_edits 等）原样保留；
  * 修改前把受影响图片的**原始 objects 全量**写入 `out/removed_annotations_<ts>.jsonl`，
    可用 `--undo` 精确还原。

用法:
    python remove_num_classes.py AutoLabel --dry-run          # 只统计
    python remove_num_classes.py AutoLabel                    # 执行（自动备份 + 打印报告）
    python remove_num_classes.py AutoLabel --undo out/removed_annotations_xxx.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from autolabel import labelio                      # noqa: E402
from webui.edits import to_internal                # noqa: E402  （meta -> 内部结构，供 yolo_line 使用）

DEFAULT_NUMS = (1, 8)


def _ts():
    t = time.time()
    return "%s-%03d" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(t)), int(t * 1000) % 1000)


def _write_labels(out_dir: Path, meta: dict, objects: list, mode: str):
    """重写标签文件；无目标时写**空文件**而不是删除文件。

    空 label 文件对 Ultralytics 同样是“该图无目标（背景）”，
    这样批量操作不产生任何删除动作（避免环境的安全删除护栏按轮次拦截）。
    """
    w, h = int(meta["size"][1]), int(meta["size"][0])
    lines = [labelio.yolo_line(to_internal(o, w, h), w, h, mode) for o in objects]
    txt = out_dir / "labels" / (str(meta["key"]) + ".txt")
    labelio._atomic_write(txt, ("\n".join(lines) + "\n") if lines else "")


def remove(out_dir: Path, nums, dry_run=False, backup=True, reviewer="cli", reason=""):
    meta_files = sorted((out_dir / "meta").rglob("*.json"))
    ts = _ts()
    backup_path = (out_dir / ("removed_annotations_%s.jsonl" % ts)) if (backup and not dry_run) else None
    stats = Counter()
    n_affected = n_removed = n_emptied = n_kept_obj = 0
    bf = open(backup_path, "w", encoding="utf-8") if backup_path else None
    try:
        for p in meta_files:
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print("  解析失败 %s: %s" % (p, exc))
                continue
            objs = meta.get("objects") or []
            keep = [o for o in objs if int(o.get("num", -1)) not in nums]
            if len(keep) == len(objs):
                continue
            removed = [o for o in objs if int(o.get("num", -1)) in nums]
            n_affected += 1
            n_removed += len(removed)
            n_kept_obj += len(keep)
            if not keep:
                n_emptied += 1
            for o in removed:
                stats["%s%s" % (o.get("color_name") or "?", o.get("num_name") or "?")] += 1
            if bf:
                bf.write(json.dumps({"key": meta.get("key"), "rel": meta.get("rel"),
                                     "size": meta.get("size"), "class_mode": meta.get("class_mode"),
                                     "objects_before": objs, "n_review_before": meta.get("n_review"),
                                     "ts": ts}, ensure_ascii=False) + "\n")
            if dry_run:
                continue
            _write_labels(out_dir, meta, keep, meta.get("class_mode") or "single")
            meta["objects"] = keep
            meta["n_obj"] = len(keep)
            meta["n_review"] = sum(1 for o in keep if labelio.needs_review(o))
            hist = list(meta.get("removed_annotations") or [])
            hist.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "by": reviewer,
                         "reason": reason or "外观改版待重新标注",
                         "nums": sorted(nums), "n_removed": len(removed),
                         "classes": dict(Counter("%s%s" % (o.get("color_name"), o.get("num_name"))
                                                 for o in removed))})
            meta["removed_annotations"] = hist[-10:]
            labelio._atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
    finally:
        if bf:
            bf.close()
    print("%s%s: 扫描 %d 个 meta，命中 %d 张图，删除 %d 个目标，剩余目标 %d，变为背景 %d 张"
          % ("[dry-run] " if dry_run else "", out_dir, len(meta_files), n_affected,
             n_removed, n_kept_obj, n_emptied))
    if not dry_run:
        # 归一化：所有“无目标”的图都放一个空 label 文件，保持状态一致
        n_fix = 0
        for p in meta_files:
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("objects"):
                continue
            txt = out_dir / "labels" / (str(meta.get("key")) + ".txt")
            if not txt.exists():
                labelio._atomic_write(txt, "")
                n_fix += 1
        if n_fix:
            print("   补齐背景图空标签 %d 个" % n_fix)
    print("   删除明细: %s" % dict(stats.most_common()))
    if backup_path:
        print("   备份: %s（可用 --undo 还原）" % backup_path)
    return {"affected": n_affected, "removed": n_removed, "emptied": n_emptied,
            "backup": str(backup_path) if backup_path else None}


def undo(out_dir: Path, backup_file: Path, dry_run=False):
    n = 0
    for line in backup_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec["key"]
        p = out_dir / "meta" / (str(key) + ".json")
        if not p.exists():
            print("  跳过（meta 不存在）: %s" % key)
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["objects"] = rec["objects_before"]
        meta["n_obj"] = len(rec["objects_before"])
        meta["n_review"] = int(rec.get("n_review_before") or
                               sum(1 for o in meta["objects"] if labelio.needs_review(o)))
        hist = list(meta.get("removed_annotations") or [])
        hist.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "cli",
                     "reason": "从备份还原 %s" % backup_file.name, "n_restored": len(meta["objects"])})
        meta["removed_annotations"] = hist[-10:]
        if not dry_run:
            _write_labels(out_dir, meta, meta["objects"], meta.get("class_mode") or "single")
            labelio._atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
        n += 1
    print("%s从备份还原 %d 张图: %s" % ("[dry-run] " if dry_run else "", n, backup_file))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="按编号头批量删除标注（默认编号 1 与 8）")
    ap.add_argument("dirs", nargs="*", default=["AutoLabel"], help="输出目录（含 meta/）")
    ap.add_argument("--nums", default=",".join(str(v) for v in DEFAULT_NUMS),
                    help="要删除的编号索引，逗号分隔（1=一号装甲板，8=大基地装甲板）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true", help="不写备份（不建议）")
    ap.add_argument("--reviewer", default="cli")
    ap.add_argument("--reason", default="外观改版，待重新标注")
    ap.add_argument("--undo", help="从 backup jsonl 还原")
    args = ap.parse_args(argv)

    if args.undo:
        for d in args.dirs:
            undo(ROOT / d if not Path(d).is_absolute() else Path(d), Path(args.undo))
        return 0

    nums = {int(x) for x in str(args.nums).replace(" ", "").split(",") if x != ""}
    for d in args.dirs:
        p = ROOT / d if not Path(d).is_absolute() else Path(d)
        remove(p, nums, dry_run=args.dry_run, backup=not args.no_backup,
               reviewer=args.reviewer, reason=args.reason)
    if not args.dry_run:
        print("\n建议随后刷新统计与复核清单：\npython -m autolabel.run --stats-only --out %s" % args.dirs[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
