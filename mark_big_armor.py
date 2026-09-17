#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把“大装甲板”目标标记为废弃（目标级废弃：标注保留、不参与训练导出）。

判定规则（三种模式，用 --mode 选）:
  geom    纯几何：板宽/灯条长 > --aspect（默认 3.0）。
          ⚠️ 实测：夜间/模糊/斜视会把步兵小装甲的该比值抬到 3.0~3.8，
          该模式会误伤约 1700 个步兵目标（已用拼图人工核对）。
  hybrid  几何 + 类别（推荐）：大装甲候选类（基地族 RB/BB/PB/NB 与 P7）用 --aspect 判定；
          其它类别要求更严的 --aspect-other（默认 4.5），基本不误伤。
  class   纯类别：只把 --classes 指定的类别标废弃（零几何误判）。

写盘内容:
  * meta 中每个被判定目标的 `deprecated` 置 true，并记录 dep_by/dep_time/dep_reason；
  * labels/*.txt 重写为「排除废弃目标」后的结果（= 可直接训练的样子）；
  * 变更前写入 `out/object_deprecations_<ts>.jsonl`（每图记录改动的 object 序号与原值），
    可用 `--undo <file>` 精确还原。

用法:
    python mark_big_armor.py AutoLabel --mode hybrid --dry-run
    python mark_big_armor.py AutoLabel --mode hybrid
    python mark_big_armor.py AutoLabel --undo AutoLabel/object_deprecations_xxx.jsonl
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
from webui.edits import to_internal                # noqa: E402

# 几何上确认为“大装甲”的类别（基地族 + 实测与基地同形的 P7）
BIG_CLASSES = ("RB", "BB", "PB", "NB", "P7")
DEFAULT_ASPECT = 3.0
DEFAULT_ASPECT_OTHER = 4.5


def _ts():
    t = time.time()
    return "%s-%03d" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(t)), int(t * 1000) % 1000)


def cls_of(o):
    return "%s%s" % (o.get("color_name") or "?", o.get("num_name") or "?")


def aspect_of(o):
    pw = float(o.get("plate_w") or 0)
    bl = float(o.get("bar_len") or 0)
    return (pw / bl) if (pw > 0 and bl > 0) else None


def is_big(o, mode, aspect, aspect_other, classes):
    c = cls_of(o)
    if mode == "class":
        return c in classes
    a = aspect_of(o)
    if a is None:
        return False
    if mode == "geom":
        return a > aspect
    # hybrid
    return a > (aspect if c in BIG_CLASSES else aspect_other)


def _recount(meta: dict):
    """重算 n_obj / n_dep_obj / n_review（废弃目标不进复核队列）。"""
    objs = meta.get("objects") or []
    meta["n_obj"] = len(objs)
    meta["n_dep_obj"] = sum(1 for o in objs if o.get("deprecated"))
    meta["n_review"] = sum(1 for o in objs if labelio.needs_review(o))
    return meta


def recount_all(out_dir: Path, dry_run=False):
    """对整库重算 n_obj/n_dep_obj/n_review（标记规则调整后的一次性维护）。"""
    n = 0
    for p in sorted((out_dir / "meta").rglob("*.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        before = (meta.get("n_obj"), meta.get("n_dep_obj"), meta.get("n_review"))
        _recount(meta)
        after = (meta.get("n_obj"), meta.get("n_dep_obj"), meta.get("n_review"))
        if before != after:
            n += 1
            if not dry_run:
                labelio._atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
    print("%s重算计数: %d 个 meta 有变化" % ("[dry-run] " if dry_run else "", n))
    return n


def _write_labels(out_dir: Path, meta: dict, objects: list):
    w, h = int(meta["size"][1]), int(meta["size"][0])
    mode = meta.get("class_mode") or "single"
    lines = [labelio.yolo_line(to_internal(o, w, h), w, h, mode)
             for o in objects if not o.get("deprecated")]
    txt = out_dir / "labels" / (str(meta["key"]) + ".txt")
    labelio._atomic_write(txt, ("\n".join(lines) + "\n") if lines else "")


def run(out_dir: Path, mode="hybrid", aspect=DEFAULT_ASPECT,
        aspect_other=DEFAULT_ASPECT_OTHER, classes=BIG_CLASSES,
        dry_run=False, backup=True, reviewer="cli", reason="大装甲板退出历史舞台"):
    metas = sorted((out_dir / "meta").rglob("*.json"))
    ts = _ts()
    backup_path = (out_dir / ("object_deprecations_%s.jsonl" % ts)) if (backup and not dry_run) else None
    bf = open(backup_path, "w", encoding="utf-8") if backup_path else None
    hit = Counter()
    n_hit = n_new = n_affect = 0
    try:
        for p in metas:
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print("  解析失败 %s: %s" % (p, exc))
                continue
            objs = meta.get("objects") or []
            changed = []
            for i, o in enumerate(objs):
                if is_big(o, mode, aspect, aspect_other, set(classes)):
                    n_hit += 1
                    hit[cls_of(o)] += 1
                    if not o.get("deprecated"):
                        o["deprecated"] = True
                        o["dep_by"] = reviewer
                        o["dep_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        o["dep_reason"] = reason
                        changed.append([i, False])          # 原值为 False/缺省
                        n_new += 1
            if not changed:
                continue
            n_affect += 1
            if bf:
                bf.write(json.dumps({"key": meta.get("key"), "changed": changed}, ensure_ascii=False) + "\n")
            if dry_run:
                continue
            hist = list(meta.get("object_deprecations") or [])
            hist.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "by": reviewer,
                         "mode": mode, "aspect": aspect, "aspect_other": aspect_other,
                         "reason": reason, "n_marked": len(changed)})
            meta["object_deprecations"] = hist[-10:]
            _recount(meta)
            _write_labels(out_dir, meta, objs)
            labelio._atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
    finally:
        if bf:
            bf.close()
    print("%s%s: mode=%s aspect=%s (其它类 %s) → 命中 %d 个目标，本次新标记 %d 个，涉及 %d 张图"
          % ("[dry-run] " if dry_run else "", out_dir, mode, aspect,
             aspect_other if mode == "hybrid" else "-", n_hit, n_new, n_affect))
    print("   命中类别: %s" % dict(hit.most_common()))
    if backup_path:
        print("   备份: %s（--undo 可还原）" % backup_path)
    return {"hit": n_hit, "new": n_new, "files": n_affect,
            "backup": str(backup_path) if backup_path else None}


def undo(out_dir: Path, backup_file: Path, dry_run=False, only_classes=None, skip_classes=None):
    """还原备份中的废弃标记；only_classes/skip_classes 可按类别筛选（用于只回退误判部分）。"""
    n_obj = n_file = 0
    only = set(c.strip() for c in (only_classes or []) if c.strip()) or None
    skip = set(c.strip() for c in (skip_classes or []) if c.strip())
    for line in backup_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p = out_dir / "meta" / (str(rec["key"]) + ".json")
        if not p.exists():
            print("  跳过（meta 不存在）: %s" % rec["key"])
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        objs = meta.get("objects") or []
        for idx, _prev in rec["changed"]:
            if not (0 <= idx < len(objs)):
                continue
            c = cls_of(objs[idx])
            if (only is not None and c not in only) or (c in skip):
                continue
            for k in ("deprecated", "dep_by", "dep_time", "dep_reason"):
                objs[idx].pop(k, None)
            n_obj += 1
        meta["n_dep_obj"] = sum(1 for o in objs if o.get("deprecated"))
        hist = list(meta.get("object_deprecations") or [])
        hist.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "cli",
                     "reason": "从备份还原 %s" % backup_file.name, "n_restored": len(rec["changed"])})
        meta["object_deprecations"] = hist[-10:]
        if not dry_run:
            _write_labels(out_dir, meta, objs)
            labelio._atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
        n_file += 1
    print("%s从备份还原 %d 张图 / %d 个目标: %s" % ("[dry-run] " if dry_run else "", n_file, n_obj, backup_file))
    return n_obj


def main(argv=None):
    ap = argparse.ArgumentParser(description="把大装甲目标标记为废弃（标注保留、不参与训练）")
    ap.add_argument("dirs", nargs="*", default=["AutoLabel"])
    ap.add_argument("--mode", default="hybrid", choices=("geom", "hybrid", "class"),
                    help="geom=纯几何 aspect；hybrid=基地族用 aspect、其它类用 aspect-other；class=纯类别")
    ap.add_argument("--aspect", type=float, default=DEFAULT_ASPECT, help="板宽/灯条长 阈值（默认 3.0）")
    ap.add_argument("--aspect-other", type=float, default=DEFAULT_ASPECT_OTHER,
                    help="hybrid 模式下非大装甲候选类别的阈值（默认 4.5）")
    ap.add_argument("--classes", default=",".join(BIG_CLASSES),
                    help="class 模式/大装甲候选类别，如 RB,BB,PB,NB,P7")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--reviewer", default="cli")
    ap.add_argument("--reason", default="大装甲板退出历史舞台")
    ap.add_argument("--undo", help="从备份 jsonl 还原")
    ap.add_argument("--undo-classes", default="",
                    help="只还原这些类别（逗号分隔，如 B3,R3,R4,B4,R7,B7,P3）；空=全部还原")
    ap.add_argument("--recount", action="store_true",
                    help="只重算 n_obj/n_dep_obj/n_review（标记规则调整后的一次性维护）")
    args = ap.parse_args(argv)

    if args.recount:
        for d in args.dirs:
            recount_all(ROOT / d if not Path(d).is_absolute() else Path(d))
        return 0

    if args.undo:
        only = [c for c in args.undo_classes.split(",") if c.strip()]
        for d in args.dirs:
            undo(ROOT / d if not Path(d).is_absolute() else Path(d), Path(args.undo),
                 only_classes=only)
        return 0

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    for d in args.dirs:
        p = ROOT / d if not Path(d).is_absolute() else Path(d)
        run(p, mode=args.mode, aspect=args.aspect, aspect_other=args.aspect_other,
            classes=classes, dry_run=args.dry_run, backup=not args.no_backup,
            reviewer=args.reviewer, reason=args.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
