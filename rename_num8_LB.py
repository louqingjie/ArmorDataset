#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把编号索引 8 的显示名从“未知(?)”批量改为“大基地装甲板(LB)”。

背景：数据侧确认编号头索引 8 = 大基地装甲板（RLB / BLB / …），此前流水线里显示为 “?”。
本脚本只改 meta 中的 `num_name` 字段（数值 `num=8`、`quad_final_px`、`flags` 等一律不动），
因此**无需重跑推理**；labels/*.txt 不受影响（single 模式类别恒为 0，merged 模式用 9 进制索引）。

用法:
    python rename_num8_LB.py AutoLabel                 # 就地改名
    python rename_num8_LB.py AutoLabel --dry-run       # 只统计不写盘
    python rename_num8_LB.py AutoLabel AutoLabel_smoke AutoLabel_webui_test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from autolabel import labelio          # noqa: E402  （复用原子写）

NEW_NAME = "LB"
KEY = "num"


def rename_in(obj):
    """就地改名：对象自身 + primary/secondary 两个子结构里的 num_name。"""
    changed = 0
    if obj.get(KEY) == 8 and obj.get("num_name") != NEW_NAME:
        obj["num_name"] = NEW_NAME
        changed += 1
    for sub in ("primary", "secondary"):
        d = obj.get(sub)
        if isinstance(d, dict) and d.get(KEY) == 8 and d.get("num_name") != NEW_NAME:
            d["num_name"] = NEW_NAME
            changed += 1
    return changed


def process(out_dir: Path, dry_run=False):
    meta_dir = out_dir / "meta"
    if not meta_dir.is_dir():
        print("跳过（无 meta 目录）: %s" % out_dir)
        return {"files": 0, "objects": 0, "fields": 0}
    n_files = n_obj = n_field = 0
    colors = Counter()
    for p in sorted(meta_dir.rglob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print("  解析失败 %s: %s" % (p, exc))
            continue
        touched = 0
        for o in meta.get("objects") or []:
            if o.get(KEY) == 8:
                n_obj += 1
                colors[o.get("color_name") or "?"] += 1
                c = rename_in(o)
                touched += c
                n_field += c
        if touched:
            n_files += 1
            if not dry_run:
                labelio._atomic_write(p, json.dumps(meta, ensure_ascii=False, indent=1))
    print("%s%s: 命中文件 %d，目标 %d，改写字段 %d，颜色分布 %s"
          % ("[dry-run] " if dry_run else "", out_dir, n_files, n_obj, n_field, dict(colors)))
    return {"files": n_files, "objects": n_obj, "fields": n_field}


def main(argv=None):
    ap = argparse.ArgumentParser(description="编号索引 8 -> LB(大基地装甲板) 批量改名")
    ap.add_argument("dirs", nargs="*", default=["AutoLabel"], help="输出目录（含 meta/）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    args = ap.parse_args(argv)
    total = {"files": 0, "objects": 0, "fields": 0}
    for d in args.dirs:
        res = process((ROOT / d) if not Path(d).is_absolute() else Path(d), args.dry_run)
        for k in total:
            total[k] += res[k]
    if not args.dry_run and total["files"]:
        print("\n完成。建议随后刷新统计与复核清单：")
        print("  python -m autolabel.run --stats-only --out %s" % args.dirs[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
