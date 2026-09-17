# -*- coding: utf-8 -*-
"""统计报告：读取 meta/*.json 与 progress.jsonl，产出 stats.json + report.md。

关注指标：精修接受率、双教师一致率、粗框→精修角点位移、类别/板宽分布、
各阶段拒绝原因直方图、耗时与失败图清单——用于判断试点 1000 张的标签质量。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import config as C


def _pct(values, qs=(50, 90, 99)):
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {("p%d" % q): round(float(np.percentile(a, q)), 3) for q in qs} | \
           {"mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3)}


def collect(out_dir):
    out_dir = Path(out_dir)
    meta_dir = out_dir / "meta"
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.rglob("*.json"))]

    flag_cnt, reason_cnt, src_cnt = Counter(), Counter(), Counter()
    color_cnt, num_cnt = Counter(), Counter()
    plate_w, d_px, d_pct, d_sec, kpt_pct = [], [], [], [], []
    n_obj = n_review = 0
    imgs_no_det = 0
    for m in metas:
        if not m["objects"]:
            imgs_no_det += 1
        for o in m["objects"]:
            n_obj += 1
            n_review += int(bool(o.get("review")))
            for f in o.get("flags", []):
                flag_cnt[f] += 1
            if o.get("reason"):
                reason_cnt[o["reason"]] += 1
            src_cnt[o.get("source", "?")] += 1
            color_cnt[o.get("color_name", "?")] += 1
            num_cnt[o.get("num_name", "?")] += 1
            if o.get("plate_w"):
                plate_w.append(o["plate_w"])
            r = o.get("refine") or {}
            if r.get("accepted") and r.get("d_coarse_to_refine_px") is not None:
                d_px.append(r["d_coarse_to_refine_px"])
                d_pct.append(r["d_coarse_to_refine_pct"])
            if o.get("d_secondary_px") is not None:
                d_sec.append(o["d_secondary_px"])
            if o.get("kpt_diff_pct") is not None:
                kpt_pct.append(o["kpt_diff_pct"])

    both = flag_cnt["agree"] + flag_cnt["conflict"]
    stats = {
        "config": {"expand": (metas[0]["config"]["expand"] if metas else C.ROI_EXPAND),
                   "agree_iou": (metas[0]["config"]["agree_iou"] if metas else C.AGREE_IOU),
                   "agree_kpt_pct": (metas[0]["config"]["agree_kpt_pct"] if metas else C.AGREE_KPT_PCT),
                   "min_refine_plate_w": (metas[0]["config"].get("min_refine_plate_w",
                                                                 C.MIN_REFINE_PLATE_W) if metas else C.MIN_REFINE_PLATE_W),
                   "class_mode": (metas[0].get("class_mode", C.CLASS_MODE) if metas else C.CLASS_MODE),
                   "teachers": (metas[0]["teachers"] if metas else [t["label"] for t in C.TEACHERS])},
        "images": {"total": len(metas), "no_detection": imgs_no_det,
                   "with_labels": len(metas) - imgs_no_det},
        "objects": {"total": n_obj, "review": n_review,
                    "review_rate": round(n_review / n_obj, 4) if n_obj else 0.0,
                    "per_image": round(n_obj / len(metas), 3) if metas else 0.0},
        "source": dict(src_cnt),
        "refine_accept_rate": round(src_cnt["refine"] / max(1, src_cnt["refine"] + src_cnt["teacher"]), 4),
        "consistency": {
            "agree": flag_cnt["agree"], "conflict": flag_cnt["conflict"],
            "primary_only": flag_cnt["primary_only"], "secondary_only": flag_cnt["secondary_only"],
            "agree_rate_over_paired": round(flag_cnt["agree"] / max(1, both), 4),
            "both_teachers_present": both,
        },
        "flags": dict(flag_cnt),
        "reasons": dict(reason_cnt),
        "color_dist": dict(color_cnt),
        "num_dist": dict(num_cnt),
        "plate_w_px": _pct(plate_w),
        "refine_shift_px": _pct(d_px),
        "refine_shift_pct_plate": _pct(d_pct),
        "secondary_vs_primary_px": _pct(d_sec),
        "kpt_diff_pct_plate": _pct(kpt_pct),
        "timing": _timing(out_dir),
    }
    return stats


def _timing(out_dir):
    path = Path(out_dir) / "progress.jsonl"
    if not path.exists():
        return {}
    ms, status = [], Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        status[rec.get("status", "?")] += 1
        if rec.get("ms"):
            ms.append(rec["ms"])
    out = {"records": sum(status.values()), "status": dict(status), "ms_per_image": _pct(ms)}
    if ms:
        out["total_minutes"] = round(sum(ms) / 60000.0, 2)
    return out


def to_markdown(stats):
    s = stats
    L = []
    L.append("# 自动标注质量报告\n")
    L.append("- 教师: %s" % " / ".join(s["config"]["teachers"]))
    L.append("- 一致性门限: IoU≥%.2f 且角点差≤%.0f%% 板宽且颜色编号一致" %
             (s["config"]["agree_iou"], 100 * s["config"]["agree_kpt_pct"]))
    L.append("- ROI 放大: %.2f×，标签类别模式: %s\n" % (s["config"]["expand"], s["config"]["class_mode"]))

    im, ob = s["images"], s["objects"]
    L.append("## 1. 规模\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 处理图片 | %d（无检出 %d） |" % (im["total"], im["no_detection"]))
    L.append("| 目标总数 | %d（每图 %.2f 个） |" % (ob["total"], ob["per_image"]))
    L.append("| 待人工复核 | %d（%.1f%%） |" % (ob["review"], 100 * ob["review_rate"]))
    L.append("| 精修接受率 | %.1f%%（refine=%d / teacher=%d） |" %
             (100 * s["refine_accept_rate"], s["source"].get("refine", 0), s["source"].get("teacher", 0)))
    L.append("| 小目标跳过精修(板宽<%.0fpx) | %d |" %
             (s["config"].get("min_refine_plate_w", 0), s["flags"].get("refine_skipped", 0)))
    cons = s["consistency"]
    L.append("| 双教师一致率 | %.1f%%（agree=%d / conflict=%d，配对 %d） |" %
             (100 * cons["agree_rate_over_paired"], cons["agree"], cons["conflict"], cons["both_teachers_present"]))
    L.append("| 单侧检出 | 仅主 %d / 仅副 %d |\n" % (cons["primary_only"], cons["secondary_only"]))

    L.append("## 2. 角点位移与板宽\n")
    for key, name, unit in (("refine_shift_px", "粗框→精修角点位移", "px"),
                            ("refine_shift_pct_plate", "粗框→精修角点位移", "%板宽"),
                            ("secondary_vs_primary_px", "两教师角点互差", "px"),
                            ("plate_w_px", "板宽", "px")):
        d = s.get(key) or {}
        if d:
            L.append("- %s(%s): mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f" %
                     (name, unit, d["mean"], d["p50"], d["p90"], d["p99"], d["max"]))
    L.append("")

    L.append("## 3. 拒绝/失败原因\n")
    if s["reasons"]:
        L.append("| 原因 | 次数 |")
        L.append("|---|---|")
        for k, v in sorted(s["reasons"].items(), key=lambda t: -t[1]):
            L.append("| %s | %d |" % (k, v))
    else:
        L.append("无")
    L.append("")

    L.append("## 4. 类别分布\n")
    L.append("| 颜色 | 数量 | | 编号 | 数量 |")
    L.append("|---|---|---|---|---|")
    cols = sorted(s["color_dist"].items(), key=lambda t: -t[1])
    nums = sorted(s["num_dist"].items(), key=lambda t: -t[1])
    for i in range(max(len(cols), len(nums))):
        a = cols[i] if i < len(cols) else ("", "")
        b = nums[i] if i < len(nums) else ("", "")
        L.append("| %s | %s | | %s | %s |" % (a[0], a[1], b[0], b[1]))
    L.append("")

    t = s.get("timing") or {}
    if t:
        L.append("## 5. 耗时\n")
        L.append("- 记录 %d 条，状态: %s" % (t["records"], json.dumps(t["status"], ensure_ascii=False)))
        if t.get("ms_per_image"):
            L.append("- 单图耗时(ms): mean=%.0f p90=%.0f max=%.0f，累计 %.1f 分钟" %
                     (t["ms_per_image"]["mean"], t["ms_per_image"]["p90"], t["ms_per_image"]["max"],
                      t.get("total_minutes", 0)))
        L.append("")
    return "\n".join(L)


def write(out_dir, stats=None, markdown=True):
    out_dir = Path(out_dir)
    stats = stats or collect(out_dir)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    if markdown:
        (out_dir / "report.md").write_text(to_markdown(stats), encoding="utf-8")
    return stats
