#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 Test/detections.json + Test/reference_kpt.json 评估各模型。

基准 = sp_vision_25 角点精修结果（见 make_reference.py）；粗框(18 模型共识)用于衡量离群度。
指标: 检出(hit, 与基准四边形 IoU>=0.3) / 四边形 IoU / 角点平均·最大误差(px 与 %板宽)
      / 颜色·编号正确率 / 置信度
输出: Test/evaluation.csv, Test/evaluation_summary.csv, Test/figures/eval_corner.png
"""
import csv
import json
import os

import cv2
import numpy as np

import make_table as MT

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Test")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

TRUTH = {"B2.png": ("B", "2"), "B3.jpg": ("B", "3"), "B4.jpg": ("B", "4"), "B7.jpg": ("B", "7"),
         "BB.jpg": ("B", "B"), "BO.jpg": ("B", "O"), "R2.jpg": ("R", "2"), "R3.png": ("R", "3"),
         "R4.png": ("R", "4"), "R7.jpg": ("R", "7"), "RB.jpg": ("R", "B")}


def poly_iou(q1, q2):
    q1 = cv2.convexHull(np.asarray(q1, np.float32))
    q2 = cv2.convexHull(np.asarray(q2, np.float32))
    a1, a2 = cv2.contourArea(q1), cv2.contourArea(q2)
    inter, _ = cv2.intersectConvexConvex(q1, q2)
    u = a1 + a2 - inter
    return float(inter / u) if u > 0 else 0.0


def symmetries(q):
    q = np.asarray(q, np.float32)
    out = []
    for rev in (False, True):
        base = q[::-1] if rev else q
        for k in range(4):
            out.append(np.roll(base, k, axis=0))
    return out


def align_err(q, ref):
    ref = np.asarray(ref, np.float32)
    best = None
    for c in symmetries(q):
        d = np.linalg.norm(c - ref, axis=1)
        if best is None or d.mean() < best[0]:
            best = (float(d.mean()), float(d.max()))
    return best


def plate_width(ref, vertical=False):
    ref = np.asarray(ref, np.float32)
    if vertical:   # 灯条长度方向(左下-左上 / 右下-右上)
        return float(np.mean([np.linalg.norm(ref[1] - ref[0]), np.linalg.norm(ref[2] - ref[3])]))
    return float(np.mean([np.linalg.norm(ref[3] - ref[0]), np.linalg.norm(ref[2] - ref[1])]))


def main():
    det = json.load(open(os.path.join(OUT, "detections.json")))["det"]
    ref = json.load(open(os.path.join(OUT, "reference_kpt.json")))
    names = list(TRUTH.keys())
    rows = []
    for model in det:
        num_names = MT.SKD_NUM_NAMES if det[model][names[0]]["kind"] == "skd" else MT.NUM_NAMES
        for nm in names:
            rec = det[model][nm]
            rq = np.asarray(ref[nm]["quad"], np.float32)
            cq = np.asarray(ref[nm]["consensus"], np.float32)
            tc, tn = TRUTH[nm]
            best = None
            for d in rec["dets"]:
                q = np.asarray(d["pts"], np.float32)
                io = poly_iou(q, rq)
                if best is None or io > best[0]:
                    best = (io, d, q)
            if best is None or best[0] < 0.3:
                rows.append([model, nm, rec["kind"], 0, "", "", "", "", "", "", "", ""])
                continue
            io, d, q = best
            e_ref, e_ref_max = align_err(q, rq)
            e_cons, _ = align_err(q, cq)
            pw = plate_width(rq)
            rows.append([model, nm, rec["kind"], 1, round(io, 3), round(e_ref, 2), round(e_ref_max, 2),
                         round(100 * e_ref / pw, 2), round(e_cons, 2),
                         int(MT.COLOR_NAMES.get(d["color"], "?") == tc), int(num_names.get(d["num"], "?") == tn),
                         round(d["score"], 3)])
    with open(os.path.join(OUT, "evaluation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "kind", "hit", "quad_iou_ref", "corner_err_ref_px", "corner_err_ref_max_px",
                    "corner_err_ref_%plate", "corner_err_vs_coarse_px", "color_ok", "num_ok", "score"])
        w.writerows(rows)

    summary = []
    for model in det:
        rs = [r for r in rows if r[0] == model]
        h = [r for r in rs if r[3] == 1]
        m = lambda i: float(np.mean([r[i] for r in h])) if h else 0.0
        summary.append({"model": model, "kind": rs[0][2], "hit": len(h), "n": len(rs),
                        "iou_ref": m(4), "err_ref_px": m(5), "err_ref_max_px": m(6), "err_ref_pct": m(7),
                        "err_vs_coarse_px": m(8), "color_acc": m(9), "num_acc": m(10), "score": m(11)})
    with open(os.path.join(OUT, "evaluation_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    print("%-26s hit   IoU   角点误差(px) 最大(px)  %%板宽  粗框偏差(px)  色   号   score" % "model")
    for s in sorted(summary, key=lambda z: (-z["hit"], z["err_ref_px"])):
        print("%-26s %2d/%d  %.3f  %8.2f %8.2f  %5.1f%%  %8.2f   %.0f%% %.0f%%  %.2f" % (
            s["model"], s["hit"], s["n"], s["iou_ref"], s["err_ref_px"], s["err_ref_max_px"],
            s["err_ref_pct"], s["err_vs_coarse_px"], 100 * s["color_acc"], 100 * s["num_acc"], s["score"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    labels = [s["model"] for s in summary]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(1, 3, figsize=(19, 7))
    ax[0].barh(y, [s["err_ref_px"] for s in summary], color="#3b7dd8")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel("平均角点误差 (px, 相对 sp_vision_25 精修基准)")
    ax[0].set_title("角点精度（越小越好）"); ax[0].grid(axis="x", alpha=.3)
    ax[1].barh(y, [s["iou_ref"] for s in summary], color="#d86a3b")
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel("平均四边形 IoU（vs 精修基准）")
    ax[1].set_title("四边形重合度（越大越好）"); ax[1].grid(axis="x", alpha=.3)
    ax[2].barh(y, [s["hit"] for s in summary], color="#4aa96c")
    ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].invert_yaxis()
    ax[2].set_xlabel("命中张数 / 11"); ax[2].set_xlim(0, 11)
    ax[2].set_title("检出情况（IoU>=0.3 记为命中）"); ax[2].grid(axis="x", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "eval_corner.png"), dpi=130)
    plt.close(fig)
    print("[save] evaluation.csv / evaluation_summary.csv / figures/eval_corner.png")


if __name__ == "__main__":
    main()
