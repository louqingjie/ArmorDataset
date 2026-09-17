#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成"基准关键点图": 每张测试图上叠加评判基准四点(带编号) + 传统视觉校验四边形 + ROI 框。

输出: Test/figures/reference_kpt.png, Test/reference_kpt_table.csv
"""
import csv
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Test")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

ref = json.load(open(os.path.join(OUT, "reference_kpt.json")))
rois = {}
_roi_path = os.path.join(OUT, "roi_list.csv")
if os.path.exists(_roi_path):
    for row in csv.DictReader(open(_roi_path)):
        rois[row["image"]] = [float(v) for v in row["roi_box(x1,y1,x2,y2)"].strip("[]").split(",")]

C, R, TS, HEAD = 4, 3, 340, 50
grid = np.full((R * TS + 70, C * TS, 3), 250, np.uint8)
cv2.putText(grid, "评判基准关键点 (绿=基准四点, 黄=传统视觉校验, 灰=评判 ROI, 数字=角点序号)",
            (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
rows = []
for i, nm in enumerate(sorted(ref.keys())):
    im = cv2.imread(os.path.join(ROOT, "BaseLine", nm), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    q = np.asarray(ref[nm]["quad"], np.float32)
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    half = max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) * 0.9 + 14
    x0, x1 = int(max(0, cx - half)), int(min(im.shape[1], cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(im.shape[0], cy + half))
    crop = im[y0:y1, x0:x1].copy()
    s = min((TS - HEAD - 16) / crop.shape[1], (TS - HEAD - 16) / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    tile = np.full((TS, TS, 3), 250, np.uint8)
    oy, ox = HEAD + (TS - HEAD - 16 - crop.shape[0]) // 2, (TS - crop.shape[1]) // 2
    tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    tr = lambda p: (np.asarray(p, np.float32) - np.array([x0, y0], np.float32)) * s + np.array([ox, oy], np.float32)
    # ROI 框(灰)
    r = rois.get(nm)
    if r:
        rb = (np.asarray([float(v) for v in r], np.float32).reshape(2, 2) - np.array([x0, y0], np.float32)) * s + np.array([ox, oy], np.float32)
        cv2.rectangle(tile, tuple(rb[0].astype(int)), tuple(rb[1].astype(int)), (150, 150, 150), 1, cv2.LINE_AA)
    if ref[nm]["cv_ok"] and ref[nm]["cv_quad"]:
        cv2.polylines(tile, [tr(ref[nm]["cv_quad"]).astype(np.int32)], True, (0, 215, 255), 2, cv2.LINE_AA)
    pts = tr(q).astype(np.int32)
    cv2.polylines(tile, [pts], True, (0, 180, 0), 2, cv2.LINE_AA)
    for k, p in enumerate(pts):
        cv2.circle(tile, tuple(p), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(tile, str(k), tuple(p + np.array([7, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    src = "传统视觉校验通过" if ref[nm]["cv_ok"] else "共识基准(视觉不可用)"
    cv2.putText(tile, "%s  [%s]" % (nm, src), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(tile, "共识:%d模型   CV-IoU:%.2f   基准尺寸:%.0fx%.0fpx" % (
        ref[nm]["n_models"], ref[nm]["iou_cv_vs_cons"], np.ptp(q[:, 0]), np.ptp(q[:, 1])),
        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1, cv2.LINE_AA)
    rr, cc = divmod(i, C)
    grid[70 + rr * TS:70 + (rr + 1) * TS, cc * TS:(cc + 1) * TS] = tile
    rows.append([nm] + ["(%d,%d)" % (round(p[0]), round(p[1])) for p in q] +
                [src, ref[nm]["iou_cv_vs_cons"], ref[nm]["n_models"]])
cv2.imwrite(os.path.join(FIG, "reference_kpt.png"), grid)
with open(os.path.join(OUT, "reference_kpt_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "p0左上", "p1左下", "p2右下", "p3右上", "基准来源", "CV与共识IoU", "共识模型数"])
    w.writerows(rows)
print("[save] figures/reference_kpt.png", grid.shape, "/ reference_kpt_table.csv")
