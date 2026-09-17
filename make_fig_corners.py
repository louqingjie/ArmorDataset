#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""角点对比图: 各图装甲板放大 + 基准(传统视觉/共识) 与代表模型(rp_0526、szu2026、SKD) 的四点叠加。"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Test")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

SHOW = [("rp_0526_fp32", (0, 0, 255)), ("szu2026_infantry_fp32", (255, 128, 0)), ("shtech/SKD250526", (255, 0, 255))]

det = json.load(open(os.path.join(OUT, "detections.json")))["det"]
ref = json.load(open(os.path.join(OUT, "reference_kpt.json")))

C, R = 4, 3
TS = 260
grid = np.full((R * TS, C * TS, 3), 245, np.uint8)
for i, nm in enumerate(sorted(ref.keys())):
    im = cv2.imread(os.path.join(ROOT, "BaseLine", nm), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    q = np.asarray(ref[nm]["quad"], np.float32)
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    half = max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) * 0.85 + 10
    x0, x1 = int(max(0, cx - half)), int(min(im.shape[1], cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(im.shape[0], cy + half))
    crop = im[y0:y1, x0:x1].copy()
    s = min((TS - 24) / crop.shape[1], (TS - 24) / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    tile = np.full((TS, TS, 3), 30, np.uint8)
    oy, ox = 12 + (TS - 24 - crop.shape[0]) // 2, 12 + (TS - 24 - crop.shape[1]) // 2
    tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    tr = lambda p: (np.asarray(p, np.float32) - np.array([x0, y0], np.float32)) * s + np.array([ox, oy], np.float32)
    cv2.polylines(tile, [tr(ref[nm]["consensus"]).astype(np.int32)], True, (170, 170, 170), 1, cv2.LINE_AA)
    for mn, col in SHOW:
        ds = det[mn][nm]["dets"]
        if not ds:
            continue
        d = max(ds, key=lambda z: z["score"])
        cv2.polylines(tile, [tr(d["pts"]).astype(np.int32)], True, col, 2, cv2.LINE_AA)
    cv2.polylines(tile, [tr(ref[nm]["quad"]).astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
    for k, pp in enumerate(tr(ref[nm]["quad"]).astype(np.int32)):
        cv2.circle(tile, tuple(pp), 3, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.putText(tile, nm, (10, TS - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 240, 200), 1, cv2.LINE_AA)
    r, c = divmod(i, C)
    grid[r * TS:(r + 1) * TS, c * TS:(c + 1) * TS] = tile
legend = np.full((46, C * TS, 3), 245, np.uint8)
items = [("绿=sp_vision_25 精修基准", (0, 200, 0)), ("灰=多模型共识粗框", (170, 170, 170)), ("红=rp_0526_fp32", (0, 0, 255)),
         ("蓝=szu2026_infantry_fp32", (255, 128, 0)), ("紫=shtech/SKD250526", (255, 0, 255))]
x = 14
for t, c in items:
    cv2.rectangle(legend, (x, 16), (x + 26, 30), c, -1)
    cv2.putText(legend, t, (x + 32, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    x += 34 + 12 * len(t) + 20
out = np.vstack([legend, grid])
cv2.imwrite(os.path.join(FIG, "corners_compare.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 94])
print("[save]", os.path.join(FIG, "corners_compare.jpg"), out.shape)
