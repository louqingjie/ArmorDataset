#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅显示"评判 ROI"的推理结果表格。

沿用 make_table.py 产生的原始推理数据(Test/detections.json)，不重新推理：
  1) 以各检测模型给出的、落在合理范围内的检测框作为候选基准框
     （合理范围: 置信度>=0.45、框面积占图 0.05%~75%、长宽比 0.25~4、被 >=2 个模型共同检出）；
  2) 多装甲板图片取其中面积最大的一个作为基准框；
  3) 评判 ROI = 基准框按中心放大 1.5 倍(宽 x1.5, 高 x1.5)，再裁剪到图像范围内；
  4) 每行一个模型、每列一张图片，格子里只显示该 ROI 区域 + 该模型在原图上的检测叠加。

输出: Test/inference_table_roi.png / .jpg, Test/roi_list.csv
"""
import csv
import glob
import json
import os

import cv2
import numpy as np

import make_table as MT

ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(ROOT, "BaseLine")
OUTDIR = os.path.join(ROOT, "Test")

EXPAND = 1.5          # 评判 ROI 放大倍数（按基准框中心放大）
SCORE_MIN = 0.45      # 候选基准框的最低置信度
AREA_MIN, AREA_MAX = 0.0005, 0.75   # 框面积 / 图像面积 的合理区间
ASPECT_MIN, ASPECT_MAX = 0.25, 4.0  # 宽高比合理区间


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pick_base_box(entries, img_w, img_h):
    """entries: [(model, box, score)] -> (base_box, info) ；多板取面积最大者。"""
    area_img = float(img_w * img_h)
    valid = []
    for model, box, score in entries:
        w, h = box[2] - box[0], box[3] - box[1]
        if w <= 3 or h <= 3 or score < SCORE_MIN:
            continue
        if not (AREA_MIN <= (w * h) / area_img <= AREA_MAX):
            continue
        if not (ASPECT_MIN <= w / max(h, 1e-6) <= ASPECT_MAX):
            continue
        valid.append((model, [float(v) for v in box], float(score)))
    if not valid:
        return None, None
    # 聚类（IoU>=0.5 视为同一块装甲），同一簇内取面积最大框为基准，并要求至少 2 个模型支持
    clusters = []
    for item in sorted(valid, key=lambda z: -((z[1][2] - z[1][0]) * (z[1][3] - z[1][1]))):
        for c in clusters:
            if iou(item[1], c["box"]) >= 0.5:
                c["models"].add(item[0])
                break
        else:
            clusters.append({"box": item[1], "models": {item[0]}, "score": item[2]})
    multi = [c for c in clusters if len(c["models"]) >= 2]
    cand = multi if multi else clusters
    best = max(cand, key=lambda c: (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))
    return best["box"], {"n_models": len(best["models"]), "score": best["score"]}


def main():
    with open(os.path.join(OUTDIR, "detections.json")) as f:
        data = json.load(f)
    names = data["images"]
    images = []
    for nm in names:
        im = cv2.imread(os.path.join(BASELINE, nm), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        images.append(im)

    model_names = [n for n, rel, kind in MT.MODELS]
    det_models = [n for n in model_names if data["det"][n][names[0]]["kind"] != "cls32"]

    # ---------------- 计算每张图的评判 ROI ----------------
    rois, roi_rows = {}, []
    for nm, im in zip(names, images):
        H, W = im.shape[:2]
        entries = []
        for mn in model_names:
            rec = data["det"][mn].get(nm)
            if not rec:
                continue
            for c in rec.get("cands") or rec.get("dets", []):
                entries.append((mn, c["box"], c["score"]))
        base, info = pick_base_box(entries, W, H)
        if base is None:
            roi = [0.0, 0.0, float(W), float(H)]
            note = "无合理检测框 -> 全图"
        else:
            cx, cy = (base[0] + base[2]) / 2, (base[1] + base[3]) / 2
            bw, bh = (base[2] - base[0]) * EXPAND, (base[3] - base[1]) * EXPAND
            roi = [max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(float(W), cx + bw / 2), min(float(H), cy + bh / 2)]
            note = "基准框 %dx%d, %d个模型共同检出, top1=%.2f" % (
                int(round(base[2] - base[0])), int(round(base[3] - base[1])), info["n_models"], info["score"])
        rois[nm] = [int(round(v)) for v in roi]
        roi_rows.append((nm, [round(v, 1) for v in (base or [0, 0, 0, 0])], rois[nm],
                         rois[nm][2] - rois[nm][0], rois[nm][3] - rois[nm][1], note))
        print("[roi] %-9s img=%dx%d -> roi=%s (%dx%d) %s" % (
            nm, W, H, rois[nm], rois[nm][2] - rois[nm][0], rois[nm][3] - rois[nm][1], note))

    # ---------------- 组装表格 ----------------
    TILE_W, TILE_H = 420, 300
    LABEL_W, HEADER_H, PAD = 260, 58, 6
    rows = []
    for mn in model_names:
        tiles = []
        for nm, im in zip(names, images):
            x0, y0, x1, y1 = rois[nm]
            crop = im[y0:y1, x0:x1].copy()
            rec = data["det"][mn][nm]
            res = {"dets": [], "cls": rec.get("cls"), "num_names": MT.SKD_NUM_NAMES if rec["kind"] == "skd" else MT.NUM_NAMES,
                   "roi": rec.get("roi")}
            for d in rec["dets"]:
                pts = np.asarray(d["pts"], np.float32) - np.array([x0, y0], np.float32)
                res["dets"].append({"pts": pts, "box": d["box"], "score": d["score"],
                                    "color": d["color"], "num": d["num"]})
            info = ["%s%s:%.2f" % (MT.COLOR_NAMES.get(d["color"], "?"),
                                   res["num_names"].get(d["num"], "?"), d["score"]) for d in rec["dets"]]
            if rec.get("cls"):
                info.append("数字%d:%.2f" % (rec["cls"][0][0] + 1, rec["cls"][0][1]))
            inside = sum(1 for d in rec["dets"] if iou(d["box"], rois[nm]) > 0.05)
            txt = (" ".join(info) if info else "无检出") + (" | ROI内%d" % inside if rec["dets"] else "")
            tiles.append(MT.compose_tile(crop, res, TILE_W, TILE_H, footer=txt))
        rows.append(tiles)
        print("[row]", mn)

    grid_w = LABEL_W + len(names) * (TILE_W + PAD) + PAD
    grid_h = HEADER_H + len(model_names) * (TILE_H + PAD) + 168
    table = np.full((grid_h, grid_w, 3), 245, np.uint8)
    cv2.putText(table, "推理结果对比表 (仅评判 ROI)   行=模型  列=图片   格子=基准框 x1.5 区域内的检测叠加",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (20, 20, 20), 2, cv2.LINE_AA)
    for j, nm in enumerate(names):
        x = LABEL_W + PAD + j * (TILE_W + PAD)
        W, H = rois[nm][2] - rois[nm][0], rois[nm][3] - rois[nm][1]
        cv2.putText(table, nm, (x + 4, HEADER_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(table, "ROI %dx%d @(%d,%d)" % (W, H, rois[nm][0], rois[nm][1]), (x + 4, HEADER_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    for i, mn in enumerate(model_names):
        y = HEADER_H + i * (TILE_H + PAD)
        fs = 0.66
        while fs > 0.38 and cv2.getTextSize("%02d %s" % (i + 1, mn), cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] > LABEL_W - 18:
            fs -= 0.04
        cv2.putText(table, "%02d %s" % (i + 1, mn), (8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 2, cv2.LINE_AA)
        kind = data["det"][mn][names[0]]["kind"]
        cv2.putText(table, kind, (8, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        for j, t in enumerate(rows[i]):
            x = LABEL_W + PAD + j * (TILE_W + PAD)
            table[y:y + TILE_H, x:x + TILE_W] = t
            cv2.rectangle(table, (x - 1, y - 1), (x + TILE_W, y + TILE_H), (170, 170, 170), 1)

    y = HEADER_H + len(model_names) * (TILE_H + PAD) + 16
    notes = [
        "说明: 1) 完全沿用 Test/detections.json 中的原推理数据(未重新推理/未改阈值)，仅改变显示区域；",
        "      2) 基准框选取: 置信度>=%.2f、框面积占图 %.2f%%~%.0f%%、长宽比 %.2f~%.1f，且被>=2个模型共同检出(IoU>=0.5 视为同一块)；多装甲板图片取面积最大的一个；" % (
            SCORE_MIN, AREA_MIN * 100, AREA_MAX * 100, ASPECT_MIN, ASPECT_MAX),
        "      3) 评判 ROI = 基准框按中心放大 %.1f 倍(宽x%.1f, 高x%.1f)后裁剪到图像范围内；格子里看不到框即表示该模型的检测落在 ROI 之外或该图无检出；" % (EXPAND, EXPAND, EXPAND),
        "      4) 已按评估口径仅保留 18 个检测类模型(裸权重 yolov5.bin 与 2 个 32x32 数字分类器已移除)；",
        "      5) 左列 kind 为解码器类型；编号/颜色映射同总表 (0->7(哨兵) 6->O(前哨站) 7->B(基地)，B蓝 R红 G灰白(未点亮) N其他)。",
    ]
    for k, s in enumerate(notes):
        cv2.putText(table, s, (12, y + k * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    p1 = os.path.join(OUTDIR, "inference_table_roi.png")
    cv2.imwrite(p1, table)
    p2 = os.path.join(OUTDIR, "inference_table_roi.jpg")
    cv2.imwrite(p2, table, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("[save]", p1, table.shape)
    print("[save]", p2)
    with open(os.path.join(OUTDIR, "roi_list.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "base_box(x1,y1,x2,y2)", "roi_box(x1,y1,x2,y2)", "roi_w", "roi_h", "note"])
        for row in roi_rows:
            wr.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
    print("[save]", os.path.join(OUTDIR, "roi_list.csv"))


if __name__ == "__main__":
    main()
