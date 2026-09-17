#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评判基准关键点生成 —— 按 TongjiSuperPower/sp_vision_25 的传统视觉角点方案重写。

参考实现（tasks/auto_aim/detector.cpp + armor.cpp, configs/standard3.yaml）：
  灯条: gray → threshold → findContours(EXTERNAL, NONE) → minAreaRect → 4 角点按 y 排序
        top = (c0+c1)/2, bottom = (c2+c3)/2（即灯条中心线两端）
        width = |c0-c1|, length = |bottom-top|, ratio = length/width
        angle_error = |atan2(bottom-top) - pi/2|
        校验: angle_error < 45°, 1.5 < ratio < 20, length > 8
  装甲板: points = [left.top, right.top, right.bottom, left.bottom]（左上→右上→右下→左下）
        ratio = |right.center-left.center| / max(len) ∈ (1,5); side_ratio < 1.5; rectangular_error < 25°
  角点精修（detector.cpp detect(Armor&, bgr_img)）:
        以当前四点外扩旋转 ROI（纵向 ±1 灯条长, 横向 ±0.75 板宽）→ ROI 内重跑灯条提取
        → 取与左/右两侧角点距离和最小的灯条 → 距离和 < 门限则用灯条端点覆盖四角

本脚本以"18 模型四点中位数"作为粗框(替代其 YOLO/粗灯条结果)，再执行上述精修；
图像尺度差异大(182px~4096px)，故对阈值做扫描、对接受门限按板宽缩放(≥15px 且 ≤0.25 板宽)。
输出: Test/reference_kpt.json, Test/reference_kpt_table.csv, Test/figures/reference_kpt.png
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

# --- sp_vision_25 configs/standard3.yaml 传统方法参数 ---
THRESHOLDS = [80, 100, 120, 140, 160, 180, 200]
MIN_LEN, MIN_RATIO, MAX_RATIO, MAX_ANG = 8.0, 1.5, 20.0, np.deg2rad(45)
MIN_ARMOR_RATIO, MAX_ARMOR_RATIO, MAX_SIDE_RATIO, MAX_RECT_ERR = 1.0, 5.0, 1.5, np.deg2rad(25)


def bars_of(gray, thr):
    """按 sp_vision_25 方式提取灯条（top/bottom 为灯条中心线两端）。"""
    _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bars = []
    for c in cnts:
        rr = cv2.minAreaRect(c)
        p = cv2.boxPoints(rr)
        p = sorted(p, key=lambda q: q[1])
        p = [np.asarray(q, np.float32) for q in p]
        top = (p[0] + p[1]) / 2.0
        bottom = (p[2] + p[3]) / 2.0
        t2b = bottom - top
        length = float(np.linalg.norm(t2b))
        width = float(np.linalg.norm(p[0] - p[1]))
        if width < 1e-3 or length < 1e-3:
            continue
        ratio = length / width
        ang_err = abs(np.arctan2(t2b[1], t2b[0]) - np.pi / 2)
        if ang_err > MAX_ANG or not (MIN_RATIO < ratio < MAX_RATIO) or length < MIN_LEN:
            continue
        bars.append({"top": top, "bottom": bottom, "center": np.asarray(rr[0], np.float32),
                     "length": length, "width": width, "ratio": ratio})
    bars.sort(key=lambda b: b["center"][0])
    return bars


def armor_ok(left, right):
    """sp_vision_25 的装甲板级校验：ratio / side_ratio / rectangular_error。"""
    width = float(np.linalg.norm(right["center"] - left["center"]))
    ratio = width / max(left["length"], right["length"])
    side_ratio = max(left["length"], right["length"]) / max(1e-3, min(left["length"], right["length"]))
    if not (MIN_ARMOR_RATIO < ratio < MAX_ARMOR_RATIO) or side_ratio > MAX_SIDE_RATIO:
        return False, ratio, side_ratio
    # 灯条方向与"两中点连线"的夹角偏差
    d = right["center"] - left["center"]
    ang = abs(np.arctan2(d[1], d[0]))
    rect_err = max(abs(left["top"][0] - left["bottom"][0]) * 0 + 0.0, 0.0)
    for b in (left, right):
        e = np.arctan2((b["bottom"] - b["top"])[1], (b["bottom"] - b["top"])[0])
        rect_err = max(rect_err, abs(abs(e - ang) - np.pi / 2))
    return (rect_err < MAX_RECT_ERR), ratio, side_ratio


def sp_refine(gray, coarse):
    """detector.cpp 的角点精修：外扩旋转 ROI → 重提取灯条 → 最近灯条端点覆盖四角。

    coarse: [LT, LB, RB, RT]（本工程模型约定）; 返回 (四点, 信息) 或 (None, 信息)
    """
    tl, lb, rb, rt = [np.asarray(p, np.float32) for p in coarse]
    lt2b, rt2b = lb - tl, rb - rt
    tl1 = (tl + lb) / 2 - lt2b
    bl1 = (tl + lb) / 2 + lt2b
    br1 = (rt + rb) / 2 + rt2b
    tr1 = (rt + rb) / 2 - rt2b
    tl2tr, bl2br = tr1 - tl1, br1 - bl1
    tl2 = (tl1 + rt) / 2 - 0.75 * tl2tr
    tr2 = (tl1 + rt) / 2 + 0.75 * tl2tr
    bl2 = (bl1 + rb) / 2 - 0.75 * bl2br
    br2 = (bl1 + rb) / 2 + 0.75 * bl2br
    box = cv2.boundingRect(np.stack([tl2, tr2, br2, bl2]).astype(np.float32))
    H, W = gray.shape[:2]
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(W, box[0] + box[2]), min(H, box[1] + box[3])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, {"reason": "ROI 过小/越界"}
    roi = gray[y0:y1, x0:x1]
    off = np.array([x0, y0], np.float32)
    tl_l, lb_l, rb_l, rt_l = tl - off, lb - off, rb - off, rt - off
    plate_w = float(np.linalg.norm(rt - tl))
    accept = max(15.0, 0.25 * plate_w)
    best = None
    for thr in THRESHOLDS:
        bars = bars_of(roi, thr)
        if len(bars) < 2:
            continue
        left = min(bars, key=lambda b: np.linalg.norm(b["top"] - tl_l) + np.linalg.norm(b["bottom"] - lb_l))
        right = min(bars, key=lambda b: np.linalg.norm(b["top"] - rt_l) + np.linalg.norm(b["bottom"] - rb_l))
        if left is right:
            continue
        d = (float(np.linalg.norm(left["top"] - tl_l) + np.linalg.norm(left["bottom"] - lb_l)) +
             float(np.linalg.norm(right["top"] - rt_l) + np.linalg.norm(right["bottom"] - rb_l)))
        ok, ratio, side = armor_ok(left, right)
        info = {"thr": thr, "dist": round(d, 2), "accept": round(accept, 1),
                "armor_ratio": round(ratio, 2), "side_ratio": round(side, 2), "armor_ok": bool(ok)}
        if d < accept and (best is None or d < best[0]):
            q = np.stack([left["top"], left["bottom"], right["bottom"], right["top"]]).astype(np.float32) + off
            best = (d, q, info)
    if best is None:
        return None, {"reason": "无满足距离/几何门限的灯条对", "accept": round(accept, 1)}
    return best[1], best[2]


def symmetries(q):
    q = np.asarray(q, np.float32)
    out = []
    for rev in (False, True):
        base = q[::-1] if rev else q
        for k in range(4):
            out.append(np.roll(base, k, axis=0))
    return out


def align(q, ref):
    return min(symmetries(q), key=lambda c: float(np.mean(np.linalg.norm(c - ref, axis=1))))


def poly_iou(q1, q2):
    q1 = cv2.convexHull(np.asarray(q1, np.float32))
    q2 = cv2.convexHull(np.asarray(q2, np.float32))
    a1 = cv2.contourArea(q1)
    a2 = cv2.contourArea(q2)
    inter, _ = cv2.intersectConvexConvex(q1, q2)
    u = a1 + a2 - inter
    return float(inter / u) if u > 0 else 0.0


def main():
    data = json.load(open(os.path.join(OUT, "detections.json")))
    names = data["images"]
    ref, rows, panels = {}, [], []
    for nm in names:
        im = cv2.imread(os.path.join(ROOT, "BaseLine", nm), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        quads = []
        for m, per in data["det"].items():
            ds = per[nm]["dets"]
            if ds:
                quads.append(np.asarray(max(ds, key=lambda z: z["score"])["pts"], np.float32))
        cons = np.median(np.stack(quads, 0), 0)
        for _ in range(2):
            cons = np.median(np.stack([align(q, cons) for q in quads], 0), 0)
        q, info = sp_refine(gray, cons)
        src = "sp_vision_25 角点精修" if q is not None else "共识(精修未通过)"
        if q is None:
            q = cons
        iou = poly_iou(q, cons)
        size = [float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))]
        ref[nm] = {"quad": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in q],
                   "consensus": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in cons],
                   "source": src, "refine": info, "iou_refine_vs_cons": round(iou, 3),
                   "n_models": len(quads), "size": size, "img_size": list(im.shape[:2])}
        print("[ref] %-9s %-22s IoU(vs粗框)=%.3f d=%s thr=%s 板宽=%.0fpx" % (
            nm, src, iou, info.get("dist"), info.get("thr"), size[0]))
        rows.append([nm] + ["(%d,%d)" % (round(p[0]), round(p[1])) for p in q] +
                    [src, info.get("dist", ""), info.get("thr", ""), round(iou, 3), round(size[0])])
        # 面板
        cx, cy = q[:, 0].mean(), q[:, 1].mean()
        half = max(size) * 0.9 + 14
        x0, x1 = int(max(0, cx - half)), int(min(im.shape[1], cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(im.shape[0], cy + half))
        crop = im[y0:y1, x0:x1].copy()
        s = min(300 / crop.shape[1], 300 / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        tile = np.full((374, 330, 3), 250, np.uint8)
        oy, ox = 50 + (300 - crop.shape[0]) // 2, (330 - crop.shape[1]) // 2
        tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        tr = lambda p: (np.asarray(p, np.float32) - np.array([x0, y0], np.float32)) * s + np.array([ox, oy], np.float32)
        cv2.polylines(tile, [tr(cons).astype(np.int32)], True, (170, 170, 170), 1, cv2.LINE_AA)
        pts = tr(q).astype(np.int32)
        cv2.polylines(tile, [pts], True, (0, 180, 0), 2, cv2.LINE_AA)
        for k, p in enumerate(pts):
            cv2.circle(tile, tuple(p), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(tile, str(k), tuple(p + np.array([7, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, nm, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(tile, src, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 130, 0) if q is not None else (140, 60, 60), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 344), (330, 374), (238, 238, 238), -1)
        cv2.putText(tile, "thr=%s  d=%spx  两灯条间距/灯条长=%.2f" % (info.get("thr", "-"), info.get("dist", "-"), info.get("armor_ratio", 0)),
                    (8, 364), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        panels.append(tile)
    C, R = 4, 3
    grid = np.full((70 + R * 374, C * 330, 3), 250, np.uint8)
    cv2.putText(grid, "评判基准关键点 (sp_vision_25 角点精修方案): 绿=基准四点(0左上/1左下/2右下/3右上), 灰=多模型共识粗框",
                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2, cv2.LINE_AA)
    for i, t in enumerate(panels):
        r, c = divmod(i, C)
        grid[70 + r * 374:70 + (r + 1) * 374, c * 330:(c + 1) * 330] = t
    cv2.imwrite(os.path.join(FIG, "reference_kpt.png"), grid)
    with open(os.path.join(OUT, "reference_kpt_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "p0左上", "p1左下", "p2右下", "p3右上", "基准来源", "精修匹配距离px", "二值化阈值", "精修↔粗框IoU", "基准宽px"])
        w.writerows(rows)
    json.dump(ref, open(os.path.join(OUT, "reference_kpt.json"), "w"), ensure_ascii=False, indent=1)
    print("[save] reference_kpt.json / reference_kpt_table.csv / figures/reference_kpt.png")


if __name__ == "__main__":
    main()
