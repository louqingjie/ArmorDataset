# -*- coding: utf-8 -*-
"""几何度量工具：直接复用 evaluate.py 中已用于评估的同一套实现。

保持"标注口径 = 评估口径"，避免自标注用一套几何定义、验收用另一套。
evaluate.py 仅在 __main__ 下执行主流程，作为模块导入只创建 Test/figures 目录。
"""
from __future__ import annotations

import numpy as np

import evaluate as EV

poly_iou = EV.poly_iou
symmetries = EV.symmetries
plate_width = EV.plate_width
align_err = EV.align_err


def align_to(quad, ref):
    """在 8 重对称中取与 ref 平均角点距离最小者，返回对齐后的四点。"""
    q = np.asarray(quad, np.float32)
    return min(symmetries(q), key=lambda c: float(np.mean(np.linalg.norm(c - ref, axis=1))))


def quad_bbox(quad):
    """四边形外接框 -> [x1, y1, x2, y2]。"""
    q = np.asarray(quad, np.float32)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def quad_size(quad):
    """返回 (板宽, 灯条长)：板宽=上下边均值, 灯条长=左右灯条均值。"""
    return plate_width(quad), plate_width(quad, vertical=True)


def roi_of_quad(quad, img_hw, expand=1.5, min_side=32):
    """以四点外接框中心放大 expand 倍并裁剪到图像范围（复用 make_table_roi 公式）。

    返回 (roi[x1,y1,x2,y2] int, info) ; 不合法时返回 (None, info)。
    """
    h, w = int(img_hw[0]), int(img_hw[1])
    x1, y1, x2, y2 = quad_bbox(quad)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * expand, (y2 - y1) * expand
    rx1 = int(max(0.0, np.floor(cx - bw / 2.0)))
    ry1 = int(max(0.0, np.floor(cy - bh / 2.0)))
    rx2 = int(min(float(w), np.ceil(cx + bw / 2.0)))
    ry2 = int(min(float(h), np.ceil(cy + bh / 2.0)))
    info = {"quad_wh_px": [round(x2 - x1, 2), round(y2 - y1, 2)],
            "roi": [rx1, ry1, rx2, ry2], "roi_wh_px": [rx2 - rx1, ry2 - ry1]}
    if rx2 - rx1 < min_side or ry2 - ry1 < min_side:
        info["reason"] = "roi_too_small"
        return None, info
    return [rx1, ry1, rx2, ry2], info
