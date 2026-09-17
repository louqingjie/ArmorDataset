# -*- coding: utf-8 -*-
"""可视化：单图面板（粗框/精修四点/编号/来源标签）与拼版图。

绘制风格沿用 make_table.py 的常量（线宽/圆点/字号），保证与既有 Test/ 报告一致。
灰色细框 = 双教师共识粗框；彩色粗框 = 最终标签四点（颜色按 B/R/P/N 着色，
回退教师角点时用橙色并在页脚标注 source=teacher）；红点带序号 0..3。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import make_table as MT

from . import config as C

TILE_W, TILE_H = 420, 300
HDR_H = 54
FOOTER_H = 34
FALLBACK_BGR = (0, 165, 255)


def _crop_region(img, quads, margin=1.25):
    pts = np.concatenate([np.asarray(q, np.float32) for q in quads], axis=0)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = max(x2 - x1, 8.0) * margin / 2.0, max(y2 - y1, 8.0) * margin / 2.0
    h, w = img.shape[:2]
    rx1, ry1 = int(max(0, np.floor(cx - hw))), int(max(0, np.floor(cy - hh)))
    rx2, ry2 = int(min(w, np.ceil(cx + hw))), int(min(h, np.ceil(cy + hh)))
    return rx1, ry1, rx2, ry2


def panel(img, objects, tile_w=TILE_W, tile_h=TILE_H, margin=1.25, footer="", note=""):
    """单张图片的标注面板：裁到目标区域后等比放进固定尺寸格子并绘制。"""
    tile = np.full((tile_h, tile_w, 3), 32, np.uint8)
    if not objects:
        cv2.putText(tile, "无检出", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)
    else:
        quads = []
        for o in objects:
            quads.append(o["quad_coarse"])
            quads.append(o["quad_final"])
        x1, y1, x2, y2 = _crop_region(img, quads, margin)
        crop = img[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        s = min((tile_w - 2) / max(cw, 1), (tile_h - 2) / max(ch, 1))
        nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (nw, nh), interpolation=interp)
        ox, oy = (tile_w - nw) // 2, (tile_h - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = crop
        off = np.array([ox - x1 * s, oy - y1 * s], np.float32)

        for o in objects:
            coarse = (np.asarray(o["quad_coarse"], np.float32) * s + off).astype(np.int32)
            cv2.polylines(tile, [coarse], True, (170, 170, 170), 1, cv2.LINE_AA)
            final = (np.asarray(o["quad_final"], np.float32) * s + off).astype(np.int32)
            c = MT.COLOR_BGR.get(o["color"], (0, 255, 0)) if o["source"] == "refine" else FALLBACK_BGR
            cv2.polylines(tile, [final], True, c, MT.LINE_W, cv2.LINE_AA)
            for k, p in enumerate(final):
                cv2.circle(tile, tuple(p), MT.DOT_R, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(tile, str(k), tuple(p + np.array([6, -5])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
            txt = "%s%s %.2f %s" % (o.get("color_name", "?"), o.get("num_name", "?"),
                                    o.get("score_primary") or o.get("score_secondary") or 0.0,
                                    o["source"])
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x = int(np.clip(final[:, 0].min(), 2, max(2, tile_w - tw - 3)))
            y = int(np.clip(final[:, 1].min() - 5, th + 4, tile_h - 4))
            cv2.putText(tile, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
            if o.get("review"):
                cv2.circle(tile, (tile_w - 14, 14), 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(tile, "R", (tile_w - 19, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if footer:
        cv2.rectangle(tile, (0, tile_h - 22), (tile_w, tile_h), (18, 18, 18), -1)
        cv2.putText(tile, footer[:64], (8, tile_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1, cv2.LINE_AA)
    if note:
        cv2.putText(tile, note[:60], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def image_panel(img, stem, objects, margin=1.25):
    """按 meta 中的对象生成面板，页脚给出目标数与复核标记。"""
    n_rev = sum(1 for o in objects if o.get("review"))
    flags = sorted({f for o in objects for f in o["flags"]})
    footer = "%s  n=%d review=%d %s" % (stem, len(objects), n_rev, ",".join(flags)[:40])
    return panel(img, objects, margin=margin, footer=footer, note="原图: %s" % stem)


def sheet(panels, cols=C.SHEET_COLS, title="", notes=(), tile_w=TILE_W, tile_h=TILE_H):
    """把面板拼成网格大图。"""
    n = max(1, len(panels))
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    pad = 6
    note_h = 22 * len(notes) + 8 if notes else 0
    grid = np.full((HDR_H + rows * (tile_h + pad) + pad + note_h, cols * (tile_w + pad) + pad, 3),
                   245, np.uint8)
    if title:
        cv2.putText(grid, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    for i, t in enumerate(panels):
        r, c = divmod(i, cols)
        y = HDR_H + r * (tile_h + pad)
        x = pad + c * (tile_w + pad)
        grid[y:y + tile_h, x:x + tile_w] = t
        cv2.rectangle(grid, (x - 1, y - 1), (x + tile_w, y + tile_h), (170, 170, 170), 1)
    if notes:
        y = HDR_H + rows * (tile_h + pad) + 12
        for k, s in enumerate(notes):
            cv2.putText(grid, s, (12, y + k * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return grid


def save(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)
