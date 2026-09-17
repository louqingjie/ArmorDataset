# -*- coding: utf-8 -*-
"""1.5× ROI 裁剪 + sp_vision_25 传统视觉角点精修。

流程（对每个目标）:
  1) 以粗框四点外接框为中心放大 ROI_EXPAND(=1.5) 倍，裁剪到图像范围；
  2) 复用 make_reference.sp_refine（灯条阈值扫描 → minAreaRect → 几何校验 →
     最近灯条端点覆盖四角），其内部还会再按 ±1 灯条长 / ±0.75 板宽外扩旋转 ROI；
  3) 精修结果做两道防跳变校验（与粗框 IoU、板宽比），通过则 source=refine，
     否则回退粗框 source=teacher 并打 refine_failed + 具体原因，进入人工复核清单。
"""
from __future__ import annotations

import numpy as np

import make_reference as MR

from . import config as C
from .geom import align_to, plate_width, poly_iou, roi_of_quad

# 精修被拒/失败时置入 object["reason"] 的原因码
REASON_ROI_SMALL = "roi_too_small"
REASON_NO_BAR = "no_bar_pair"
REASON_IOU_GUARD = "iou_guard"
REASON_SIZE_GUARD = "size_guard"
REASON_PLATE_SMALL = "plate_too_small"


def sp_refine_two_stage(sub_gray, quad_local, extra_thresholds=C.REFINE_THRESHOLDS_EXTRA):
    """两阶段精修：先原始阈值档（与既有基准一致），失败再用补充档。"""
    MR.THRESHOLDS = list(C.REFINE_THRESHOLDS_BASE)      # sp_refine 读模块级阈值表
    q, info = MR.sp_refine(sub_gray, quad_local)
    if q is not None:
        info = dict(info or {})
        info["stage"] = "base"
        return q, info
    if extra_thresholds:
        MR.THRESHOLDS = list(extra_thresholds)
        q2, info2 = MR.sp_refine(sub_gray, quad_local)
        if q2 is not None:
            info2 = dict(info2 or {})
            info2["stage"] = "extra"
            return q2, info2
    info = dict(info or {})
    info["stage"] = "base+extra"
    return None, info


def refine_object(gray, obj, expand=C.ROI_EXPAND, min_side=C.ROI_MIN_SIDE,
                  iou_guard=C.REFINE_IOU_GUARD,
                  size_lo=C.REFINE_SIZE_LO, size_hi=C.REFINE_SIZE_HI,
                  tiny_w=C.TINY_PLATE_W, min_plate_w=C.MIN_REFINE_PLATE_W,
                  extra_thresholds=C.REFINE_THRESHOLDS_EXTRA):
    """在 1.5× ROI 上精修单个目标；就地补充 quad_final / source / refine / flags / reason。"""
    quad = np.asarray(obj["quad_coarse"], np.float32)
    pw_coarse = float(plate_width(quad))
    if min_plate_w and pw_coarse < min_plate_w:
        # 小目标跳过精修：阈值法不可靠，教师角点相对精度已足够，不产生复核负担
        obj["quad_final"] = quad
        obj["source"] = "teacher"
        obj["reason"] = REASON_PLATE_SMALL
        obj["flags"] = list(obj["flags"]) + ["refine_skipped", REASON_PLATE_SMALL]
        if is_tiny(quad, tiny_w):
            obj["flags"].append("tiny")
        obj["refine"] = {"expand": expand, "plate_w_coarse": round(pw_coarse, 2),
                         "skipped": True, "accepted": False, "reason": REASON_PLATE_SMALL}
        return obj
    roi, roi_info = roi_of_quad(quad, gray.shape[:2], expand=expand, min_side=min_side)
    info = {"expand": expand, "roi": roi_info.get("roi"), "roi_wh_px": roi_info.get("roi_wh_px"),
            "plate_w_coarse": round(pw_coarse, 2), "accepted": False}

    def fallback(reason, extra=None):
        obj["quad_final"] = quad
        obj["source"] = "teacher"
        obj["reason"] = reason
        obj["flags"] = list(obj["flags"]) + ["refine_failed", reason]
        if is_tiny(quad, tiny_w):
            obj["flags"].append("tiny")
        if extra:
            info.update(extra)
        obj["refine"] = info
        return obj

    if roi is None:
        return fallback(REASON_ROI_SMALL)

    x1, y1, x2, y2 = roi
    sub = np.ascontiguousarray(gray[y1:y2, x1:x2])
    q_local = quad - np.array([x1, y1], np.float32)
    q_ref, rinfo = sp_refine_two_stage(sub, q_local, extra_thresholds)
    if rinfo:
        info.update(rinfo)
    if q_ref is None:
        return fallback(REASON_NO_BAR)

    q = np.asarray(q_ref, np.float32) + np.array([x1, y1], np.float32)
    io = float(poly_iou(q, quad))
    pw = float(plate_width(q))
    ratio = pw / max(pw_coarse, 1e-6)
    d_mean = float(np.mean(np.linalg.norm(align_to(q, quad) - quad, axis=1)))
    info.update({"iou_vs_coarse": round(io, 3), "plate_w_refine": round(pw, 2),
                 "plate_w_ratio": round(ratio, 3),
                 "d_coarse_to_refine_px": round(d_mean, 2),
                 "d_coarse_to_refine_pct": round(100.0 * d_mean / max(pw_coarse, 1e-6), 2)})

    if io < iou_guard:
        return fallback(REASON_IOU_GUARD)
    if not (size_lo <= ratio <= size_hi):
        return fallback(REASON_SIZE_GUARD)

    obj["quad_final"] = q
    obj["source"] = "refine"
    obj["reason"] = None
    obj["flags"] = list(obj["flags"]) + ["refine_ok"]
    if is_tiny(q, tiny_w):
        obj["flags"].append("tiny")
    info["accepted"] = True
    obj["refine"] = info
    return obj


def is_tiny(quad, tiny_w=C.TINY_PLATE_W):
    """板宽过小（远距离目标），角点标注不可靠，标记但保留。"""
    return float(plate_width(quad)) < tiny_w


def refine_objects(gray, objects, **kw):
    """对一张图的全部目标做精修，返回 (objects, 本图统计)。"""
    out, n_ok = [], 0
    for obj in objects:
        refine_object(gray, obj, **kw)
        if obj["source"] == "refine":
            n_ok += 1
        out.append(obj)
    return out, {"n_obj": len(out), "n_refined": n_ok,
                 "n_failed": len(out) - n_ok}
