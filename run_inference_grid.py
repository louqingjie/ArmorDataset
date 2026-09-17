#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboMaster 装甲板权重批量推理 + 结果表格图生成

流程:
  1. 扫描 Weights/ 下所有权重 (*.onnx / *.bin)
  2. 对 BaseLine/ 下每张图片做推理 (letterbox 预处理 + 自动解析输出头)
  3. 把推理结果 (检测框 / 关键点 / 热力图 / 分类概率) 叠回原图
  4. 生成 "行 = 模型, 列 = 图片" 的表格大图, 输出到 Test/

用法:
  python run_inference_grid.py
  python run_inference_grid.py --conf 0.3 --cell-width 300 --save-overlays
  python run_inference_grid.py --models rp_0526 szu2026        # 只跑名字含这些子串的权重
  python run_inference_grid.py --exclude probe tiny_resnet     # 跳过部分权重
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "Weights"
IMAGE_DIR = ROOT / "BaseLine"
TEST_DIR = ROOT / "Test"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WEIGHT_EXTS = {".onnx", ".bin", ".engine", ".trt", ".pt"}

PAD_COLOR = 114
CLASS_COLORS = [
    (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255),
    (128, 255, 0), (255, 0, 128),
]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)


def nms(boxes, scores, iou_thr=0.45):
    """纯 numpy NMS, boxes 为 xyxy。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.clip(area_i + area_r - inter, 1e-9, None)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def parse_names(meta):
    """从 onnx metadata 里解析类别名 / 关键点形状。"""
    names, kpt_shape = None, None
    for key, value in (meta or {}).items():
        k = key.lower()
        if k == "names" and value:
            try:
                obj = ast.literal_eval(value)
                if isinstance(obj, dict):
                    names = {int(k2): str(v2) for k2, v2 in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    names = {i: str(v2) for i, v2 in enumerate(obj)}
            except Exception:
                pass
        elif k == "kpt_shape" and value:
            try:
                obj = ast.literal_eval(value)
                if isinstance(obj, (list, tuple)) and len(obj) == 2:
                    kpt_shape = (int(obj[0]), int(obj[1]))
            except Exception:
                pass
    return names, kpt_shape


# --------------------------------------------------------------------------- #
# ONNX 会话封装
# --------------------------------------------------------------------------- #
class OrtModel:
    """封装 onnxruntime 会话 + letterbox 预处理。"""

    def __init__(self, path: Path, default_hw=(480, 640)):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.path = Path(path)
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        self.provider = self.sess.get_providers()[0]
        try:
            self.meta = dict(self.sess.get_modelmeta().custom_metadata_map)
        except Exception:
            self.meta = {}
        self.names, self.kpt_shape = parse_names(self.meta)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()

        spec = self.inputs[0].shape
        self.input_hw, self.input_ch, self.layout = self._resolve_input(spec, default_hw)

    @staticmethod
    def _resolve_input(spec, default_hw):
        hw, ch, layout = default_hw, 3, "NCHW"
        if len(spec) == 4:
            dims = list(spec)
            if isinstance(dims[-1], int) and dims[-1] == 3 and not (isinstance(dims[1], int) and dims[1] == 3):
                layout = "NHWC"
            if layout == "NCHW":
                c, h, w = dims[1], dims[2], dims[3]
            else:
                h, w, c = dims[1], dims[2], dims[3]
            if isinstance(c, int) and c > 0:
                ch = c
            if isinstance(h, int) and h > 0:
                hw = (h, hw[1])
            if isinstance(w, int) and w > 0:
                hw = (hw[0], w)
        elif len(spec) == 2:  # 少见: 平坦输入
            layout = "FLAT"
        return (int(hw[0]), int(hw[1])), int(ch), layout

    def preprocess(self, img):
        """letterbox 到模型输入尺寸, 返回 blob / ratio / (left, top)。"""
        h, w = self.input_hw
        ih, iw = img.shape[:2]
        r = min(h / ih, w / iw)
        nh, nw = max(1, int(round(ih * r))), max(1, int(round(iw * r)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h, w, 3), PAD_COLOR, np.uint8)
        top, left = (h - nh) // 2, (w - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.input_ch == 1:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            blob = gray[None]
        elif self.layout == "NHWC":
            blob = rgb[None]
        else:
            blob = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]
        if self.input_ch > 0 and self.input_ch % blob.shape[1] == 0 and self.input_ch != blob.shape[1]:
            blob = np.concatenate([blob] * (self.input_ch // blob.shape[1]), axis=1)
        return np.ascontiguousarray(blob.astype(np.float32)), r, (left, top)

    def run(self, img):
        blob, r, pad = self.preprocess(img)
        # 多输入模型 (如双帧 / 多尺度) 统一用同一张图填充
        feed = {spec.name: blob for spec in self.inputs}
        outs = self.sess.run(None, feed)
        return [np.asarray(o, dtype=np.float32) for o in outs], r, pad

    def describe(self):
        return {
            "input": [list(i.shape) for i in self.inputs],
            "input_used_hw": list(self.input_hw),
            "channels": self.input_ch,
            "layout": self.layout,
            "outputs": [list(o.shape) for o in self.outputs],
            "providers": self.sess.get_providers(),
            "names": self.names,
            "kpt_shape": list(self.kpt_shape) if self.kpt_shape else None,
        }


# --------------------------------------------------------------------------- #
# 输出头解析
# --------------------------------------------------------------------------- #
def _matrix_candidates(tensor):
    """把任意输出张量转成若干候选 (N, C) 矩阵。"""
    a = np.nan_to_num(np.asarray(tensor, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.squeeze(a)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    mats = []
    if a.ndim == 2:
        n0, c0 = a.shape
        if n0 >= 10 and 5 <= c0 <= 200 and c0 < n0:
            mats.append(a)
        if c0 >= 10 and 5 <= n0 <= 200 and n0 < c0:
            mats.append(np.ascontiguousarray(a.T))
    return a, mats


def _box_layout(matrix):
    """判断 box 是 xyxy 还是 cxcywh (按几何合理性投票)。"""
    b = matrix[:, :4]
    xyxy_ok = np.mean((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]))
    return "xyxy" if xyxy_ok > 0.9 else "cxcywh"


def _score_matrix(p, kind, nc, ndim, hint_nc, hint_kpt):
    """对一种 "布局假设" 打分并解码。返回 (quality, result) 或 None。"""
    n = p.shape[0]
    off = 5 if kind == "det5" else 4
    cls_block = p[:, off:off + nc]
    if nc == 1:
        cls_block = cls_block.reshape(n, 1)

    frac_raw = float(np.mean((cls_block >= -0.02) & (cls_block <= 1.02)))
    cls_sig = sigmoid(cls_block)
    frac_sig = float(np.mean((cls_sig >= -0.02) & (cls_sig <= 1.02)))
    if nc == 1:
        frac_raw = frac_sig = 1.0

    if frac_raw >= 0.9:
        # 类别块本身就是概率, 无需变换
        probs, frac, use_sigmoid, col_ok = np.clip(cls_block, 0.0, 1.0), frac_raw, False, 1.0
    else:
        # 需要 sigmoid; 同时判断这些列是否其实是坐标列(如关键点/框被误当作类别)
        col_med = np.median(np.abs(cls_block), axis=0)
        col_ok = 1.0 - float(np.mean(col_med > 1.0))
        probs, frac, use_sigmoid = cls_sig, frac_sig, True

    best_cls = probs.argmax(1)
    best_score = probs.max(1)

    if kind == "e2e":
        boxes = p[:, :4]
        confs = np.clip(p[:, 4], 0.0, 1.0)
        cls_ids = np.rint(p[:, 5]).astype(np.int64)
        int_ratio = float(np.mean(np.abs(p[:, 5] - cls_ids) < 0.15))
        frac = float(np.mean((p[:, 4] >= -0.02) & (p[:, 4] <= 1.02))) * int_ratio
        boxes_layout = _box_layout(p)
    else:
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        confs = best_score if kind != "det5" else np.clip(p[:, 4], 0, 1) * best_score
        cls_ids = best_cls
        boxes_layout = "cxcywh"

    # 坐标尺度: 归一化 or 像素
    vals = boxes.reshape(-1)
    q99 = float(np.percentile(np.abs(vals), 99)) if vals.size else 0.0
    norm_coords = q99 <= 1.5

    ref = 1.0 if norm_coords else float(np.mean([p.shape[0], 1]) or 1.0)
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]

    if norm_coords:
        lo, hi = -0.05, 1.05
    else:
        span = float(np.percentile(np.abs(boxes), 99)) + 1e-6
        lo, hi = -0.05 * span, 1.05 * span
    inb = float(np.mean((boxes[:, 0] >= lo) & (boxes[:, 2] <= hi) &
                        (boxes[:, 1] >= lo) & (boxes[:, 3] <= hi)))
    pos = float(np.mean((ws > 0) & (hs > 0)))

    top = np.sort(confs)[-min(20, n):]
    top_conf = float(np.mean(top)) if top.size else 0.0

    quality = frac * pos * (0.35 + 0.65 * col_ok) * (0.5 + 0.5 * inb) * (0.5 + 0.5 * top_conf)
    if hint_nc and nc == hint_nc:
        quality *= 1.6
    if kind == "e2e":
        quality *= 1.25
    if hint_kpt and ndim == hint_kpt[0] * hint_kpt[1]:
        quality *= 1.3

    result = {
        "kind": kind,
        "nc": int(nc),
        "boxes": boxes.astype(np.float32),
        "confs": confs.astype(np.float32),
        "classes": cls_ids.astype(np.int64),
        "norm_coords": bool(norm_coords),
        "raw_layout": "xyxy" if kind == "e2e" else "cxcywh",
        "sigmoid": bool(use_sigmoid),
        "kpts": None,
        "kpt_ndim": 0,
    }

    if kind in ("pose",) and ndim > 0:
        kraw = p[:, off + nc:off + nc + ndim]
        result["kpts"] = kraw.astype(np.float32)
        result["kpt_ndim"] = int(ndim)
        result["kpt_shape"] = (ndim // kdim_per_kpt(ndim), kdim_per_kpt(ndim))

    return float(quality), result


def kdim_per_kpt(ndim):
    for d in (2, 3, 1):
        if ndim % d == 0:
            return d
    return 2


def _hypotheses(matrix, hint_nc, hint_kpt):
    c = matrix.shape[1]
    combos = []
    for nc in range(1, 31):
        if c == 4 + nc:
            combos.append(("det", nc, 0))
        if c == 5 + nc:
            combos.append(("det5", nc, 0))
        for nnodes in (4, 2, 3):
            for dims in (2, 3, 1):
                ndim = nnodes * dims
                if c == 4 + nc + ndim:
                    combos.append(("pose", nc, ndim))
    if c == 6:
        combos.append(("e2e", 1, 0))
    if c == 7:
        combos.append(("e2e", 1, 0))
    if hint_nc:
        combos.sort(key=lambda t: 0 if t[1] == hint_nc else 1)
    return combos


def _decode_matrix(matrix, hint_nc, hint_kpt):
    best = None
    for kind, nc, ndim in _hypotheses(matrix, hint_nc, hint_kpt):
        out = _score_matrix(matrix, kind, nc, ndim, hint_nc, hint_kpt)
        if out is None:
            continue
        q, res = out
        if best is None or q > best[0]:
            best = (q, res)
    return best


def decode_detections(raw_outs, hint_nc=None, hint_kpt=None):
    """尝试把 raw 输出解析成检测结果, 返回 (quality, result, debug)。"""
    mats, notes = [], []
    for idx, tensor in enumerate(raw_outs):
        arr, cands = _matrix_candidates(tensor)
        notes.append({"out": idx, "raw_shape": list(np.shape(tensor)), "squeezed": list(arr.shape),
                      "matrices": [list(m.shape) for m in cands]})
        for ci, m in enumerate(cands):
            mats.append((f"out{idx}#{ci}", m))

    best_q, best_res, best_tag = 0.0, None, None
    for tag, m in mats:
        got = _decode_matrix(m, hint_nc, hint_kpt)
        if got and got[0] > best_q:
            best_q, best_res, best_tag = got[0], got[1], tag

    # 多输出: 按通道拼接后统一解码 (split 头的情况)
    if len(mats) > 1:
        for flip in (False, True):
            group = [(t, (m.T if flip else m)) for t, m in mats]
            n_rows = {m.shape[0] for _, m in group}
            if len(n_rows) != 1:
                continue
            merged = np.concatenate([m for _, m in group], axis=1)
            got = _decode_matrix(merged, hint_nc, hint_kpt)
            if got and got[0] > best_q:
                best_q, best_res, best_tag = got[0], got[1], "concat:" + "+".join(t for t, _ in group)

    return best_q, best_res, {"outputs": notes, "chosen": best_tag}


def scale_to_original(res, pad, ratio):
    """把模型坐标系下的结果映射回原图坐标。"""
    left, top = pad
    boxes = res["boxes"].copy()
    if res["norm_coords"]:
        pass  # 归一化坐标在绘制阶段单独处理
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    kpts = res.get("kpts")
    kpts_out = None
    if kpts is not None:
        kpts = kpts.copy()
        kpts[:, 0::res["kpt_ndim"]] = (kpts[:, 0::res["kpt_ndim"]] - left) / ratio
        kpts[:, 1::res["kpt_ndim"]] = (kpts[:, 1::res["kpt_ndim"]] - top) / ratio
        kpts_out = kpts
    return boxes, kpts_out


def postprocess(res, conf_thr, iou_thr, norm_wh=None):
    """阈值 + NMS, 返回 (boxes, confs, classes, kpts)。"""
    boxes = res["boxes"].copy()
    if res["norm_coords"]:
        h, w = norm_wh
        boxes = np.stack([boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h], axis=1)
        if res.get("kpts") is not None and res["kpt_ndim"] >= 2:
            k = res["kpts"].copy()
            k[:, 0::res["kpt_ndim"]] *= w
            k[:, 1::res["kpt_ndim"]] *= h
            res["kpts"] = k
    confs, classes = res["confs"], res["classes"]
    keep = confs >= conf_thr
    boxes, confs, classes = boxes[keep], confs[keep], classes[keep]
    kpts = res["kpts"][keep] if res.get("kpts") is not None else None
    if boxes.size:
        order = confs.argsort()[::-1]
        boxes, confs, classes = boxes[order], confs[order], classes[order]
        kpts = kpts[order] if kpts is not None else None
        idx = nms(boxes, confs, iou_thr)
        boxes, confs, classes = boxes[idx], confs[idx], classes[idx]
        kpts = kpts[idx] if kpts is not None else None
    return boxes, confs, classes, kpts


# --------------------------------------------------------------------------- #
# 结果可视化
# --------------------------------------------------------------------------- #
def draw_detections(img, boxes, confs, classes, kpts, kpt_ndim, names, max_det=20):
    out = img.copy()
    n = len(confs)
    for i in range(min(n, max_det)):
        x1, y1, x2, y2 = boxes[i]
        cls_id, conf = int(classes[i]), float(confs[i])
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cname = names.get(cls_id, str(cls_id)) if names else str(cls_id)
        label = f"{cname} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        if kpts is not None and kpt_ndim >= 2:
            pts = kpts[i].reshape(-1, kpt_ndim)
            vis = []
            for p in pts:
                if kpt_ndim == 3 and p[2] < 0.2:
                    continue
                vis.append((int(round(p[0])), int(round(p[1]))))
            for j in range(len(vis) - 1):
                cv2.line(out, vis[j], vis[j + 1], (0, 255, 255), 2, cv2.LINE_AA)
            if len(vis) > 2:
                cv2.line(out, vis[-1], vis[0], (0, 255, 255), 2, cv2.LINE_AA)
            for p in vis:
                cv2.circle(out, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def draw_badge(img, text, color=(30, 30, 30), bg=(255, 255, 255)):
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (4, 4), (12 + tw, 12 + th), bg, -1)
    cv2.rectangle(out, (4, 4), (12 + tw, 12 + th), color, 1)
    cv2.putText(out, text, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def render_heatmap(img, tensor):
    a = np.squeeze(np.asarray(tensor, dtype=np.float32))
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3:
        heat = np.abs(a).mean(0)
    elif a.ndim == 2:
        heat = np.abs(a)
    else:
        return None
    heat -= heat.min()
    if heat.max() > 1e-6:
        heat /= heat.max()
    ih, iw = img.shape[:2]
    heat = cv2.resize(heat, (iw, ih), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.45, colored, 0.55, 0)


def render_text_panel(img, lines, title=None):
    out = img.copy()
    h = out.shape[0]
    scale = max(0.6, min(2.2, out.shape[1] / 700.0))
    y = int(h * 0.10)
    if title:
        cv2.putText(out, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)
        y += int(scale * 42)
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += int(scale * 38)
    return out


# --------------------------------------------------------------------------- #
# 表格图
# --------------------------------------------------------------------------- #
def find_font():
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return p
            except Exception:
                continue
    return None


def vis_base(img, target_w):
    """用于展示的缩放 (文字在缩放后的小图上才看得清)。"""
    scale = target_w / img.shape[1]
    if abs(scale - 1.0) < 1e-3:
        return img.copy(), 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp), scale


def fit_cell(bgr, cw, ch):
    ih, iw = bgr.shape[:2]
    r = min(cw / iw, ch / ih)
    nw, nh = max(1, int(round(iw * r))), max(1, int(round(ih * r)))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    canvas = np.full((ch, cw, 3), 255, np.uint8)
    x0, y0 = (cw - nw) // 2, (ch - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def build_grid(cells, row_labels, col_labels, out_path, cell_w=300, subtitle=""):
    from PIL import Image, ImageDraw, ImageFont

    cell_h = int(round(cell_w * 0.78))
    label_w = max(240, int(cell_w * 0.95))
    head_h = 40
    gap = 6
    title_h = 60
    foot_h = 40
    n_rows, n_cols = len(row_labels), len(col_labels)
    W = label_w + n_cols * (cell_w + gap) + gap
    H = title_h + head_h + n_rows * (cell_h + gap) + gap + foot_h
    canvas = Image.new("RGB", (W, H), (245, 245, 247))
    draw = ImageDraw.Draw(canvas)

    font_path = find_font()
    def font(size, bold=True):
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    f_title = font(30)
    f_head = font(20)
    f_row = font(19)
    f_foot = font(17)

    draw.text((gap + 6, 16), "ArmorDataset - Weights x BaseLine Inference Grid", font=f_title, fill=(20, 20, 20))
    if subtitle:
        draw.text((W - 12, 24), subtitle, font=f_foot, fill=(90, 90, 90), anchor="ra")

    for c, name in enumerate(col_labels):
        x0 = label_w + gap + c * (cell_w + gap)
        draw.rectangle([x0, title_h, x0 + cell_w, title_h + head_h - gap], fill=(225, 230, 238))
        draw.text((x0 + cell_w // 2, title_h + (head_h - gap) // 2), name, font=f_head,
                  fill=(20, 20, 20), anchor="mm")

    for r, label in enumerate(row_labels):
        y0 = title_h + head_h + r * (cell_h + gap)
        draw.rectangle([gap, y0, gap + label_w - gap, y0 + cell_h], fill=(232, 236, 242))
        short = label
        if len(short) > 30:
            short = short[:29] + "..."
        draw.text((gap + 10, y0 + cell_h // 2), short, font=f_row, fill=(15, 15, 15), anchor="lm")

    for r in range(n_rows):
        for c in range(n_cols):
            x0 = label_w + gap + c * (cell_w + gap)
            y0 = title_h + head_h + r * (cell_h + gap)
            cell = cells[r][c]
            if cell is None:
                cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
            fitted = fit_cell(cell, cell_w, cell_h)
            canvas.paste(Image.fromarray(cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)), (x0, y0))
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(190, 195, 205))

    draw.text((gap + 6, H - foot_h + 8), subtitle, font=f_foot, fill=(70, 70, 70))
    draw.text((W - 12, H - foot_h + 8), "cell badge = detection count", font=f_foot,
              fill=(120, 120, 120), anchor="ra")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=False)
    return out_path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def discover_models(include=None, exclude=None):
    models = []
    root_weights = sorted([p for p in WEIGHTS_DIR.iterdir()
                           if p.is_file() and p.suffix.lower() in WEIGHT_EXTS], key=natural_key)
    for p in root_weights:
        models.append(p)
    for d in sorted([p for p in WEIGHTS_DIR.iterdir() if p.is_dir()], key=natural_key):
        for p in sorted([q for q in d.iterdir() if q.is_file() and q.suffix.lower() in WEIGHT_EXTS], key=natural_key):
            models.append(p)

    def keep(p):
        rel = str(p.relative_to(WEIGHTS_DIR))
        if include and not any(s.lower() in rel.lower() for s in include):
            return False
        if exclude and any(s.lower() in rel.lower() for s in exclude):
            return False
        return True

    return [p for p in models if keep(p)]


def model_label(path: Path):
    rel = path.relative_to(WEIGHTS_DIR)
    return str(rel)


def analyse_outputs(raw_outs, hint_nc, hint_kpt):
    """返回 (kind, payload, debug): kind in {det, heatmap, cls, unknown}"""
    quality, res, dbg = decode_detections(raw_outs, hint_nc, hint_kpt)
    dbg["quality"] = quality
    if res is not None and quality >= 0.45:
        dbg["nc"] = res["nc"]
        return "det", (quality, res), dbg

    # 特征图 -> 热力图
    for i, t in enumerate(raw_outs):
        a = np.squeeze(t)
        if a.ndim == 3 and min(a.shape) >= 4:
            dbg["fallback"] = f"feature map out{i} shape={list(np.shape(t))}"
            return "heatmap", (i, t), dbg

    # 分类向量
    for i, t in enumerate(raw_outs):
        a = np.squeeze(np.asarray(t, dtype=np.float32))
        if a.ndim == 1 and 2 <= a.size <= 200:
            dbg["fallback"] = f"vector out{i} size={a.size}"
            return "cls", (i, a), dbg

    dbg["fallback"] = "undecodable"
    return "unknown", (quality, res), dbg


def main():
    ap = argparse.ArgumentParser(description="权重批量推理 + 结果表格图")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    ap.add_argument("--cell-width", type=int, default=300, help="表格中每格宽度(px)")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"),
                    help="动态输入模型的默认分辨率")
    ap.add_argument("--models", nargs="*", default=None, help="只跑名称包含这些子串的权重")
    ap.add_argument("--exclude", nargs="*", default=None, help="跳过名称包含这些子串的权重")
    ap.add_argument("--save-overlays", action="store_true", help="额外保存逐模型叠加图")
    ap.add_argument("--max-det", type=int, default=20, help="每格最多绘制多少个目标")
    ap.add_argument("--out", type=str, default="inference_grid.png", help="表格图文件名 (输出到 Test/)")
    args = ap.parse_args()

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[x] 当前 python 环境缺少 onnxruntime, 请先执行: pip install onnxruntime")
        return 2

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS], key=natural_key)
    weights = discover_models(args.models, args.exclude)
    if not images:
        print(f"[x] {IMAGE_DIR} 中没有图片")
        return 1
    if not weights:
        print("[x] 没有找到可用权重")
        return 1

    print(f"[i] 图片 {len(images)} 张, 权重 {len(weights)} 个")
    imgs = {}
    for p in images:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            print(f"[!] 读取失败, 跳过: {p.name}")
            continue
        imgs[p.name] = im
    col_names = list(imgs.keys())
    target_w = min(640, max(480, args.cell_width * 2))

    rows, row_labels, report = [], [], {}
    t_start = time.time()

    for wpath in weights:
        label = model_label(wpath)
        row_labels.append(label)
        info = {"path": str(wpath), "size_mb": round(wpath.stat().st_size / 1e6, 2)}
        cells, per_image = [], {}
        print(f"\n>>> {label}")

        if wpath.suffix.lower() != ".onnx":
            msg = "non-ONNX weight (TensorRT engine) - skipped"
            info["error"] = msg
            print(f"    [!] {msg}")
            for name in col_names:
                base, _ = vis_base(imgs[name], target_w)
                cell = render_text_panel(base, [msg], title=label.split("/")[-1])
                cells.append(cell)
            report[label] = info
            rows.append(cells)
            continue

        try:
            model = OrtModel(wpath, tuple(args.imgsz))
            info["model"] = model.describe()
            print(f"    input={model.input_hw} ch={model.input_ch} layout={model.layout} "
                  f"provider={model.provider}")
            print(f"    outputs={[list(o.shape) for o in model.outputs]}")
        except Exception as exc:
            msg = f"load failed: {exc}"
            info["error"] = msg
            print(f"    [!] {msg}")
            for name in col_names:
                base, _ = vis_base(imgs[name], target_w)
                cell = render_text_panel(base, [msg[:70]], title=label.split("/")[-1])
                cells.append(cell)
            report[label] = info
            rows.append(cells)
            continue

        nc_hint = len(model.names) if model.names else None
        for name in col_names:
            img = imgs[name]
            scale_vis = target_w / img.shape[1]
            ratio, pad = None, None
            try:
                raw, ratio, pad = model.run(img)
                kind, payload, dbg = analyse_outputs(raw, nc_hint, model.kpt_shape)
                if kind == "det":
                    quality, res = payload
                    # 归一化输出 -> 先还原到模型输入尺度, 再统一映射回原图
                    boxes, confs, classes, kpts = postprocess(res, args.conf, args.iou, model.input_hw)
                    boxes, kpts = scale_to_original(
                        {"boxes": boxes, "norm_coords": False, "kpts": kpts,
                         "kpt_ndim": res["kpt_ndim"]}, pad, ratio)
                    vis = cv2.resize(img, None, fx=scale_vis, fy=scale_vis, interpolation=cv2.INTER_AREA)
                    b = boxes * scale_vis
                    k = kpts * scale_vis if kpts is not None else None
                    vis = draw_detections(vis, b, confs, classes, k, res["kpt_ndim"], model.names,
                                          max_det=args.max_det)
                    cell = draw_badge(vis, f"dets={len(confs)} max={confs.max():.2f}" if len(confs) else "dets=0")
                    per_image[name] = {
                        "type": "detect", "n_dets": int(len(confs)), "quality": round(float(quality), 3),
                        "max_conf": round(float(confs.max()), 3) if len(confs) else 0.0,
                        "classes": [int(c) for c in np.unique(classes)] if len(confs) else [],
                        "decode": res["kind"], "chosen": dbg.get("chosen"),
                        "kpt_shape": res.get("kpt_shape"),
                    }
                elif kind == "heatmap":
                    idx, tensor = payload
                    vis = render_heatmap(img, tensor)
                    vis = cv2.resize(vis, None, fx=scale_vis, fy=scale_vis, interpolation=cv2.INTER_AREA)
                    shape = list(np.shape(raw[idx]))
                    cell = draw_badge(vis, f"feature {shape}", color=(255, 255, 255), bg=(60, 60, 60))
                    per_image[name] = {"type": "heatmap", "out_shape": shape}
                elif kind == "cls":
                    idx, vec = payload
                    probs = softmax(vec) if (vec.max() > 1.0 or vec.min() < 0.0) else vec
                    top = np.argsort(probs)[::-1][:5]
                    lines = [f"{int(i)}: {float(probs[i]):.3f}" for i in top]
                    lines.append(f"width={np.shape(raw[idx])}")
                    vis = cv2.resize(img, None, fx=scale_vis, fy=scale_vis, interpolation=cv2.INTER_AREA)
                    cell = render_text_panel(vis, lines, title="cls top-5")
                    per_image[name] = {"type": "classify", "top5": [[int(i), float(probs[i])] for i in top]}
                else:
                    shapes = [list(o.shape) for o in raw]
                    vis = cv2.resize(img, None, fx=scale_vis, fy=scale_vis, interpolation=cv2.INTER_AREA)
                    cell = render_text_panel(vis, [f"undecodable outputs:", *[str(s) for s in shapes[:4]]],
                                             title=f"{label.split('/')[-1]}")
                    per_image[name] = {"type": "unknown", "outputs": shapes,
                                       "quality": round(float(payload[0]), 3)}
                    print(f"    [!] {name}: 无法解析输出 {shapes} (quality={payload[0]:.3f})")
                info.setdefault("decode_debug", {})[name] = dbg
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                vis = cv2.resize(img, None, fx=scale_vis, fy=scale_vis, interpolation=cv2.INTER_AREA)
                cell = render_text_panel(vis, [err[:60]], title="INFERENCE ERROR")
                per_image[name] = {"type": "error", "error": err}
                print(f"    [!] {name}: {err}")
                traceback.print_exc(limit=1)
            cells.append(cell)

        if args.save_overlays:
            odir = TEST_DIR / "overlays" / re.sub(r"[^\w.\-]+", "_", label)
            odir.mkdir(parents=True, exist_ok=True)
            for name, cell in zip(col_names, cells):
                cv2.imwrite(str(odir / f"{Path(name).stem}.jpg"), cell)

        types = [v.get("type") for v in per_image.values()]
        info["per_image"] = per_image
        info["summary"] = {
            "type": max(set(types), key=types.count) if types else None,
            "avg_dets": round(float(np.mean([v.get("n_dets", 0) for v in per_image.values()])), 2)
            if all(v.get("type") == "detect" for v in per_image.values()) and per_image else None,
        }
        print(f"    -> {info['summary']}")
        report[label] = info
        rows.append(cells)

    # 报告
    rep_path = TEST_DIR / "inference_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    subtitle = f"models={len(row_labels)}  images={len(col_names)}  conf={args.conf}  iou={args.iou}"
    grid_path = TEST_DIR / args.out
    build_grid(rows, row_labels, col_names, grid_path, cell_w=args.cell_width, subtitle=subtitle)

    # 预览小图
    prev_path = TEST_DIR / (Path(args.out).stem + "_preview" + Path(args.out).suffix)
    try:
        from PIL import Image
        with Image.open(grid_path) as im:
            w, h = im.size
            scale = min(1.0, 1800.0 / max(w, h))
            if scale < 1.0:
                im.resize((int(w * scale), int(h * scale)), Image.LANCZOS).save(prev_path)
        print(f"[√] 预览图: {prev_path}")
    except Exception as exc:
        print(f"[!] 预览图生成失败: {exc}")

    print(f"\n[√] 表格图: {grid_path}")
    print(f"[√] 报告:   {rep_path}")
    print(f"[√] 耗时:   {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
