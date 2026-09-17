#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Weights/ 下的全部 ONNX 权重对 BaseLine/ 图片做推理，生成"模型 x 图片"结果表格图。

输出:
  Test/inference_table.png   总表（行 = 模型，列 = 测试图片）
  Test/summary.csv           每个模型的检测摘要

解码说明（依据 ONNX 计算图与实测标定得到）:
  * rp_0526 系列 / shtech/SZU0526_fp32      : 输出 [N,22] = kpt(8, 输入图像素) + conf(1) + color(4) + num(9)
  * rp_0526_norm / rp_0526_split            : 同上布局，kpt 已归一化(×640 还原)，其余已过 sigmoid
  * szu2026 系列                            : 输出 [N,21] = color(4) + armor/num(9) + kpt(8)
  * szu2026_*_splitout                      : 三个输出 color(4) / armor(9) / kpt(8)
  * shtech/SKD250526                        : 输出 [N,21] = kpt(8, 网格相对) + maxconf + num(8) + conf + color
  * sp_vision_25/tiny_resnet*               : 32x32 灰度数字分类器（无检测框）
"""
import glob
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, "Weights")
BASELINE = os.path.join(ROOT, "BaseLine")
OUTDIR = os.path.join(ROOT, "Test")
os.makedirs(OUTDIR, exist_ok=True)

# 编号头 9 类；索引 8 为“大基地装甲板(LB)”（原显示为“?”，经数据侧确认后更名）
NUM_NAMES = {0: "7", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "O", 7: "B", 8: "LB"}
# shtech/SKD250526 的编号通道索引顺序与其它模型不同（按 BaseLine/RawPic 实测标定）
SKD_NUM_NAMES = {0: "B", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "O", 7: "7"}
COLOR_NAMES = {0: "B", 1: "R", 2: "P", 3: "N"}
COLOR_BGR = {0: (255, 160, 0), 1: (0, 0, 255), 2: (255, 0, 255), 3: (160, 160, 160)}

# ---------------------------------------------------------------- 模型清单
MODELS = [
    ("rp_0526 (fp16)",          "rp_0526.onnx",                          "kpt22_raw"),
    ("rp_0526_fp32",            "rp_0526_fp32.onnx",                     "kpt22_raw"),
    ("rp_0526_norm",            "rp_0526_norm.onnx",                     "kpt22_norm"),
    ("rp_0526_split",           "rp_0526_split.onnx",                    "kpt22_split"),
    ("rp_0526_probe",           "rp_0526_probe.onnx",                    "kpt22_raw"),
    ("rp_0526_probe_concat22",  "rp_0526_probe_concat22.onnx",           "kpt22_raw"),
    ("rp_0526_probe_concat23",  "rp_0526_probe_concat23.onnx",           "kpt22_raw"),
    ("rp_0526_probe_cv3",       "rp_0526_probe_cv3.onnx",                "kpt22_raw"),
    ("rp_0526_probe_model10",   "rp_0526_probe_model10.onnx",            "kpt22_raw"),
    ("shtech/SZU0526_fp32",     "shtech/SZU0526_fp32.onnx",              "kpt22_raw"),
    ("shtech/SKD250526",        "shtech/SKD250526.onnx",                 "skd"),
    ("szu2026_infantry_fp32",   "szu2026/szu2026_infantry_fp32.onnx",    "szu21"),
    ("szu2026_fp32_op11",       "szu2026/szu2026_infantry_fp32_op11.onnx", "szu21"),
    ("szu2026_fp32_op11_direct", "szu2026/szu2026_infantry_fp32_op11_direct.onnx", "szu21"),
    ("szu2026_A_kptnorm",       "szu2026/szu2026_infantry_A_kptnorm.onnx", "szu21_kptnorm"),
    ("szu2026_B_splitout",      "szu2026/szu2026_infantry_B_splitout.onnx", "szu_split"),
    ("szu2026_512x640_fp32",    "szu2026_512x640/szu2026_infantry_fp32_op11_512x640.onnx", "szu21"),
    ("szu2026_512x640_splitout", "szu2026_512x640/szu2026_infantry_B_splitout_512x640.onnx", "szu_split"),
]

CONF_THRES = 0.35
MAX_DET = 3
CAND_POOL = 120     # 送入阈值筛选的锚点池上限（先按 score 排序再逐个判 conf）


def letterbox(im, size, color=114):
    h, w = im.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size[1] - nw) / 2, (size[0] - nh) / 2
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t, b = int(round(dh - 0.1)), int(round(dh + 0.1))
    l, rr = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, t, b, l, rr, cv2.BORDER_CONSTANT, value=color), r, (l, t)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(dets, iou_thr=0.45):
    keep = []
    for d in sorted(dets, key=lambda z: -z["score"]):
        ok = True
        for k in keep:
            a, b = d["box"], k["box"]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            if ua > 0 and inter / ua > iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def build_grid(n_anchors, levels):
    """SKD250526: 网格中心 + 2*stride 的相对尺度。levels = [(gh, gw, stride), ...]"""
    grid = np.zeros((n_anchors, 2), np.float32)
    unit = np.zeros((n_anchors, 1), np.float32)
    off = 0
    for gh, gw, stride in levels:
        n = gh * gw
        yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid[off:off + n, 0] = xx.ravel() * stride + stride / 2
        grid[off:off + n, 1] = yy.ravel() * stride + stride / 2
        unit[off:off + n] = 2.0 * stride
        off += n
    return grid, unit


class Model:
    def __init__(self, name, rel, kind, conf_thres=None, max_det=None, nms_iou=0.45, cand_topk=20):
        """conf_thres / max_det / nms_iou / cand_topk 为可选覆盖。

        默认 (None / 0.45 / 20) 时完全沿用模块常量 CONF_THRES / MAX_DET 与既有
        nms 阈值、cands 截断，保证 make_table.py、make_table_roi.py、evaluate.py
        的既有行为与输出不变。
        """
        self.name, self.rel, self.kind = name, rel, kind
        self.num_names = SKD_NUM_NAMES if kind == "skd" else NUM_NAMES
        self.conf_thres = (0.30 if kind == "skd" else CONF_THRES) if conf_thres is None else float(conf_thres)
        self.max_det = MAX_DET if max_det is None else int(max_det)
        self.nms_iou = float(nms_iou)
        self.cand_topk = int(cand_topk)
        path = os.path.join(WEIGHTS, rel)
        try:                    # GPU 可用时预加载 CUDA 运行库（缺库则静默回退 CPU）
            from autolabel.gpu import preload_cuda_libraries
            preload_cuda_libraries()
        except Exception:
            pass
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _thr = os.environ.get("AUTOLABEL_ORT_THREADS")      # 多进程下限制线程数，避免 CPU 抢占
        if _thr and str(_thr).isdigit():
            so.intra_op_num_threads = int(_thr)
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.in_shape = self.sess.get_inputs()[0].shape
        self.in_type = self.sess.get_inputs()[0].type
        self.size = (int(self.in_shape[2]), int(self.in_shape[3])) if self.in_shape[1] == 3 else (32, 32)
        self.grid = None
        if kind == "skd":
            h, w = self.size
            self.grid, self.unit = build_grid((h // 8) * (w // 8) + (h // 16) * (w // 16) + (h // 32) * (w // 32),
                                              [(h // 8, w // 8, 8), (h // 16, w // 16, 16), (h // 32, w // 32, 32)])

    # -------------------------------------------------- 推理 +
    def run(self, im, roi=None):
        """返回 dict(dets=[...], cls=..., roi=..., note=str)"""
        t0 = time.time()
        if self.kind == "cls32":
            # 32x32 灰度数字分类器：输入取参考检测框裁剪出的装甲板区域（最贴近其用途）
            H, W = im.shape[:2]
            if roi is not None:
                x1, y1, x2, y2 = roi
                mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
                crop = im[max(0, int(y1 - my)):min(H, int(y2 + my)), max(0, int(x1 - mx)):min(W, int(x2 + mx))]
            else:
                crop = im
            if crop.size == 0:
                crop = im
            g = cv2.cvtColor(cv2.resize(crop, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)[None, None] / 255.0
            o = self.sess.run(None, {self.in_name: g})[0][0]
            p = np.exp(o - o.max()) / np.exp(o - o.max()).sum()
            top = np.argsort(-p)[:3]
            return {"dets": [], "cands": [], "cls": [(int(i), float(p[i])) for i in top], "roi": roi,
                    "num_names": NUM_NAMES, "note": "32x32灰度数字分类", "ms": (time.time() - t0) * 1000}

        lb, r, pad = letterbox(im, self.size)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        if self.in_type == "tensor(float16)":
            x = x.astype(np.float16)
        outs = self.sess.run(None, {self.in_name: x})
        to_orig = lambda p: (np.asarray(p, np.float32).reshape(-1, 2) - np.array(pad, np.float32)) / r

        score = num = col = None
        kpt = None
        if self.kind in ("kpt22_raw", "kpt22_norm", "kpt22_split"):
            if self.kind == "kpt22_split":
                kc, color, numv = [o[0].astype(np.float32) for o in outs]      # (N,9) (N,4) (N,9)
                kpt, conf = kc[:, :8] * 640.0, kc[:, 8]   # 该变体 kpt 已归一化(÷640)，需还原
                num, col = numv, color
            else:
                o = outs[0][0].astype(np.float32)
                kpt = o[:, :8] * (640.0 if self.kind == "kpt22_norm" else 1.0)
                conf = o[:, 8] if self.kind == "kpt22_norm" else sigmoid(o[:, 8])
                col = o[:, 9:13] if self.kind == "kpt22_norm" else sigmoid(o[:, 9:13])
                num = o[:, 13:22] if self.kind == "kpt22_norm" else sigmoid(o[:, 13:22])
            score = conf
        elif self.kind in ("szu21", "szu21_kptnorm", "szu_split"):
            # szu2026 输出为 [1, C, N]（通道在前），需转置为 [N, C]
            if self.kind == "szu_split":
                col, numv, kpt = [o[0].astype(np.float32).T for o in outs]
                num = numv
                score = num.max(axis=1)
            else:
                o = outs[0][0].astype(np.float32).T
                col, num, kpt = o[:, 0:4], o[:, 4:13], o[:, 13:21]
                if self.kind == "szu21_kptnorm":
                    kpt = kpt * 640.0
                score = num.max(axis=1)
        elif self.kind == "skd":
            # ch0-7 kpt(网格相对), ch8 maxconf, ch9-16 num(8类), ch17 红, ch18 蓝, ch19 conf
            o = outs[0][0].astype(np.float32)
            kpt = (self.grid[:, None, :] + o[:, :8].reshape(-1, 4, 2) * self.unit[:, None, :]).reshape(-1, 8).astype(np.float32)
            num = o[:, 9:17]
            col = np.stack([o[:, 18], o[:, 17], o[:, 20]], axis=1)
            score = np.maximum(o[:, 8], num.max(axis=1))
        else:
            raise ValueError(self.kind)

        dets = []
        for i in np.argsort(-score)[:CAND_POOL]:
            if score[i] < self.conf_thres:
                break
            pts = to_orig(kpt[i])
            box = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
            ci = int(np.argmax(col[i])) if col is not None else 0
            ni = int(np.argmax(num[i])) if num is not None else 0
            dets.append({"pts": pts, "box": box, "score": float(score[i]),
                         "color": ci, "num": ni,
                         "pcolor": float(col[i].max()) if col is not None else 0.0,
                         "pnum": float(num[i].max()) if num is not None else 0.0})
        cands = [{"box": d["box"], "score": d["score"]} for d in dets[:self.cand_topk]]   # ROI 判定用候选（未 NMS 截断）
        dets = self.finalize(dets)
        return {"dets": dets, "cands": cands, "cls": None, "num_names": self.num_names, "note": "",
                "ms": (time.time() - t0) * 1000}

    def finalize(self, dets, max_det=None, nms_iou=None):
        """NMS + 截断，供 run() 与外部调用共用；参数为 None 时用实例默认值。"""
        md = self.max_det if max_det is None else int(max_det)
        thr = self.nms_iou if nms_iou is None else float(nms_iou)
        return nms(dets, thr)[:md]


# ---- 统一的绘制尺寸（单位: 表格格子像素，所有模型/所有图片完全一致）----
LINE_W = 3          # 检测框线宽
DOT_R = 4           # 关键点圆点半径
LABEL_FS = 0.55     # 标签字号
LABEL_TH = 1        # 标签笔画粗细
ROI_LINE_W = 3      # 分类器 ROI 黄框线宽


def compose_tile(im, res, tile_w, tile_h, footer="", bg=32):
    """把图像等比缩放放进固定尺寸的格子，再在格子坐标系中绘制检测结果。

    因为线宽/点大小/字号都是格子空间中的常量，所以整张表格里所有格子的
    框线粗细、关键点标注大小、标签文字大小完全统一。
    """
    h, w = im.shape[:2]
    s = min((tile_w - 2) / w, (tile_h - 2) / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    small = cv2.resize(im, (nw, nh), interpolation=interp)
    tile = np.full((tile_h, tile_w, 3), bg, np.uint8)
    ox, oy = (tile_w - nw) // 2, (tile_h - nh) // 2
    tile[oy:oy + nh, ox:ox + nw] = small
    off = np.array([ox, oy], np.float32)
    num_names = res.get("num_names", NUM_NAMES)
    for d in res["dets"]:
        pts = np.asarray(d["pts"], np.float32) * s + off
        ip = pts.astype(np.int32)
        c = COLOR_BGR.get(d["color"], (0, 255, 0))
        cv2.polylines(tile, [ip], True, c, LINE_W, cv2.LINE_AA)
        for p in ip:
            cv2.circle(tile, tuple(p), DOT_R, c, -1, cv2.LINE_AA)
        txt = "%s%s %.2f" % (COLOR_NAMES.get(d["color"], "?"), num_names.get(d["num"], "?"), d["score"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FS, LABEL_TH)
        x = int(np.clip(ip[:, 0].min(), 2, max(2, tile_w - tw - 3)))
        y = int(np.clip(ip[:, 1].min() - 5, th + 4, tile_h - 4))
        cv2.putText(tile, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FS, (0, 0, 0), LABEL_TH + 3, cv2.LINE_AA)
        cv2.putText(tile, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FS, c, LABEL_TH, cv2.LINE_AA)

    if footer:
        cv2.rectangle(tile, (0, tile_h - 22), (tile_w, tile_h), (18, 18, 18), -1)
        cv2.putText(tile, footer, (8, tile_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1, cv2.LINE_AA)
    return tile


def main():
    imgs = sorted(glob.glob(os.path.join(BASELINE, "*")))
    names = [os.path.basename(p) for p in imgs]
    images = []
    for p in imgs:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        images.append(im)

    models = []
    for name, rel, kind in MODELS:
        models.append(Model(name, rel, kind))
        print("[load] %-26s %s" % (name, rel))

    ref_rois = [None] * len(names)

    TILE_W, TILE_H = 420, 300
    LABEL_W, HEADER_H, PAD = 260, 56, 6
    rows = []
    csv = []
    dump = {"images": names, "size": {nm: list(im.shape[:2]) for nm, im in zip(names, images)}, "det": {}}
    for m in models:
        tiles = []
        dump["det"][m.name] = {}
        for im, nm, roi in zip(images, names, ref_rois):
            res = m.run(im, roi)
            dump["det"][m.name][nm] = {
                "dets": [{"box": [round(v, 2) for v in d["box"]], "score": round(d["score"], 4),
                          "color": d["color"], "num": d["num"], "pts": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in d["pts"]]}
                         for d in res["dets"]],
                "cands": [{"box": [round(v, 2) for v in c["box"]], "score": round(c["score"], 4)} for c in res.get("cands", [])],
                "cls": res["cls"],
                "roi": [round(v, 2) for v in res["roi"]] if res.get("roi") else None,
                "kind": m.kind,
            }
            info = []
            for d in res["dets"]:
                info.append("%s%s:%.2f" % (COLOR_NAMES.get(d["color"], "?"),
                                           res["num_names"].get(d["num"], "?"), d["score"]))
            if res["cls"]:
                info.append("数字%d:%.2f" % (res["cls"][0][0] + 1, res["cls"][0][1]))
            txt = (" ".join(info) if info else "无检出")
            tiles.append(compose_tile(im, res, TILE_W, TILE_H, footer=txt))
            csv.append((m.name, nm, len(res["dets"]), txt, round(res["ms"], 1)))
            print("   %-26s %-9s -> %s (%.0f ms)" % (m.name, nm, txt, res["ms"]))
        rows.append(tiles)

    grid_w = LABEL_W + len(names) * (TILE_W + PAD) + PAD
    grid_h = HEADER_H + len(models) * (TILE_H + PAD) + 152
    table = np.full((grid_h, grid_w, 3), 245, np.uint8)
    cv2.putText(table, "YOLO/ONNX 检测结果对比表 (18 个检测模型)   行=模型  列=BaseLine 图片   框=装甲板四点, 标签=颜色+编号+置信度",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (20, 20, 20), 2, cv2.LINE_AA)
    for j, (nm, im) in enumerate(zip(names, images)):
        x = LABEL_W + PAD + j * (TILE_W + PAD)
        cv2.putText(table, nm, (x + 4, HEADER_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(table, "%dx%d" % (im.shape[1], im.shape[0]), (x + 4, HEADER_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    for i, (m, tiles) in enumerate(zip(models, rows)):
        y = HEADER_H + i * (TILE_H + PAD)
        fs = 0.66
        while fs > 0.38 and cv2.getTextSize("%02d %s" % (i + 1, m.name), cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] > LABEL_W - 18:
            fs -= 0.04
        cv2.putText(table, "%02d %s" % (i + 1, m.name), (8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(table, m.kind, (8, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(table, m.rel, (8, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        for j, t in enumerate(tiles):
            x = LABEL_W + PAD + j * (TILE_W + PAD)
            table[y:y + TILE_H, x:x + TILE_W] = t
            cv2.rectangle(table, (x - 1, y - 1), (x + TILE_W, y + TILE_H), (170, 170, 170), 1)

    y = HEADER_H + len(models) * (TILE_H + PAD) + 14
    notes = [
        "说明: 1) 表中为 18 个检测类 ONNX 权重；Weights/sp_vision_25/yolov5.bin 为无格式说明的裸权重(2.8MB)无法加载、sp_vision_25 下 2 个 32x32 数字分类器(无检测框)按要求已从评估中移除；",
        "      2) 检测模型输出通道布局由 ONNX 计算图常量确定(kpt/conf/color/num)，编号索引实测标定: 0->7(哨兵) 1->1 2->2 3->3 4->4 5->5 6->O(前哨站) 7->B(基地) 8->未知；颜色: B蓝 R红 P紫 N灰；",
        "      3) shtech/SKD250526 的编号通道索引顺序与其它模型不同(实测: 0->基地 6->前哨站 7->7号)，表内按其自身映射显示；",
        "      4) 框为模型回归的装甲板四点(凸包)，标签为 颜色+编号+置信度；每张图最多展示 3 个经 NMS 的检测；",
        "      5) shtech/SKD250526 为蒸馏模型，其 kpt 通道为网格相对量，按 网格中心 + raw*(2*stride) 解码，框位置相对真值略偏(11 张图平均 IoU≈0.51)。",
    ]
    for k, s in enumerate(notes):
        cv2.putText(table, s, (12, y + k * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    out = os.path.join(OUTDIR, "inference_table.png")
    cv2.imwrite(out, table)
    outj = os.path.join(OUTDIR, "inference_table.jpg")
    cv2.imwrite(outj, table, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("\n[save]", out, table.shape)
    print("[save]", outj)
    with open(os.path.join(OUTDIR, "summary.csv"), "w") as f:
        f.write("model,image,n_det,top1,ms\n")
        for row in csv:
            f.write("%s,%s,%d,%s,%.1f\n" % row)
    # 原始推理数据落盘，供 ROI 表格复用（避免重复推理）
    dump_path = os.path.join(OUTDIR, "detections.json")
    with open(dump_path, "w") as f:
        json.dump(dump, f, ensure_ascii=False)
    print("[save]", dump_path)


if __name__ == "__main__":
    main()
