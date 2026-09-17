# -*- coding: utf-8 -*-
"""双教师推理与一致性判定。

主教师 rp_0526_fp32（角点误差 2.0% 板宽 / 编号 100%）给出主体结果；
副教师 shtech/SZU0526_fp32 做交叉校验，二者一致 → agree（高可信伪标签），
不一致 → conflict，单侧检出 → no_match + primary_only/secondary_only，
全部进入复核清单由人工决定去留（按用户确认的失败标注策略）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import make_table as MT

from . import config as C
from .geom import align_err, align_to, plate_width, poly_iou


@dataclass
class Detection:
    """单个教师的一次检出（四点顺序 [LT, LB, RB, RT]）。"""

    role: str
    model: str
    quad: np.ndarray            # (4, 2) 原图像素
    box: list                   # 外接框 xyxy
    score: float
    color: int
    num: int
    pcolor: float
    pnum: float

    @property
    def color_name(self):
        return MT.COLOR_NAMES.get(self.color, "?")

    @property
    def num_name(self):
        return MT.NUM_NAMES.get(self.num, "?")

    def info(self):
        return {"model": self.model, "score": round(self.score, 4),
                "color": int(self.color), "color_name": self.color_name,
                "num": int(self.num), "num_name": self.num_name,
                "pcolor": round(self.pcolor, 4), "pnum": round(self.pnum, 4),
                "quad": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.quad],
                "box": [round(float(v), 2) for v in self.box]}


class Teacher:
    """封装 make_table.Model（已实测标定的 letterbox + 解码路径）。"""

    def __init__(self, spec, conf_thres=C.TEACHER_CONF, max_det=C.TEACHER_MAX_DET,
                 cand_topk=C.TEACHER_CAND_TOPK, nms_iou=0.45):
        self.role = spec["role"]
        self.label = spec["label"]
        self.model = MT.Model(spec["label"], spec["rel"], spec["kind"],
                              conf_thres=conf_thres, max_det=max_det,
                              nms_iou=nms_iou, cand_topk=cand_topk)

    def infer(self, img):
        res = self.model.run(img)
        dets = []
        for d in res["dets"]:
            if d["score"] < C.DROP_CONF:
                continue
            dets.append(Detection(role=self.role, model=self.label,
                                  quad=np.asarray(d["pts"], np.float32), box=list(d["box"]),
                                  score=float(d["score"]), color=int(d["color"]), num=int(d["num"]),
                                  pcolor=float(d["pcolor"]), pnum=float(d["pnum"])))
        return dets, res


def _quantize_quad(quad):
    return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in np.asarray(quad, np.float32)]


def _make_object(primary, secondary, flags, consensus):
    """按"主教师优先、一致时取均值"的规则确定粗框与类别。"""
    quad = np.asarray(primary.quad if primary is not None else secondary.quad, np.float32)
    quad_secondary = None
    if primary is not None and secondary is not None:
        s_aligned = align_to(secondary.quad, primary.quad)
        quad_secondary = s_aligned
        if consensus == "mean":
            quad = (primary.quad + s_aligned) / 2.0
    chosen = primary if primary is not None else secondary
    pw, bar_l = plate_width(quad), plate_width(quad, vertical=True)
    return {
        "quad_coarse": quad,
        "flags": list(flags),
        "color": int(chosen.color), "num": int(chosen.num),
        "color_name": chosen.color_name, "num_name": chosen.num_name,
        "pcolor": float(chosen.pcolor), "pnum": float(chosen.pnum),
        "score_primary": None if primary is None else round(float(primary.score), 4),
        "score_secondary": None if secondary is None else round(float(secondary.score), 4),
        "plate_w": round(float(pw), 2), "bar_len": round(float(bar_l), 2),
        "primary": None if primary is None else primary.info(),
        "secondary": None if secondary is None else secondary.info(),
        # 仅供一致性报告使用的中间量（写盘前剔除）
        "_d_secondary_px": None if quad_secondary is None else
        round(float(np.mean(np.linalg.norm(quad_secondary - primary.quad, axis=1))), 3),
        "_iou_two_teachers": None if (primary is None or secondary is None)
        else round(float(poly_iou(primary.quad, secondary.quad)), 4),
    }


def match_teachers(primary_dets, secondary_dets, agree_iou=C.AGREE_IOU,
                   agree_kpt_pct=C.AGREE_KPT_PCT, match_iou=C.MATCH_IOU,
                   consensus=C.CONSENSUS, agree_color_num=C.AGREE_COLOR_NUM):
    """一对一贪心匹配 + 一致性判定，返回 object 列表（含 flags）。"""
    pairs = []
    for i, p in enumerate(primary_dets):
        for j, s in enumerate(secondary_dets):
            io = poly_iou(p.quad, s.quad)
            if io >= match_iou:
                pairs.append((io, i, j))
    pairs.sort(key=lambda t: -t[0])

    used_p, used_s, match = set(), set(), {}
    for _io, i, j in pairs:
        if i in used_p or j in used_s:
            continue
        used_p.add(i)
        used_s.add(j)
        match[i] = j

    objects = []
    for i, p in enumerate(primary_dets):
        j = match.get(i)
        if j is None:
            objects.append(_make_object(p, None, ["no_match", "primary_only"], consensus))
            continue
        s = secondary_dets[j]
        pw = plate_width(p.quad)
        d_mean, d_max = align_err(s.quad, p.quad)
        io = poly_iou(p.quad, s.quad)
        cls_ok = (p.color == s.color and p.num == s.num)
        flags = []
        if io >= agree_iou and d_mean <= agree_kpt_pct * pw and (cls_ok or not agree_color_num):
            flags.append("agree")
        else:
            flags.append("conflict")
            if io < agree_iou:
                flags.append("conflict_iou")
            if d_mean > agree_kpt_pct * pw:
                flags.append("conflict_kpt")
            if not cls_ok:
                flags.append("conflict_cls")
        obj = _make_object(p, s, flags, consensus)
        obj["_iou_two_teachers"] = round(float(io), 4)
        obj["_kpt_diff_px"] = round(float(d_mean), 3)
        obj["_kpt_diff_pct"] = round(float(100.0 * d_mean / max(pw, 1e-6)), 2)
        objects.append(obj)

    for j, s in enumerate(secondary_dets):
        if j not in used_s:
            objects.append(_make_object(None, s, ["no_match", "secondary_only"], consensus))
    return objects


def strip_internal(obj):
    """写盘/统计前去掉下划线开头的中间量。"""
    return {k: v for k, v in obj.items() if not k.startswith("_")}
