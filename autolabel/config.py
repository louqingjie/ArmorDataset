# -*- coding: utf-8 -*-
"""自动标注流水线的配置与常量（单一数据源）。

路径、教师模型清单、阈值、类别映射集中在此处；类别名与颜色名直接引用
make_table.py 中已实测标定的映射表，避免各处重复定义产生分歧。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent        # /home/wpie/ArmorDataset
WEIGHTS_DIR = ROOT / "Weights"
DEFAULT_IMAGES = ROOT / "RawPic" / "SUM"
DEFAULT_OUT = ROOT / "AutoLabel"
BASELINE_DIR = ROOT / "BaseLine"

# ------------------------------------------------------------------ 教师模型
# 依据 BaseLine 11 张评估：rp_0526_fp32 角点误差 3.31px(2.0% 板宽)、编号 100%；
# shtech/SZU0526_fp32 3.32px(2.0%)、互差 0.07px，适合做一致性交叉校验。
TEACHERS = (
    {"role": "primary", "label": "rp_0526_fp32", "rel": "rp_0526_fp32.onnx", "kind": "kpt22_raw"},
    {"role": "secondary", "label": "shtech/SZU0526_fp32", "rel": "shtech/SZU0526_fp32.onnx",
     "kind": "kpt22_raw"},
)

# 教师推理取全量候选（置信度阈值在流水线内统一施加，低于默认 0.35/3 个截断）
TEACHER_CONF = 0.25
TEACHER_MAX_DET = 20
TEACHER_CAND_TOPK = 40

# 低于该置信度的检出直接丢弃（不算目标，不进复核清单）
DROP_CONF = 0.25

# ------------------------------------------------------------ 一致性判定门限
AGREE_IOU = 0.85           # 双教师四边形 IoU ≥ 该值
AGREE_KPT_PCT = 0.05       # 且角点平均差 ≤ 5% 板宽
AGREE_COLOR_NUM = True     # 且颜色/编号一致
MATCH_IOU = 0.10           # 匹配阶段的最小 IoU（仅用于找对应关系）

# ------------------------------------------------------------------ ROI 裁剪
ROI_EXPAND = 1.5           # 以四点外接框为中心放大倍数（复用 make_table_roi.py 公式）
# ROI 最小边长：装甲板窄边较短（如 33x15px 的板，1.5x 后高度仅 23px），门限取
# make_reference.sp_refine 内部同款 8px，实测 11 张 BaseLine 全部精修成功。
ROI_MIN_SIDE = 8

# ---------------------------------------------------------------- 来源分组（防"机器预习"）
# 每张图的 meta json 里记录 source_group：它来自哪一批采集/比赛，避免训练集与验证集同源。
# 由路径推导（标注时写入，之后不再随目录结构变化）：
#   RawPic/SUM/0001.jpg      -> "SUM"
#   RawPic/SUM/val/x.jpg     -> "SUM_val"      ← 用户自留的跨比赛/跨相机 holdout
#   BaseLine/B2.png          -> "BaseLine"     ← 冒烟/回归用
IMAGE_ROOT_NAMES = ("RawPic", "BaseLine")
HOLDOUT_GROUPS = ("SUM_val",)      # 视为验证集来源的分组：导出时强制隔离、绝不进 train


def source_group_of_path(path):
    """从路径推导来源分组：数据批次[_子批次]。

    兼容相对路径与绝对路径（meta 里的 path 是绝对路径），锚定最后一个 RawPic/BaseLine：
      RawPic/SUM/0001.jpg                  -> SUM
      RawPic/SUM/val/outpost592.jpg         -> SUM_val     ← holdout
      /home/xxx/ArmorDataset/BaseLine/B2.png-> BaseLine
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if len(parts) > 1 and "." in parts[-1]:
        parts = parts[:-1]                                  # 去掉文件名
    idx = None
    for i, seg in enumerate(parts):
        if seg in IMAGE_ROOT_NAMES:
            idx = i                                         # 取最后一个（绝对路径前缀里可能有同名目录）
    if idx is not None:
        root, parts = parts[idx], parts[idx + 1:]
        if not parts:
            return root                                     # BaseLine/xxx.png -> BaseLine
    if not parts:
        return "unknown"
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def is_holdout(group):
    """该来源分组是否属于验证集来源（可配置扩展，如以后加 "RM2026_zone2"）。"""
    return str(group) in set(HOLDOUT_GROUPS)


# ---------------------------------------------------------------- 精修后校验
REFINE_IOU_GUARD = 0.30    # 精修四点 vs 粗框 IoU 下限（防止跳到错误灯条）
REFINE_SIZE_LO = 0.5       # 精修板宽 / 粗框板宽 允许区间
REFINE_SIZE_HI = 2.0

# 二值化阈值扫描：两阶段，先跑 make_reference 原始档，全部失败才用补充档。
# 试点实测：板宽≥30px 的精修失败目标用原始 7 档 0/40 通过，补充档可救回约 30%；
# 单阶段直接放宽会让原本成功的图（BO/R2）选中更差的灯条对，故必须分两阶段。
REFINE_THRESHOLDS_BASE = (80, 100, 120, 140, 160, 180, 200)     # make_reference 原实现
REFINE_THRESHOLDS_EXTRA = (20, 40, 60, 220, 240)                # 仅在第一阶段失败后使用

# 板宽小于该值时跳过传统视觉精修（灯条仅数像素宽，阈值法不可靠；教师角点
# 相对误差仍只有 2% 板宽，直接采用并记 refine_skipped，不产生人工复核负担）
MIN_REFINE_PLATE_W = 24.0

# 精修后板宽低于该像素值时标记 tiny（远距离小目标，角点不可靠，进入复核清单）
TINY_PLATE_W = 12.0

# ------------------------------------------------------------------ 标签输出
CLASS_MODE = "single"      # single: cls=0 单类；merged: cls = color*9 + num
KPT_ORDER = ("LT", "LB", "RB", "RT")   # 左上, 左下, 右下, 右上（本工程约定）

# ---------------------------------------------------------------- 其他参数
CONSENSUS = "mean"         # 共识四点: mean(两教师均值) | primary(主教师)
PANEL_LIMIT = 60           # 默认生成的抽检面板数量上限
SHEET_COLS = 5
