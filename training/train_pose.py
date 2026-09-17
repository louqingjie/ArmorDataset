# -*- coding: utf-8 -*-
"""训练姿态学生模型（ultralytics）。

* 结构来自 training/model_yaml.py 生成的任务 yaml（nc=1，kpt_shape=[4,2]，flip_idx=...）；
* 默认从官方 COCO 权重迁移可对齐的层（pose 头因关键点数不同会重建，属预期）；
* 配置里 ultralytics 不认识的键会被过滤并提示，避免版本差异导致报错；
* 训练结束自动调用 training.eval_kpts 评估（可用 --no-eval 关闭）。

用法:
  python -m training.train_pose                      # 主线：yolo26n-pose
  python -m training.train_pose --set pose.student=yolov8n-pose --set pose.epochs=80   # 对照
  python -m training.train_pose --set pose.epochs=2 --set data.limit=200               # 冒烟
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

from training import model_yaml
from training.common import (ROOT, apply_overrides, check_imgsz_orientation, imgsz_str,
                             load_config, parse_imgsz, paths, resolve, warn)

# 这些键是脚本自用/结构参数，不传给 ultralytics
NON_TRAIN_KEYS = {"student", "weights", "kpt_shape", "flip_idx"}


def train_kwargs(pose: dict, data_yaml: Path, runs_dir: Path, name: str):
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT as ALLOWED
        allowed = set(ALLOWED.keys())
    except Exception:                                    # 版本差异兜底
        allowed = None
    kw, dropped = {}, []
    for k, v in pose.items():
        if k in NON_TRAIN_KEYS:
            continue
        if allowed is not None and k not in allowed:
            dropped.append(k)
            continue
        kw[k] = v
    kw.update({"data": str(data_yaml), "project": str(runs_dir / "pose"),
               "name": name, "exist_ok": True, "verbose": True})
    return kw, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.train_pose")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="覆盖配置，如 --set pose.epochs=2")
    ap.add_argument("--no-pretrain", action="store_true", help="不从 COCO 权重迁移，直接随机初始化")
    ap.add_argument("--no-eval", action="store_true", help="训练后不做评估")
    args = ap.parse_args(argv)

    cfg, cfg_path = load_config(args.config)
    apply_overrides(cfg, args.set)
    work, ds_dir, models, runs, _, _ = paths(cfg)
    data_yaml = ds_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit("找不到 %s，请先运行 python -m training.prepare" % data_yaml.relative_to(ROOT))

    pose = cfg["pose"]
    pose["imgsz"] = check_imgsz_orientation(parse_imgsz(pose.get("imgsz"), 640))
    student = pose["student"]
    kpt = tuple(pose.get("kpt_shape", (4, 2)))
    flip = tuple(pose.get("flip_idx", (3, 2, 1, 0)))
    print("配置: %s" % cfg_path.relative_to(ROOT))
    print("数据: %s" % data_yaml.relative_to(ROOT))
    print("学生: %s（kpt_shape=%s, flip_idx=%s, imgsz=%s 高x宽）"
          % (student, list(kpt), list(flip), imgsz_str(pose["imgsz"])))

    yml = models / ("%s-armor.yaml" % student)
    model_yaml.build(student, yml, nc=1, kpt_shape=kpt, flip_idx=flip)
    model = YOLO(str(yml))

    weights = str(pose.get("weights") or "").strip()
    if weights and not args.no_pretrain:
        wp = resolve(weights)
        target = str(wp) if wp.exists() else weights        # 不存在则交给 ultralytics 下载
        print("迁移预训练权重: %s" % target)
        model.load(target)
    else:
        warn("未使用预训练权重：全部随机初始化（收敛更慢，仅用于对照）")

    n_param = sum(p.numel() for p in model.model.parameters())
    ks = getattr(model.model, "kpt_shape", None) or getattr(model.model, "kpt_shape", None)
    print("模型参数量: %.2f M   kpt_shape=%s" % (n_param / 1e6, ks))

    kw, dropped = train_kwargs(pose, data_yaml, runs, student)
    if dropped:
        warn("以下配置项当前 ultralytics 版本不支持，已忽略: %s" % ", ".join(dropped))
    print("开始训练：epochs=%s batch=%s imgsz=%s amp=%s device=%s"
          % (kw.get("epochs"), kw.get("batch"), kw.get("imgsz"), kw.get("amp"), kw.get("device")))
    model.train(**kw)

    best = runs / "pose" / student / "weights" / "best.pt"
    last = runs / "pose" / student / "weights" / "last.pt"
    print("\n训练完成：\n  best: %s\n  last: %s" % (best.relative_to(ROOT), last.relative_to(ROOT)))
    if cfg.get("eval", {}).get("auto", True) and not args.no_eval:
        cmd = [sys.executable, "-m", "training.eval_kpts", "--weights", str(best)]
        if args.config:
            cmd += ["--config", args.config]
        for s in args.set:
            cmd += ["--set", s]
        print("评估: %s" % " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=str(ROOT), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
