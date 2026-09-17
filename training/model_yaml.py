# -*- coding: utf-8 -*-
"""按任务改写的模型结构 yaml（从已安装的 ultralytics 复制，避免手抄架构）。

只改三个顶层键，不动 backbone/head：
  nc        类别数（装甲板 = 1）
  kpt_shape [4, 2]   左上/左下/右下/右上，无可见性
  flip_idx  [3, 2, 1, 0]  水平翻转的关键点配对
文件名保持 `yolo26n-pose-armor.yaml` 这种形式，ultralytics 的 guess_model_scale()
才能认出尺度（n），否则会退化成默认 width/depth。

用法:
  python -m training.model_yaml --family yolo26n-pose --out TrainSet/armor26/models/yolo26n-pose-armor.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.common import ROOT, kv_table

# 家族名 -> (ultralytics cfg 子目录, 家族 yaml)。尺度由文件名决定。
FAMILIES = {
    "yolo26n-pose": ("26", "yolo26-pose.yaml"),
    "yolo26s-pose": ("26", "yolo26-pose.yaml"),
    "yolo26m-pose": ("26", "yolo26-pose.yaml"),
    "yolo11n-pose": ("11", "yolo11-pose.yaml"),
    "yolo11s-pose": ("11", "yolo11-pose.yaml"),
    "yolov8n-pose": ("v8", "yolov8-pose.yaml"),
    "yolov8s-pose": ("v8", "yolov8-pose.yaml"),
}


def family_yaml_path(family: str) -> Path:
    import ultralytics
    if family not in FAMILIES:
        raise SystemExit("未知模型家族: %s（可选 %s）" % (family, "/".join(FAMILIES)))
    sub, name = FAMILIES[family]
    p = Path(ultralytics.__file__).parent / "cfg" / "models" / sub / name
    if not p.exists():
        raise SystemExit("当前 ultralytics 版本没有 %s（%s）" % (name, p))
    return p


def build(family: str, out_path, nc=1, kpt_shape=(4, 2), flip_idx=(3, 2, 1, 0), verbose=True):
    """生成任务专用模型 yaml，返回输出路径。"""
    src = family_yaml_path(family)
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    before = {"nc": d.get("nc"), "kpt_shape": d.get("kpt_shape"),
              "flip_idx": d.get("flip_idx"), "end2end": d.get("end2end"),
              "reg_max": d.get("reg_max")}
    d["nc"] = int(nc)
    d["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
    d["flip_idx"] = [int(v) for v in flip_idx]
    out = Path(out_path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    text = ("# 由 training/model_yaml.py 生成（源: %s）\n"
            "# 仅改写 nc / kpt_shape / flip_idx，backbone 与 head 与官方一致\n" % src.name)
    text += yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
    out.write_text(text, encoding="utf-8")
    if verbose:
        print("模型结构: %s -> %s" % (family, out.relative_to(ROOT)))
        print("  源 %s：nc=%s kpt_shape=%s" % (src.name, before["nc"], before["kpt_shape"]))
        print("  改后：nc=%s kpt_shape=%s flip_idx=%s end2end=%s reg_max=%s"
              % (d["nc"], d["kpt_shape"], d["flip_idx"], d.get("end2end"), d.get("reg_max")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.model_yaml")
    ap.add_argument("--family", default="yolo26n-pose", choices=sorted(FAMILIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--kpt-shape", default="4,2")
    ap.add_argument("--flip-idx", default="3,2,1,0")
    args = ap.parse_args(argv)
    build(args.family, args.out, args.nc,
          tuple(int(v) for v in args.kpt_shape.split(",")),
          tuple(int(v) for v in args.flip_idx.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
