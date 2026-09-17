# -*- coding: utf-8 -*-
"""颜色 / 编号分类头（裁块级小网络，部署友好）。

* 数据：training.prepare 生成的 cls/crops/{color,num}/{train,val}/<类名>/*.jpg
* 网络：tiny CNN（4 次下采样 + GAP + FC，约 0.1~0.3 M 参数），默认 arch=tiny
  - 与 sp_vision_25 的 yolov5.bin + tiny_resnet 部署链路同思路：姿态模型只出框和四点，
    颜色/编号交给独立小分类器，避免改动 YOLO26 的 e2e/RLE 头；
  - arch=yolo 时改用 ultralytics 分类模型（yolo26n-cls），适合追求精度上限。
* 类别不均衡（G 灰白仅 5.4%、N 几乎空置）：按 (N/(K*n_c))^0.5 加权。
* 产物：cls/{task}-best.pt、cls/{task}.onnx、cls/report.json

用法:
  python -m training.train_cls                        # 颜色 + 编号
  python -m training.train_cls --task color           # 只训颜色
  python -m training.train_cls --set cls.epochs=2     # 冒烟
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training.common import (ROOT, apply_overrides, kv_table, load_config, paths, warn, write_json)

# AMP：优先用新接口 torch.amp，旧版退化为 torch.cuda.amp
try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def make_scaler(device):
        return _GradScaler(device.type, enabled=(device.type == "cuda"))

    def amp_ctx(device):
        return _autocast(device.type, enabled=(device.type == "cuda"))
except Exception:                                        # pragma: no cover
    def make_scaler(device):
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def amp_ctx(device):
        return torch.cuda.amp.autocast(enabled=(device.type == "cuda"))


# --------------------------------------------------------------------------- #
# 网络
# --------------------------------------------------------------------------- #
class TinyCls(nn.Module):
    """极简 CNN：3→w→2w→4w→4w 下采样，GAP + FC。输入 size×size。"""

    def __init__(self, n_cls: int, in_ch: int = 3, width: int = 32):
        super().__init__()

        def blk(i, o, stride):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, stride, 1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            blk(in_ch, width, 2), blk(width, width * 2, 2),
            blk(width * 2, width * 4, 2), blk(width * 4, width * 4, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(width * 4, n_cls))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_loaders(root: Path, size: int, batch: int, workers: int, augment: bool):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_tr = [transforms.Resize((size, size))]
    if augment:
        tf_tr += [transforms.ColorJitter(0.25, 0.25, 0.25, 0.02), transforms.RandomApply(
            [transforms.GaussianBlur(3)], p=0.15)]
    tf_tr += [transforms.ToTensor(), norm]
    tf_va = [transforms.Resize((size, size)), transforms.ToTensor(), norm]
    ds_tr = datasets.ImageFolder(str(root / "train"), transforms.Compose(tf_tr))
    ds_va = datasets.ImageFolder(str(root / "val"), transforms.Compose(tf_va))
    # ImageFolder 会按各自目录重新编号类别；val 缺类时索引会错位（例如 N 只在 train 出现），
    # 因此把 val 的标签统一映射到 train 的 class_to_idx，映射不到的样本丢弃
    remap = {i: ds_tr.class_to_idx.get(c, -1) for i, c in enumerate(ds_va.classes)}
    missing = sorted({ds_va.classes[i] for _, i in ds_va.samples if remap.get(i, -1) < 0})
    if missing:
        n_bad = sum(1 for _, i in ds_va.samples if remap.get(i, -1) < 0)
        warn("val 中 %s 未出现在 train，丢弃 %d 个 val 样本" % ("/".join(missing), n_bad))
    ds_va.samples = [(p, remap[i]) for p, i in ds_va.samples if remap.get(i, -1) >= 0]
    ds_va.targets = [t for _, t in ds_va.samples]
    ds_va.classes = list(ds_tr.classes)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=workers,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return ds_tr, ds_va, dl_tr, dl_va


def class_weights(ds, n_cls, device, power=0.5):
    cnt = Counter(y for _, y in ds.samples)
    total = sum(cnt.values())
    w = torch.ones(n_cls, dtype=torch.float32)
    for c in range(n_cls):
        n = cnt.get(c, 0)
        w[c] = (total / (n_cls * n)) ** power if n else 1.0
    return w.to(device), {ds.classes[c]: cnt.get(c, 0) for c in range(n_cls)}


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train_tiny(task, root, cls_cfg, device, out_dir):
    size = int(cls_cfg.get("size", 64))
    epochs = int(cls_cfg.get("epochs", 60))
    batch = int(cls_cfg.get("batch", 256))
    lr = float(cls_cfg.get("lr", 3e-3))
    wd = float(cls_cfg.get("weight_decay", 5e-4))
    patience = int(cls_cfg.get("patience", 15))
    workers = int(cls_cfg.get("workers", 4))
    width = int(cls_cfg.get("width", 32))
    ds_tr, ds_va, dl_tr, dl_va = build_loaders(root, size, batch, workers, augment=True)
    n_cls = len(ds_tr.classes)
    if ds_tr.classes != ds_va.classes:
        warn("train/val 类别集合不一致，按 train 为准")
    model = TinyCls(n_cls, in_ch=3, width=width).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    weight, counts = class_weights(ds_tr, n_cls, device) if cls_cfg.get("class_weight", True) else (None, {})
    print("  [%s] train=%d val=%d 类数=%d 参数=%.0fK size=%d batch=%d"
          % (task, len(ds_tr), len(ds_va), n_cls, n_param / 1e3, size, batch))
    print("  [%s] 类别计数: %s" % (task, counts))
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = make_scaler(device)

    best_acc, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * y.numel()
            seen += y.numel()
        sched.step()
        acc_va, per_cls = evaluate(model, dl_va, device, n_cls, ds_va.classes)
        hist.append({"epoch": ep, "loss": round(tot / max(1, seen), 4), "val_acc": round(acc_va, 4)})
        print("  [%s] ep%3d  loss=%.4f  val_acc=%.4f  (%.0fs)"
              % (task, ep, tot / max(1, seen), acc_va, time.time() - t0), flush=True)
        if acc_va > best_acc:
            best_acc, best_ep, bad = acc_va, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("  [%s] 早停于 ep%d（best=%.4f @ ep%d）" % (task, ep, best_acc, best_ep))
                break

    if best_state:
        model.load_state_dict(best_state)
    acc_va, per_cls = evaluate(model, dl_va, device, n_cls, ds_va.classes)
    ckpt = out_dir / ("%s-best.pt" % task)
    torch.save({"state_dict": model.state_dict(), "classes": ds_tr.classes,
                "size": size, "width": width, "arch": "tiny", "val_acc": acc_va,
                "task": task}, ckpt)
    onnx_path = export_onnx(model, size, out_dir / ("%s.onnx" % task))
    return {"task": task, "arch": "tiny", "classes": ds_tr.classes, "size": size,
            "width": width, "n_param": n_param, "best_acc": round(best_acc, 4),
            "final_acc": round(acc_va, 4), "per_class_acc": per_cls, "history": hist,
            "ckpt": str(ckpt.relative_to(ROOT)),
            "onnx": str(onnx_path.relative_to(ROOT)) if onnx_path else None,
            "counts": counts}


def train_yolo(task, root, cls_cfg, out_dir):
    """arch=yolo：直接用 ultralytics 分类模型（精度上限更高，部署体积更大）。"""
    from ultralytics import YOLO
    size = int(cls_cfg.get("size", 64))
    name = "%s-cls" % task
    model = YOLO("yolo26n-cls.pt")
    model.train(data=str(root), imgsz=size, epochs=int(cls_cfg.get("epochs", 60)),
                batch=int(cls_cfg.get("batch", 256)) if int(cls_cfg.get("batch", 256)) <= 512 else 512,
                project=str(out_dir / "runs"), name=name, exist_ok=True, verbose=True)
    best = out_dir / "runs" / name / "weights" / "best.pt"
    onnx_path = None
    try:
        m = YOLO(str(best))
        onnx_path = Path(m.export(format="onnx", imgsz=size))
    except Exception as exc:
        warn("导出 ONNX 失败: %s" % exc)
    return {"task": task, "arch": "yolo", "size": size, "ckpt": str(best.relative_to(ROOT)),
            "onnx": str(onnx_path.relative_to(ROOT)) if onnx_path else None}


def evaluate(model, dl, device, n_cls, classes):
    model.eval()
    conf = torch.zeros(n_cls, n_cls, dtype=torch.int64)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            with amp_ctx(device):
                p = model(x).argmax(1).cpu()
            for t, q in zip(y, p):
                conf[t, q] += 1
    tot = conf.sum().item()
    acc = float(conf.diag().sum().item() / tot) if tot else 0.0
    per = {}
    for c in range(n_cls):
        s = conf[c].sum().item()
        per[classes[c]] = round(float(conf[c, c].item() / s), 4) if s else None
    model._conf = conf
    return acc, per


def export_onnx(model, size, path: Path):
    """导出 ONNX 并用 onnxruntime 校验输出一致（部署前自检）。

    torch>=2.11 默认走 dynamo 导出器，需要额外安装 onnxscript；这里优先用内置的
    旧导出器（dynamo=False），失败再回退，避免给环境强加新依赖。
    """
    try:
        import onnxruntime as ort
    except Exception:
        ort = None
    if next(model.parameters()).device.type != "cpu":
        model = model.to("cpu")                 # 导出统一在 CPU 上做，避免 device 不一致
    model.eval()
    dummy = torch.zeros(1, 3, size, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for kw in ({"dynamo": False, "opset_version": 11},
               {"opset_version": 11},
               {"dynamo": False, "opset_version": 13}):
        try:
            torch.onnx.export(model, dummy, str(path), input_names=["input"],
                              output_names=["logits"], do_constant_folding=True, **dict(kw))
            last_err = None
            break
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        warn("ONNX 导出失败（可 pip install onnxscript 后重试）: %s" % last_err)
        return None
    if ort is None:
        return path
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(dummy).numpy()
        got = sess.run(None, {"input": dummy.numpy()})[0]
        import numpy as np
        diff = float(np.abs(ref - got).max())
        print("  ONNX 校验: %s  max|Δ|=%.2e %s" % (path.name, diff, "OK" if diff < 1e-3 else "偏差偏大"))
    except Exception as exc:
        warn("ONNX 校验失败: %s" % exc)
    return path


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.train_cls")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--task", default="both", choices=("color", "num", "both"))
    ap.add_argument("--arch", default=None, choices=("tiny", "yolo"))
    args = ap.parse_args(argv)

    cfg, cfg_path = load_config(args.config)
    apply_overrides(cfg, args.set)
    work, _, _, _, cls_dir, _ = paths(cfg)
    cls_cfg = dict(cfg.get("cls") or {})
    if args.arch:
        cls_cfg["arch"] = args.arch
    arch = cls_cfg.get("arch", "tiny")
    tasks = ("color", "num") if args.task == "both" else (args.task,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("配置: %s\n设备: %s  网络: %s" % (cfg_path.relative_to(ROOT), device, arch))
    report = {"arch": arch, "device": str(device), "tasks": {}}
    for task in tasks:
        root = cls_dir / "crops" / task
        if not (root / "train").is_dir():
            warn("跳过 %s：%s 不存在（先跑 training.prepare）" % (task, root))
            continue
        print("\n== 训练 %s 分类头 ==" % ("颜色" if task == "color" else "编号"))
        res = train_tiny(task, root, cls_cfg, device, cls_dir) if arch == "tiny" \
            else train_yolo(task, root, cls_cfg, cls_dir)
        report["tasks"][task] = res
        if res.get("per_class_acc"):
            rows = [["类别", "val 准确率"]]
            for k, v in res["per_class_acc"].items():
                rows.append([k, "%.3f" % v if v is not None else "-"])
            kv_table("  %s 每类准确率：" % task, rows)
    out = write_json(cls_dir / "report.json", report)
    print("\n报告 -> %s" % out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
