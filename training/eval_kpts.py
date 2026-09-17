# -*- coding: utf-8 -*-
"""评估姿态学生模型（口径与教师表一致，便于横向对比）。

指标：四边形 IoU、角点平均/最大误差(px 与 %板宽)、检出率；可选颜色/编号准确率。
GT 有两套：
  A) BaseLine 11 张 + Test/reference_kpt.json（与 rp_0526 / SZU0526 的表格同口径）
  B) 导出的 val 集（labels/val/*.txt 即双教师+精修得到的伪标签，量大更稳）

用法:
  python -m training.eval_kpts --weights TrainSet/armor26/runs/pose/yolo26n-pose/weights/best.pt
  python -m training.eval_kpts --weights best.pt --val-limit 100 --no-cls
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from autolabel.geom import align_err, plate_width, poly_iou
from training.common import (ROOT, apply_overrides, check_imgsz_orientation, imgsz_str,
                             kv_table, load_config, parse_imgsz, paths, resolve, warn, write_json)


# --------------------------------------------------------------------------- #
# GT 读取
# --------------------------------------------------------------------------- #
def read_label_txt(path: Path, w: int, h: int):
    """YOLO-pose 13 字段 -> [(quad_px, cls, kpt_norm)]，quad 顺序 [LT,LB,RB,RT]。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) != 13:
            continue
        a = np.asarray(v, np.float32)
        kp = a[5:].reshape(4, 2)
        out.append({"cls": int(a[0]), "quad": np.stack([kp[:, 0] * w, kp[:, 1] * h], 1),
                    "kpt_norm": kp, "box": a[1:5]})
    return out


def baseline_items(cfg, out_dir: Path):
    """BaseLine 11 张 + 参考四点（若 json 里带 color/num 也一并作为分类 GT）。"""
    ev = cfg.get("eval") or {}
    ref_path = resolve(ev.get("reference", "Test/reference_kpt.json"))
    base_dir = resolve(ev.get("baseline_dir", "BaseLine"))
    if not ref_path.exists():
        warn("参考文件不存在: %s" % ref_path)
        return []
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    items = []
    for name, rec in ref.items():
        p = base_dir / name
        if not p.exists():
            cand = list(base_dir.glob(Path(name).stem + ".*"))
            if not cand:
                continue
            p = cand[0]
        items.append({"name": name, "image": p, "quad": np.asarray(rec["quad"], np.float32),
                      "color": rec.get("color"), "num": rec.get("num")})
    return items


def val_items(cfg, dataset_dir: Path, limit: int, seed: int = 0):
    """导出 val 集：labels/val/*.txt 为 GT，图片同名（symlink）。"""
    lab_dir = dataset_dir / "labels" / "val"
    img_dir = dataset_dir / "images" / "val"
    if not lab_dir.is_dir():
        return []
    files = sorted(lab_dir.glob("*.txt"))
    if limit and len(files) > limit:
        files = random.Random(seed).sample(files, limit)
    items = []
    for f in files:
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            p = img_dir / (f.stem + ext)
            if p.exists():
                img = p
                break
        if img is None:
            continue
        im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = read_label_txt(f, w, h)
        if not gts:
            continue
        items.append({"name": f.stem, "image": img, "quads": [g["quad"] for g in gts],
                      "gt": gts, "size": (w, h)})
    return items


# --------------------------------------------------------------------------- #
# 推理与指标
# --------------------------------------------------------------------------- #
def predict_quads(model, images, imgsz, conf, iou):
    """返回 {图片路径: [quad_px, ...]}；ultralytics 的 keypoints.xy 已是像素坐标。"""
    res = {}
    for r in model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=conf, iou=iou,
                           verbose=False, stream=False):
        path = str(r.path)
        quads = []
        if r.keypoints is not None and r.keypoints.xy is not None:
            kp = r.keypoints.xy.cpu().numpy()
            for q in kp:
                if q.shape[0] == 4 and np.isfinite(q).all():
                    quads.append(q.astype(np.float32))
        res[path] = quads
    return res


def match_metrics(gt_quads, pred_quads, iou_thr=0.3):
    """每个 GT 找 IoU 最大的预测；返回 (iou, 角点误差 px, 是否命中, 配对索引)。"""
    rows = []
    for gq in gt_quads:
        best = (0.0, None, -1)
        for j, pq in enumerate(pred_quads):
            v = poly_iou(pq, gq)
            if v > best[0]:
                best = (v, pq, j)
        iou_v, pq, j = best
        if iou_v < iou_thr or pq is None:
            rows.append({"iou": iou_v, "err": None, "hit": False, "pred": None})
        else:
            err, _ = align_err(pq, gq)
            rows.append({"iou": iou_v, "err": err, "hit": True, "pred": pq})
    return rows


def summarize(name, rows, n_pred, n_gt):
    hits = [r for r in rows if r["hit"]]
    ious = np.array([r["iou"] for r in rows], np.float32)
    errs = np.array([r["err"] for r in hits], np.float32)
    return {
        "set": name, "n_gt": int(n_gt), "n_pred": int(n_pred), "n_hit": len(hits),
        "hit_rate": round(len(hits) / max(1, n_gt), 4),
        "iou_mean": round(float(ious.mean()), 4) if len(ious) else 0.0,
        "iou_min": round(float(ious.min()), 4) if len(ious) else 0.0,
        "kpt_err_mean": round(float(errs.mean()), 3) if len(errs) else None,
        "kpt_err_max": round(float(errs.max()), 3) if len(errs) else None,
        "iou_hit_mean": round(float(np.mean([r["iou"] for r in hits])), 4) if hits else 0.0,
        "err_pct_plate": None,
        "fp": int(max(0, n_pred - len(hits))),
    }


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.eval_kpts")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", default=None, help="整数或 高,宽（如 512,640）")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--no-cls", action="store_true", help="跳过颜色/编号准确率")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg, cfg_path = load_config(args.config)
    apply_overrides(cfg, args.set)
    work, ds_dir, _, _, cls_dir, ev_dir = paths(cfg)
    ev = cfg.get("eval") or {}
    imgsz = check_imgsz_orientation(
        parse_imgsz(args.imgsz if args.imgsz else (cfg.get("pose") or {}).get("imgsz"), 640))
    conf = args.conf if args.conf is not None else float(ev.get("conf", 0.25))
    iou = args.iou if args.iou is not None else float(ev.get("iou", 0.6))
    val_limit = args.val_limit if args.val_limit is not None else int(ev.get("val_limit", 300))

    wp = resolve(args.weights)
    if not wp.exists():
        raise SystemExit("权重不存在: %s" % wp)
    from ultralytics import YOLO
    model = YOLO(str(wp))
    print("评估权重: %s" % wp.relative_to(ROOT))
    print("配置: %s   imgsz=%s(高x宽) conf=%.2f iou=%.2f"
          % (cfg_path.relative_to(ROOT), imgsz_str(imgsz), conf, iou))

    report = {"weights": str(wp.relative_to(ROOT)), "imgsz": imgsz, "conf": conf, "iou": iou,
              "sets": {}, "time": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}

    # ---- A. BaseLine（与教师表同口径）----
    out_dir = Path(cfg["data"]["source_out"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    base = baseline_items(cfg, out_dir)
    if base:
        preds = predict_quads(model, [b["image"] for b in base], imgsz, conf, iou)
        rows, n_pred, per_img = [], 0, []
        for b in base:
            pq = preds.get(str(b["image"]), [])
            n_pred += len(pq)
            m = match_metrics([b["quad"]], pq)
            rows += m
            best = m[0]
            pw = plate_width(b["quad"])
            per_img.append({"name": b["name"], "iou": round(float(best["iou"]), 4),
                            "err_px": None if best["err"] is None else round(float(best["err"]), 2),
                            "err_pct": None if best["err"] is None else round(100 * float(best["err"]) / pw, 2)})
        s = summarize("BaseLine(11)", rows, n_pred, len(base))
        errs = [r["err"] for r in rows if r["hit"]]
        pcts = []
        for b, r in zip(base, rows):
            if r["err"] is not None:
                pcts.append(100 * float(r["err"]) / plate_width(b["quad"]))
        s["err_pct_plate"] = round(float(np.mean(pcts)), 2) if pcts else None
        s["images"] = per_img
        report["sets"]["baseline"] = s
        rows_t = [["图片", "IoU", "角点误差 px", "%板宽"]]
        for it in per_img:
            rows_t.append([it["name"], "%.3f" % it["iou"],
                           "-" if it["err_px"] is None else "%.2f" % it["err_px"],
                           "-" if it["err_pct"] is None else "%.2f%%" % it["err_pct"]])
        kv_table("BaseLine 逐图：", rows_t)

    # ---- B. 导出 val 集（伪标签 GT，样本量大）----
    if val_limit != 0:
        vals = val_items(cfg, ds_dir, val_limit)
        if vals:
            preds = predict_quads(model, [v["image"] for v in vals], imgsz, conf, iou)
            rows, n_pred, n_gt = [], 0, 0
            for v in vals:
                pq = preds.get(str(v["image"]), [])
                n_pred += len(pq)
                n_gt += len(v["quads"])
                rows += match_metrics(v["quads"], pq)
            s = summarize("val(%d 张)" % len(vals), rows, n_pred, n_gt)
            hits = [r for r in rows if r["hit"]]
            pcts = []
            idx = 0
            for v in vals:
                for g in v["quads"]:
                    r = rows[idx]
                    idx += 1
                    if r["err"] is not None:
                        pcts.append(100 * float(r["err"]) / plate_width(g))
            s["err_pct_plate"] = round(float(np.mean(pcts)), 2) if pcts else None
            report["sets"]["val"] = s

    # ---- 汇总表 ----
    rows_t = [["数据集", "GT", "命中", "命中率", "IoU均值", "角点误差px", "%板宽", "误检"]]
    for k, s in report["sets"].items():
        rows_t.append([s["set"], s["n_gt"], s["n_hit"], "%.1f%%" % (100 * s["hit_rate"]),
                       "%.3f" % s["iou_hit_mean"],
                       "-" if s["kpt_err_mean"] is None else "%.2f" % s["kpt_err_mean"],
                       "-" if s["err_pct_plate"] is None else "%.2f%%" % s["err_pct_plate"],
                       s["fp"]])
    kv_table("\n汇总（IoU/误差只统计命中目标）：", rows_t)
    print("\n参照教师（同一套指标）：rp_0526_fp32 = 3.31px / 2.0%板宽，SZU0526 = 3.32px / 2.0%板宽")

    # ---- C. 颜色/编号分类头准确率（可选）----
    if not args.no_cls and report["sets"].get("val"):
        cls_report = eval_cls(cfg, cls_dir, report, model, imgsz, conf, iou)
        if cls_report:
            report["cls"] = cls_report

    out = Path(args.out) if args.out else (ev_dir / ("eval_%s.json" % wp.stem))
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, report)
    print("报告 -> %s" % out.relative_to(ROOT))
    return 0


def eval_cls(cfg, cls_dir: Path, report, model, imgsz, conf, iou):
    """用预测框裁块送进颜色/编号分类头，与 val 集 GT（meta 里的 color/num）比准确率。"""
    import torch
    from torchvision import transforms
    from training.train_cls import TinyCls

    ckpts = {}
    for task in ("color", "num"):
        p = cls_dir / ("%s-best.pt" % task)
        if p.exists():
            ckpts[task] = p
    if not ckpts:
        return None
    print("\n颜色/编号分类头：")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for task, p in ckpts.items():
        blob = torch.load(str(p), map_location="cpu")
        net = TinyCls(len(blob["classes"]), 3, int(blob.get("width", 32)))
        net.load_state_dict(blob["state_dict"])
        net.eval().to(device)
        models[task] = (net, blob["classes"], int(blob["size"]))
        print("  %-5s %s（val_acc=%s，%d 类）" % (task, p.name, blob.get("val_acc"), len(blob["classes"])))

    # 分类头的 GT 直接来自 meta：按导出的 val key 反查（key 里的 '/' 在导出时变成 '__'）
    from webui import api as W
    out_dir = resolve(cfg["data"]["source_out"])
    vals = val_items(cfg, paths(cfg)[1], 200)
    if not vals:
        return None
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.ToTensor(), norm])
    stats = {t: {"n": 0, "ok": 0} for t in models}
    for v in vals:
        key = str(v["name"]).replace("__", "/")
        try:
            meta = W.read_meta(out_dir, key)
            img_path = W.resolve_image_path(meta, out_dir)
        except Exception:
            continue
        im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        h, w = im.shape[:2]
        # 用预测四点（而非 GT）裁块，反映真实链路
        pq = predict_quads(model, [img_path], imgsz, conf, iou).get(str(img_path), [])
        quads = pq if pq else [g["quad"] for g in v["gt"]]
        for obj, q in zip(meta.get("objects") or [], quads):
            if obj.get("deprecated"):
                continue
            gq = np.asarray(obj.get("quad_final_px"), np.float32) if obj.get("quad_final_px") else None
            if gq is None or poly_iou(q, gq) < 0.5:      # 预测与 GT 对不上就不计
                continue
            xs, ys = q[:, 0], q[:, 1]
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            bw, bh = max(4.0, (xs.max() - xs.min()) * 1.25), max(4.0, (ys.max() - ys.min()) * 1.25)
            x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            x2, y2 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
            crop = im[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            for task, (net, classes, size) in models.items():
                tile = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                x = tf(rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    idx = int(net(x).argmax(1).item())
                gt = obj.get("color_name") if task == "color" else obj.get("num_name")
                stats[task]["n"] += 1
                stats[task]["ok"] += int(classes[idx] == gt)
    res = {}
    for task, s in stats.items():
        acc = s["ok"] / s["n"] if s["n"] else None
        res[task] = {"n": s["n"], "ok": s["ok"], "acc": None if acc is None else round(acc, 4)}
        print("  %-5s 用预测框裁块: n=%d  准确率=%s"
              % (task, s["n"], "-" if acc is None else "%.4f" % acc))
    return res


if __name__ == "__main__":
    raise SystemExit(main())
