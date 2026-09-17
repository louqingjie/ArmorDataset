# -*- coding: utf-8 -*-
"""自动标注流水线 CLI（主入口）。

用法示例:
  # 试点 1000 张（分层采样，默认）
  python -m autolabel.run --limit 1000

  # BaseLine 11 张冒烟
  python -m autolabel.run --images BaseLine --limit 0 --no-val --out AutoLabel_smoke

  # 全量续跑（已完成的图按 meta json 跳过）
  python -m autolabel.run --limit 0 --workers 4

  # 只重算统计/报告/面板
  python -m autolabel.run --stats-only

流程: 双教师全图推理 → 双教师一致性判定 → 1.5× ROI 裁剪 → 传统视觉精修角点
      → YOLO-pose txt + meta json 原子写 → 复核清单/面板/统计报告
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import labelio, stats as stats_mod, viz
from .refine import refine_objects
from .teachers import Teacher, match_teachers

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE_BUCKETS = ((300, "small"), (800, "medium"), (10 ** 9, "large"))


# --------------------------------------------------------------------------- #
# 图像读取与采样
# --------------------------------------------------------------------------- #
def read_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError("无法读取图像: %s" % path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    elif im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def list_images(images_dir, include_val=True):
    root = Path(images_dir)
    out = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    val = root / "val"
    if include_val and val.is_dir():
        for p in sorted(val.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.append(p)
    return out


def _size_bucket(path):
    """读图像头拿尺寸（PIL 只读头，快）；失败返回 None。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        m = min(w, h)
        for lim, name in SIZE_BUCKETS:
            if m < lim:
                return name
    except Exception:
        return None
    return "large"


def sample_images(paths, limit, mode="stratified", seed=0, log=print):
    """分层采样：按 (目录, 尺寸桶) 分层按比例分配，层内随机；不足则随机补足。"""
    if limit <= 0 or limit >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    if mode == "first":
        return list(paths[:limit])
    if mode == "random":
        return rng.sample(list(paths), limit)

    groups = {}
    unknown = []
    for p in paths:
        b = _size_bucket(p)
        if b is None:
            unknown.append(p)
            continue
        groups.setdefault((p.parent.name, b), []).append(p)
    if not groups or len(unknown) > 0.3 * len(paths):       # 尺寸信息不可靠 -> 纯随机
        log("[sample] 尺寸分层不可用(未知 %d/%d)，退化为随机采样" % (len(unknown), len(paths)))
        return rng.sample(list(paths), limit)

    picked, total = [], sum(len(v) for v in groups.values())
    for key in sorted(groups):
        v = groups[key]
        quota = int(round(limit * len(v) / total))
        quota = min(quota, len(v))
        picked += rng.sample(v, quota)
    pool = [p for p in paths if p not in set(picked)]
    rng.shuffle(pool)
    picked += pool[:max(0, limit - len(picked))]
    log("[sample] 分层采样 %d/%d 张: %s" % (
        len(picked), len(paths),
        ", ".join("%s/%s=%d" % (k[0] or ".", k[1], len(groups[k])) for k in sorted(groups))))
    return picked[:limit]


# --------------------------------------------------------------------------- #
# 单图处理
# --------------------------------------------------------------------------- #
def rel_path(path, images_root):
    """相对输入目录的路径（含扩展名），用于唯一标识与断点续跑匹配。"""
    p = Path(path)
    try:
        rel = p.relative_to(Path(images_root))
    except ValueError:
        rel = Path(p.name)
    return str(rel).replace("\\", "/")


def assign_keys(paths, images_root):
    """为每张图分配唯一标签 key（相对路径去扩展名）。

    同目录同 stem 不同扩展名（如 val/1166.jpg 与 val/1166.png，全库 14 例）会冲突，
    这类样本追加扩展名后缀（val/1166_png）以保证稳定且可复现。
    """
    from collections import Counter as _Counter
    rels = {str(p): rel_path(p, images_root) for p in paths}
    raw = {s: r.rsplit(".", 1)[0] if "." in r.rsplit("/", 1)[-1] else r for s, r in rels.items()}
    cnt = _Counter(raw.values())
    keys, used = {}, set()
    for s in sorted(rels):
        k = raw[s]
        if cnt[k] > 1:
            k = "%s_%s" % (k, rels[s].rsplit(".", 1)[-1].lower() if "." in rels[s] else "dup")
        n = 2
        base = k
        while k in used:
            k = "%s_%d" % (base, n)
            n += 1
        used.add(k)
        keys[s] = k
    return keys


def process_one(teachers, item, cfg, log=None):
    """处理单张图并落盘，返回进度记录 dict。item = (path, key)。"""
    t0 = time.time()
    path, key = item
    p = Path(path)
    rel = rel_path(p, cfg["images_root"])
    img = read_image(path)
    h, w = img.shape[:2]
    t_read = time.time()

    primary_dets, _ = teachers["primary"].infer(img)
    secondary_dets = []
    if "secondary" in teachers:
        secondary_dets, _ = teachers["secondary"].infer(img)
    t_infer = time.time()

    objects = match_teachers(primary_dets, secondary_dets, agree_iou=cfg["agree_iou"],
                             agree_kpt_pct=cfg["agree_kpt_pct"], consensus=cfg["consensus"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects, refine_stat = refine_objects(gray, objects, expand=cfg["expand"],
                                         min_side=cfg["min_roi_side"],
                                         min_plate_w=cfg["min_refine_plate_w"],
                                         extra_thresholds=cfg["refine_extra_thresholds"])
    t_refine = time.time()

    rec = labelio.write_image_labels(cfg["out"], key, w, h, objects, image_name=p.name,
                                     image_path=str(p), rel=rel, teachers=cfg["teacher_labels"],
                                     run_config=cfg["meta_config"], mode=cfg["class_mode"])
    ms = (time.time() - t0) * 1000.0
    return {
        "image": key, "rel": rel, "status": "ok", "ms": round(ms, 1),
        "ms_read": round((t_read - t0) * 1000, 1),
        "ms_infer": round((t_infer - t_read) * 1000, 1),
        "ms_refine": round((t_refine - t_infer) * 1000, 1),
        "n_obj": rec["n_obj"], "n_review": rec["n_review"], "n_refined": refine_stat["n_refined"],
        "flags": sorted(set(rec["flags"])),
    }


def worker_main(chunk, cfg, wid):
    """子进程入口：独立创建 ORT 会话，逐图处理，进度写 progress.w<id>.jsonl。"""
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    prog = out / ("progress.w%d.jsonl" % wid)
    teachers = {"primary": Teacher(C.TEACHERS[0], conf_thres=cfg["conf"],
                                   max_det=cfg["max_det"], cand_topk=C.TEACHER_CAND_TOPK)}
    if len(C.TEACHERS) > 1:
        teachers["secondary"] = Teacher(C.TEACHERS[1], conf_thres=cfg["conf"],
                                        max_det=cfg["max_det"], cand_topk=C.TEACHER_CAND_TOPK)
    with open(prog, "a", encoding="utf-8") as f:
        for item in chunk:
            try:
                rec = process_one(teachers, item, cfg)
            except Exception as exc:                       # 单图异常隔离，不中断整批
                path, key = item
                rec = {"image": key, "rel": rel_path(path, cfg["images_root"]),
                       "status": "error", "ms": 0.0,
                       "error": "%s: %s" % (type(exc).__name__, exc),
                       "trace": traceback.format_exc(limit=3).splitlines()[-1][:200]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return str(prog)


def _chunkify(seq, n):
    n = max(1, n)
    k, m = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        out.append(seq[i:i + size])
        i += size
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# 面板 / 报告
# --------------------------------------------------------------------------- #
def build_panels(out_dir, images_dir, limit, cols, tag, prefer_review=True, log=print):
    """抽检面板与拼版图；prefer_review=True 时优先选含待复核目标的图片。"""
    out_dir, images_dir = Path(out_dir), Path(images_dir)
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_dir / "meta").rglob("*.json"))]
    if not metas:
        return None
    with_obj = [m for m in metas if m["objects"]]
    review = [m for m in with_obj if m["n_review"] > 0]
    others = [m for m in with_obj if m["n_review"] == 0]
    picked = (review + others) if prefer_review else (others + review)
    picked = picked[:limit]
    panels = []
    for m in picked:
        p = Path(m["path"]) if m.get("path") else (images_dir / (m["image"] + ".jpg"))
        if not p.exists():
            cands = list(images_dir.rglob(m["image"] + ".*"))
            if not cands:
                continue
            p = cands[0]
        try:
            img = read_image(p)
        except Exception:
            continue
        objects = [_panel_object(o) for o in m["objects"]]
        panels.append(viz.image_panel(img, m["image"], objects))
    if not panels:
        return None
    notes = (
        "图例: 灰细框=双教师共识粗框; 彩色粗框=最终标签四点(0左上/1左下/2右下/3右上，橙=精修未通过回退教师); 右上红点R=待人工复核",
        "流程: 教师全图推理(单次) -> 双教师一致性(IoU≥%.2f,角点差≤%.0f%%板宽,颜色编号一致) -> %.2f× ROI 裁剪 -> sp_vision_25 传统视觉角点精修 -> 门限校验" % (
            C.AGREE_IOU, 100 * C.AGREE_KPT_PCT, C.ROI_EXPAND),
    )
    grid = viz.sheet(panels, cols=cols, title="自动标注抽检面板 (%s): 行=拼版 列=图片" % tag, notes=notes)
    path = viz.save(grid, out_dir / "figures" / ("panels_%s.png" % tag))
    log("[panel] %d 张 -> %s" % (len(panels), path))
    return path


def _panel_object(o):
    """meta 结构 -> viz.panel 需要的对象。"""
    return {"quad_coarse": np.asarray(o["quad_coarse_px"], np.float32),
            "quad_final": np.asarray(o["quad_final_px"], np.float32),
            "color": o.get("color", 0), "color_name": o.get("color_name", "?"),
            "num_name": o.get("num_name", "?"), "source": o.get("source"),
            "score_primary": o.get("score_primary"), "score_secondary": o.get("score_secondary"),
            "review": o.get("review"), "flags": o.get("flags", [])}


REVIEW_COLS = ["image", "obj", "source", "flags", "reason", "color", "num",
               "score_p", "score_s", "plate_w", "iou2", "kpt_diff_pct", "d_refine_px"]


def _review_line(r, cols=REVIEW_COLS):
    return ",".join("" if r.get(c) is None else str(r.get(c)).replace(",", "|") for c in cols)


def write_review_list(out_dir, log=print):
    rows = labelio.review_rows(out_dir)
    if not rows:
        log("[review] 无需人工复核的目标")
        return None
    path = Path(out_dir) / "review" / "review_list.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(REVIEW_COLS)] + [_review_line(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("[review] %d 个目标待人工复核 -> %s" % (len(rows), path))
    return path


def update_review_list(out_dir, keys, log=print):
    """增量更新复核清单：只重算给定 key 的行，其余原样保留。

    全量重写要读 21k 个 meta（约 0.8 s），交互式保存/自动保存每秒都可能触发，
    因此这里按 key 局部刷新（毫秒级）。keys 为空则等价于不做任何事。
    """
    keys = [str(k) for k in keys if str(k)]
    if not keys:
        return None
    path = Path(out_dir) / "review" / "review_list.csv"
    keep = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
            if ln.strip() and ln.split(",", 1)[0] not in keys:
                keep.append(ln)
    rows = labelio.review_rows(out_dir, keys=keys)
    lines = [",".join(REVIEW_COLS)] + keep + [_review_line(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("[review] 增量更新 %d 个 key（清单共 %d 行）" % (len(keys), len(lines) - 1))
    return path


def merge_progress(out_dir, log=print):
    """合并各 worker 的进度文件到 progress.jsonl。"""
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob("progress.w*.jsonl"))
    if not parts:
        return
    recs = []
    for p in parts:
        recs += [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(out_dir / "progress.jsonl", "a", encoding="utf-8") as f:
        for l in recs:
            f.write(l + "\n")
    for p in parts:
        p.unlink()
    log("[progress] 合并 %d 条记录 -> progress.jsonl" % len(recs))


def report_errors(out_dir, log=print):
    """批量结束打印失败图清单并落盘 errors.log。"""
    path = Path(out_dir) / "progress.jsonl"
    if not path.exists():
        return []
    errs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("status") != "ok":
            errs.append(rec)
    if errs:
        p = Path(out_dir) / "errors.log"
        p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in errs) + "\n", encoding="utf-8")
        log("[errors] %d 张处理失败 -> %s" % (len(errs), p))
        for e in errs[:10]:
            log("   %s: %s" % (e.get("image"), e.get("error")))
    return errs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m autolabel.run",
                                 description="装甲板自动标注（蒸馏伪标签）流水线")
    ap.add_argument("--images", default=str(C.DEFAULT_IMAGES), help="输入图片目录")
    ap.add_argument("--out", default=str(C.DEFAULT_OUT), help="输出目录（AutoLabel/）")
    ap.add_argument("--limit", type=int, default=1000, help="处理图片数；0=全部")
    ap.add_argument("--sample", choices=("stratified", "random", "first"), default="stratified")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-val", action="store_true", help="不含 images/val 子目录")
    ap.add_argument("--class-mode", choices=("single", "merged"), default=C.CLASS_MODE)
    ap.add_argument("--expand", type=float, default=C.ROI_EXPAND, help="精修 ROI 放大倍数")
    ap.add_argument("--min-roi-side", type=int, default=C.ROI_MIN_SIDE)
    ap.add_argument("--conf", type=float, default=C.TEACHER_CONF, help="教师置信度阈值")
    ap.add_argument("--max-det", type=int, default=C.TEACHER_MAX_DET)
    ap.add_argument("--min-refine-plate-w", type=float, default=C.MIN_REFINE_PLATE_W,
                    help="板宽小于该值时跳过传统视觉精修（0=不跳过）")
    ap.add_argument("--refine-extra-thresholds", default=",".join(str(v) for v in C.REFINE_THRESHOLDS_EXTRA),
                    help="精修第二阶段补充阈值（第一阶段固定用 make_reference 的 7 档）；空=不启用第二阶段")
    ap.add_argument("--agree-iou", type=float, default=C.AGREE_IOU)
    ap.add_argument("--agree-kpt-pct", type=float, default=C.AGREE_KPT_PCT)
    ap.add_argument("--consensus", choices=("mean", "primary"), default=C.CONSENSUS)
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（spawn，各自建会话）")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有 meta，全部重跑")
    ap.add_argument("--panels", type=int, default=C.PANEL_LIMIT, help="抽检面板数量上限；0=不生成")
    ap.add_argument("--sheet-cols", type=int, default=C.SHEET_COLS)
    ap.add_argument("--stats-only", action="store_true", help="只重算统计/报告/面板")
    return ap


def parse_thresholds(text):
    if not text:
        return []
    return [int(t) for t in str(text).replace(" ", "").split(",") if t]


def cfg_from_args(args):
    thrs = parse_thresholds(args.refine_extra_thresholds)
    return {
        "out": str(args.out), "class_mode": args.class_mode, "expand": args.expand,
        "images_root": str(args.images),
        "min_roi_side": args.min_roi_side, "conf": args.conf, "max_det": args.max_det,
        "min_refine_plate_w": args.min_refine_plate_w, "refine_extra_thresholds": thrs,
        "agree_iou": args.agree_iou, "agree_kpt_pct": args.agree_kpt_pct,
        "consensus": args.consensus,
        "teacher_labels": [t["label"] for t in C.TEACHERS],
        "meta_config": {"expand": args.expand, "min_roi_side": args.min_roi_side,
                        "conf": args.conf, "max_det": args.max_det,
                        "min_refine_plate_w": args.min_refine_plate_w,
                        "refine_thresholds_base": list(C.REFINE_THRESHOLDS_BASE),
                        "refine_thresholds_extra": thrs,
                        "agree_iou": args.agree_iou, "agree_kpt_pct": args.agree_kpt_pct,
                        "consensus": args.consensus,
                        "teachers": [t["label"] for t in C.TEACHERS]},
    }


def _gpu_banner():
    """打印一次推理设备信息，便于在日志里确认是否真的用上 GPU。"""
    try:
        from autolabel import gpu as gpu_mod
        info = gpu_mod.describe()
        providers = info.get("providers") or []
        dev = "CUDA" if "CUDAExecutionProvider" in providers else "CPU"
        return ("推理设备=%s（onnxruntime %s, providers=%s, 预加载 %d 库/失败 %d）"
                % (dev, info.get("onnxruntime", "?"), ",".join(providers),
                   len(info.get("loaded") or []), len(info.get("failed") or [])))
    except Exception as exc:
        return "设备探测失败: %s" % exc


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = cfg_from_args(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        paths = list_images(args.images, include_val=not args.no_val)
        # 先在全量上采样（同参数可复现），再按已完成的 rel 过滤，保证重跑同命令是幂等的
        picked = sample_images(paths, args.limit, args.sample, args.seed)
        keys = assign_keys(paths, args.images)
        if not args.no_resume:
            meta_dir = out_dir / "meta"
            done = {json.loads(p.read_text(encoding="utf-8")).get("rel", "")
                    for p in meta_dir.rglob("*.json")} if meta_dir.exists() else set()
            before = len(picked)
            picked = [p for p in picked if rel_path(p, args.images) not in done]
            if before != len(picked):
                print("[resume] 跳过已完成 %d 张，本次处理 %d 张" % (before - len(picked), len(picked)))
        print("[input] %s -> 处理 %d 张 (sample=%s limit=%d)" % (args.images, len(picked), args.sample, args.limit))

        (out_dir / "run_config.json").write_text(json.dumps(
            {"args": vars(args), "teachers": list(C.TEACHERS)}, ensure_ascii=False, indent=1), encoding="utf-8")

        t0 = time.time()
        chunks = _chunkify([(str(p), keys[str(p)]) for p in picked], args.workers)
        print("[gpu] %s" % _gpu_banner())
        if args.workers > 1 and len(chunks) > 1:
            print("[run] %d 进程并行，每进程 %d~%d 张" % (len(chunks), min(map(len, chunks)), max(map(len, chunks))))
            ctx = get_context("spawn")
            jobs = [(chunk, cfg, i) for i, chunk in enumerate(chunks)]
            with ctx.Pool(len(chunks)) as pool:
                for wid, _ in enumerate(pool.starmap(worker_main, jobs)):
                    print("[run] worker %d 完成" % wid)
        else:
            worker_main(_chunkify([(str(p), keys[str(p)]) for p in picked], 1)[0]
                        if picked else [], cfg, 0)
        merge_progress(out_dir)
        report_errors(out_dir)
        print("[run] 完成 %d 张，用时 %.1f 分钟" % (len(picked), (time.time() - t0) / 60.0))

    write_review_list(out_dir)
    st = stats_mod.write(out_dir)
    _print_summary(st, out_dir)
    if args.panels > 0:
        build_panels(out_dir, args.images, args.panels, args.sheet_cols, tag="check")
        review_metas = [m for m in (out_dir / "meta").rglob("*.json")
                        if json.loads(m.read_text(encoding="utf-8"))["n_review"] > 0]
        if review_metas:
            build_panels(out_dir, args.images, min(len(review_metas), args.panels),
                         args.sheet_cols, tag="review")
    print("[done] 标签: %s/labels  元信息: %s/meta  复核: %s/review  图: %s/figures" %
          (out_dir, out_dir, out_dir, out_dir))
    return 0


def _print_summary(st, out_dir):
    ob, im = st["objects"], st["images"]
    print("\n==== 自动标注统计 ====")
    print("图片 %d（无检出 %d） 目标 %d 待复核 %d (%.1f%%)" %
          (im["total"], im["no_detection"], ob["total"], ob["review"], 100 * ob["review_rate"]))
    print("精修接受率 %.1f%%  双教师一致率 %.1f%%  仅主/仅副 %d/%d" % (
        100 * st["refine_accept_rate"], 100 * st["consistency"]["agree_rate_over_paired"],
        st["consistency"]["primary_only"], st["consistency"]["secondary_only"]))
    if st.get("plate_w_px"):
        print("板宽 px: mean=%.1f p50=%.1f p90=%.1f   粗框→精修位移: mean=%s px (%s%%板宽)" % (
            st["plate_w_px"]["mean"], st["plate_w_px"]["p50"], st["plate_w_px"]["p90"],
            (st.get("refine_shift_px") or {}).get("mean"), (st.get("refine_shift_pct_plate") or {}).get("mean")))
    if st.get("timing", {}).get("ms_per_image"):
        t = st["timing"]
        print("耗时: %.1f 分钟，单图 %.0f ms，状态 %s" % (
            t.get("total_minutes", 0), t["ms_per_image"]["mean"], json.dumps(t["status"], ensure_ascii=False)))
    print("报告: %s/report.md  统计: %s/stats.json" % (out_dir, out_dir))


if __name__ == "__main__":
    sys.exit(main())
