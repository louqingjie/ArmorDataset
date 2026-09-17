# -*- coding: utf-8 -*-
"""颜色类审计：标签里的颜色索引（B/R/P/N）到底对应什么真实颜色？

做法（纯像素证据，不看命名）：
  1. 统计 color × num 共现、板宽分布；
  2. 按四点取出左/右灯条（LT→LB、RT→RB 两条边）附近的像素，
     算 HSV 的饱和度与色相：灯条发光时饱和度高，未发光/灰板时饱和度低；
  3. 输出每个颜色类的 高饱和比例 / 饱和度中位数 / 色相中位数（按发光像素统计），
     并把抽样裁剪拼成一张图供肉眼确认。

用法:
  python color_audit.py AutoLabel                    # 统计 + 拼图
  python color_audit.py AutoLabel --max-per-class 300 --fig Test/figures/color_audit.png
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np

import make_table as MT

ROOT = Path(__file__).resolve().parent


def load_objects(out_dir: Path, max_per_class: int, seed: int = 0, min_plate_w: float = 24.0):
    """扫描 meta，按颜色类抽样对象（排除废弃目标、过小板）。"""
    per_color = collections.defaultdict(list)
    crosstab = collections.Counter()
    plate_w = collections.defaultdict(list)
    for p in sorted((out_dir / "meta").rglob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        for i, o in enumerate(m.get("objects") or []):
            if o.get("deprecated"):
                continue
            ci, ni = o.get("color"), o.get("num")
            crosstab[(ci, ni)] += 1
            pw = float(o.get("plate_w") or 0)
            plate_w[ci].append(pw)
            if pw < min_plate_w:
                continue
            per_color[ci].append({"key": m["key"], "idx": i, "path": m["path"], "obj": o})
    rnd = random.Random(seed)
    for ci, lst in per_color.items():
        rnd.shuffle(lst)
        del lst[max_per_class:]
    return per_color, crosstab, plate_w


def bar_pixels(bgr, quad):
    """沿左/右灯条取样，返回 HSV 像素数组（H 0-179, S 0-255, V 0-255）。"""
    h, w = bgr.shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pts = []
    for a, b in ((0, 1), (3, 2)):                    # 左灯条 LT→LB，右灯条 RT→RB
        for t in np.linspace(0.18, 0.82, 7):
            px, py = q[a] + t * (q[b] - q[a])
            xi, yi = int(round(px)), int(round(py))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = xi + dx, yi + dy
                    if 0 <= x < w and 0 <= y < h:
                        pts.append(hsv[y, x])
    return np.asarray(pts, np.int32) if pts else np.zeros((0, 3), np.int32)


def hue_name(h_deg):
    """色相角(0-360) -> 粗略颜色名（用于人类可读报告）。"""
    if h_deg < 0:
        return "-"
    if h_deg >= 345 or h_deg < 15:
        return "红"
    if h_deg < 45:
        return "橙黄"
    if h_deg < 150:
        return "绿"
    if h_deg < 195:
        return "青"
    if h_deg < 260:
        return "蓝"
    if h_deg < 305:
        return "紫"
    return "品红"


HUE_BINS = [(345, 15, "红"), (15, 45, "橙黄"), (45, 150, "绿"),
            (150, 195, "青"), (195, 260, "蓝"), (260, 305, "紫"), (305, 345, "品红")]


def crop_stats(bgr, quad, expand=1.3, sat_thr=90, val_thr=60):
    """以四点外接框作物，统计"有颜色的像素"占比与主色相（比灯条细带更抗噪）。

    灰板（未发光/无灯条）→ mode_frac 很低；发光板 → 主色相明确且占饱和像素多数。
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    x1, y1 = q[:, 0].min(), q[:, 1].min()
    x2, y2 = q[:, 0].max(), q[:, 1].max()
    cx, cy, cw, ch = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * expand, (y2 - y1) * expand
    x1, y1 = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    x2 = int(min(bgr.shape[1], cx + cw / 2))
    y2 = int(min(bgr.shape[0], cy + ch / 2))
    sub = bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    S, V, H = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    mask = (S >= sat_thr) & (V >= val_thr)
    frac = float(mask.mean())
    res = {"crop_frac": frac, "crop_px": int(sub.shape[0] * sub.shape[1]),
           "s_med": float(np.median(S)), "hue_deg": -1.0, "hue_name": "-", "mode_frac": 0.0}
    if mask.sum() >= 20:
        hist = np.bincount(H[mask].ravel(), minlength=180).astype(np.float64)
        k = np.array([1, 2, 3, 2, 1], np.float64)
        hist = np.convolve(hist, k / k.sum(), mode="same")       # 轻微平滑，避免单点噪声
        peak = int(hist.argmax())
        lo, hi = (peak - 10) % 180, (peak + 10) % 180
        sel = hist[lo:hi + 1].sum() if lo <= hi else hist[lo:].sum() + hist[:hi + 1].sum()
        res["hue_deg"] = peak * 2.0
        res["hue_name"] = hue_name(peak * 2.0)
        res["mode_frac"] = float(sel / max(1.0, hist.sum()))
    return res


def classify(hsv_px, sat_thr=90):
    """按灯条像素判定：发光比例 + 发光像素的色相中位数 + 饱和度中位数。"""
    if not len(hsv_px):
        return None
    s = hsv_px[:, 1].astype(np.float32)
    lit = hsv_px[s >= sat_thr]
    top = hsv_px[s >= max(sat_thr, float(np.percentile(s, 90)))]
    hue = np.median(top[:, 0]) * 2.0 if len(top) else -1.0
    return {"lit_frac": float((s >= sat_thr).mean()),
            "s_med": float(np.median(s)),
            "s_p90": float(np.percentile(s, 90)),
            "hue_deg": float(hue),
            "hue_name": hue_name(hue) if hue >= 0 else "-",
            "n_px": int(len(hsv_px)),
            "lit_frac_of_lit": float(len(lit) / len(hsv_px))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["AutoLabel"])
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--min-plate-w", type=float, default=24.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default="Test/figures/color_audit.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.dirs[0])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    per_color, crosstab, plate_w = load_objects(out_dir, args.max_per_class,
                                                args.seed, args.min_plate_w)

    print("=" * 78)
    print("颜色类分布（color × num，已排除废弃目标）")
    print("=" * 78)
    cols = sorted({c for c, _ in crosstab})
    nums = sorted({n for _, n in crosstab})
    print("%-6s %s %s" % ("color", " ".join("%6s" % MT.NUM_NAMES.get(n, n) for n in nums), " 合计"))
    for c in cols:
        row = [crosstab.get((c, n), 0) for n in nums]
        print("%-6s %s %6d" % (MT.COLOR_NAMES.get(c, c), " ".join("%6d" % v for v in row), sum(row)))
    print()
    print("%-6s %8s %10s %10s" % ("color", "样本数", "板宽中位", "板宽p90"))
    for c in cols:
        pw = np.asarray(plate_w[c], np.float32)
        print("%-6s %8d %10.1f %10.1f" % (MT.COLOR_NAMES.get(c, c), len(pw),
                                          float(np.median(pw)), float(np.percentile(pw, 90))))

    # ---- 逐样本取色 ----
    img_cache = {}
    rows = collections.defaultdict(list)
    tiles_by_class = collections.defaultdict(list)
    for c in sorted(per_color):
        for it in per_color[c]:
            path = str(it["path"])
            if not Path(path).is_absolute():
                path = str(ROOT / path)
            if path not in img_cache:
                bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if bgr is not None and bgr.ndim == 3 and bgr.shape[2] == 4:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
                img_cache[path] = bgr
                if len(img_cache) > 60:
                    img_cache.pop(next(iter(img_cache)))
            bgr = img_cache[path]
            if bgr is None:
                continue
            quad = it["obj"].get("quad_final_px")
            if not quad:
                continue
            st = classify(bar_pixels(bgr, quad))
            cs = crop_stats(bgr, quad)
            if st is None or cs is None:
                continue
            st = {**st, **cs}
            st.update({"color": c, "num": it["obj"].get("num"), "plate_w": it["obj"].get("plate_w")})
            rows[c].append(st)
            if len(tiles_by_class[c]) < 8 and len(rows[c]) % 10 == 1:   # 每类固定抽 8 块，保证跨类可对比
                tiles_by_class[c].append((bgr, quad, c, it["obj"], st))

    print()
    print("=" * 78)
    print("像素实测（板宽 ≥ %.0fpx，每类最多 %d 个）" % (args.min_plate_w, args.max_per_class))
    print("=" * 78)
    print("%-5s %5s %8s %9s %9s %9s %8s  %s" %
          ("标签色", "样本", "有彩占比", "彩色样本", "主色相中位", "S中位", "判名", "色相分布(彩色样本)"))
    for c in sorted(rows):
        r = rows[c]
        col = np.array([x["crop_frac"] for x in r])          # 作物内"有颜色"像素占比
        smed = np.array([x["s_med"] for x in r])
        colorful = [x for x in r if x["crop_frac"] >= 0.02 and x["hue_deg"] >= 0]
        hues = np.array([x["hue_deg"] for x in colorful])
        names = collections.Counter(x["hue_name"] for x in colorful)
        top = ", ".join("%s %.0f%%" % (k, 100 * v / max(1, len(colorful)))
                        for k, v in names.most_common(3))
        print("%-5s %5d %7.0f%% %8.0f%% %10.0f %9.1f %8s  %s" %
              (MT.COLOR_NAMES.get(c, c), len(r), 100 * (col >= 0.02).mean(),
               100 * len(colorful) / max(1, len(r)),
               float(np.median(hues)) if len(hues) else -1, float(np.median(smed)),
               hue_name(float(np.median(hues))) if len(hues) else "-", top))
    print()
    print("判读：『有彩占比』= 四点外接框作物里 S≥90 且 V≥60 的像素比例中位；")
    print("      灰板/未发光 → 有彩占比≈0（『彩色样本』比例低）；发光板 → 有彩占比高且主色相明确。")

    # ---- 拼图（按类分组，每类 8 块）----
    tiles = [t for c in sorted(tiles_by_class) for t in tiles_by_class[c]]
    if tiles:
        tiles_out = []
        for bgr, quad, c, o, st in tiles:
            q = np.asarray(quad, np.float32)
            x1, y1 = q[:, 0].min(), q[:, 1].min()
            x2, y2 = q[:, 0].max(), q[:, 1].max()
            cw, ch = max(4.0, (x2 - x1)), max(4.0, (y2 - y1))
            ex, ey = cw * 0.5, ch * 0.5
            x1, y1 = int(max(0, x1 - ex)), int(max(0, y1 - ey))
            x2 = int(min(bgr.shape[1], x2 + ex)); y2 = int(min(bgr.shape[0], y2 + ey))
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tile = cv2.resize(crop, (220, 150), interpolation=cv2.INTER_NEAREST)
            label = "标签%s %s号 %.0fpx 有彩%.0f%% %s" % (
                MT.COLOR_NAMES.get(c, c), MT.NUM_NAMES.get(o.get("num"), "?"),
                float(o.get("plate_w") or 0), 100 * st["crop_frac"], st["hue_name"])
            cv2.rectangle(tile, (0, 0), (219, 20), (0, 0, 0), -1)
            cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles_out.append(tile)
        if tiles_out:
            ncol = 4
            nrow = (len(tiles_out) + ncol - 1) // ncol
            h, w = tiles_out[0].shape[:2]
            sheet = np.full((nrow * h, ncol * w, 3), 24, np.uint8)
            for i, t in enumerate(tiles_out):
                r, cc = divmod(i, ncol)
                sheet[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t
            fig = Path(args.fig)
            if not fig.is_absolute():
                fig = ROOT / fig
            fig.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fig), sheet)
            print("\n拼图 -> %s（%d 块：标签色 + 编号 + 板宽 + 实测饱和度/色相）" % (fig, len(tiles_out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
