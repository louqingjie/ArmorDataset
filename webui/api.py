# -*- coding: utf-8 -*-
"""Web 工作台的服务层实现：内存索引、预览图、统计、后台任务。

设计要点:
  * 不引入数据库：索引直接来自 out/meta/**.json，签名 (文件数, 最大 mtime) 变化时重建；
  * 21k 规模下重建较慢 → 条目数超阈值时走后台线程，接口返回 building 进度，前端轮询；
  * 预览图按 (路径, mtime, max) 做 LRU 缓存，避免 4K 图反复解码；
  * 所有路径经 safe_path 校验，拒绝目录穿越与越界访问。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import numpy as np

from autolabel import config as C

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(__file__).resolve().parent / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("webui.api")

ALLOWED_ROOTS = [ROOT]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INDEX_SYNC_LIMIT = 4000          # meta 文件数不超过该值时同步建索引，否则后台建
PRIORITY = {"conflict": 0, "no_match": 1, "refine_failed": 2, "tiny": 3, "refine_skipped": 5}
RECENT_CONFIRMED = 5       # 队列顶部只保留"最近确认"的 N 张，其余已确认沉到队列末尾（废弃之前）
FILTERS = ("all", "review", "agree", "conflict", "no_match", "refine_failed", "tiny",
           "refine_skipped", "background", "reviewed", "edited",
           "num_LB", "num_B")     # 编号=大基地装甲板 / 编号=基地（按图内是否含该类目标筛选）


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
# 路径安全
# --------------------------------------------------------------------------- #
def safe_path(p, must_exist=True, allow_roots=None):
    """把用户输入解析为绝对路径，并校验位于白名单根目录内。"""
    if p is None or str(p).strip() == "":
        raise ApiError("缺少路径参数")
    s = str(p).strip()
    if "\x00" in s:
        raise ApiError("非法路径")
    path = Path(s)
    if not path.is_absolute():
        path = ROOT / path
    rp = path.resolve()
    roots = [Path(r).resolve() for r in (allow_roots or ALLOWED_ROOTS)]
    for root in roots:
        try:
            rp.relative_to(root)
            break
        except ValueError:
            continue
    else:
        raise ApiError("路径越界（仅允许工作目录内）: %s" % s, 403)
    if must_exist and not rp.exists():
        raise ApiError("路径不存在: %s" % s, 404)
    return rp


def rel_to_root(p: Path):
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


# --------------------------------------------------------------------------- #
# LRU 缓存
# --------------------------------------------------------------------------- #
class LRU:
    def __init__(self, maxsize=64):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)


# --------------------------------------------------------------------------- #
# 后台任务注册器（统计刷新 / 导出等耗时操作）
# --------------------------------------------------------------------------- #
class Task(dict):
    pass


_TASKS: dict = {}
_TASKS_LOCK = threading.Lock()


def submit(kind, fn, *args, **kwargs):
    tid = uuid.uuid4().hex[:12]
    task = Task(id=tid, kind=kind, status="running", progress=0.0, message="",
                result=None, error=None, started=time.time(), finished=None)
    with _TASKS_LOCK:
        _TASKS[tid] = task

    def _run():
        try:
            def report(p, msg=""):
                task["progress"] = round(float(p), 3)
                if msg:
                    task["message"] = msg
            task["result"] = fn(report, *args, **kwargs)
            task["status"] = "done"
        except Exception as exc:                       # 后台异常也要可见
            LOG.exception("task %s(%s) failed", tid, kind)
            task["status"] = "error"
            task["error"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            task["finished"] = time.time()

    threading.Thread(target=_run, name="task-%s" % kind, daemon=True).start()
    return task


def task_status(tid):
    with _TASKS_LOCK:
        task = _TASKS.get(tid)
    if not task:
        raise ApiError("任务不存在: %s" % tid, 404)
    return task


def drop_task(tid):
    with _TASKS_LOCK:
        _TASKS.pop(tid, None)


# --------------------------------------------------------------------------- #
# 输出目录索引
# --------------------------------------------------------------------------- #
def _entry_from_meta(meta: dict, mtime: float, dep_info=None):
    objs = meta.get("objects", []) or []
    flags = Counter()
    sources = Counter()
    colors, nums, review_kinds, pairs = Counter(), Counter(), Counter(), Counter()
    shifts = []
    plate_w = 0.0
    score = 0.0
    reasons = []
    for o in objs:
        if o.get("deprecated"):
            continue          # 废弃目标不参与 flags/优先级/统计（仅计入 dep_objs）
        oflags = o.get("flags", []) or []
        for f in oflags:
            flags[f] += 1
        sources[o.get("source") or "?"] += 1
        colors[o.get("color_name") or "?"] += 1
        nums[o.get("num_name") or "?"] += 1
        pair = "%s%s" % (o.get("color_name") or "?", o.get("num_name") or "?")
        pairs[pair] += 1
        plate_w = max(plate_w, float(o.get("plate_w") or 0.0))
        score = max(score, float(o.get("score_primary") or o.get("score_secondary") or 0.0))
        r = o.get("refine") or {}
        if r.get("accepted") and r.get("d_coarse_to_refine_pct") is not None:
            shifts.append(float(r["d_coarse_to_refine_pct"]))
        if o.get("reason"):
            reasons.append(o["reason"])
        if o.get("review"):
            if any(f.startswith("conflict") for f in oflags):
                review_kinds["conflict"] += 1
            elif "no_match" in oflags:
                review_kinds["no_match"] += 1
            elif "tiny" in oflags:
                review_kinds["tiny"] += 1
            else:
                review_kinds["refine_failed"] += 1
    n_dep_obj = sum(1 for o in objs if o.get("deprecated"))
    prio = 9
    for f in flags:
        prio = min(prio, PRIORITY.get(f, 9))
    return {
        "key": meta.get("key"), "image": meta.get("image"), "rel": meta.get("rel"),
        "n_obj": len(objs), "n_review": int(meta.get("n_review") or 0),
        "flags": sorted(flags), "flag_counts": dict(flags), "sources": sorted(sources),
        "reason": reasons[0] if reasons else None,
        "plate_w": round(plate_w, 2), "score": round(score, 4),
        "size": meta.get("size") or [0, 0], "priority": prio,
        "reviewed": any(f == "reviewed" for f in flags),
        "reviewed_at": _reviewed_at(meta),        # 最近一次确认时间（队列分区用）
        # 来源分组（raw pic 批次）：holdout=True 表示来自验证集来源，导出时不得进 train
        "group": meta.get("source_group"),
        "holdout": C.is_holdout(meta.get("source_group")),
        "edited": bool(meta.get("review_edit")),
        "colors": dict(colors), "nums": dict(nums), "review_kinds": dict(review_kinds),
        "pairs": dict(pairs),                      # 颜色×编号 组合（用于导出时交错均衡排序）
        "shift_pct": round(sum(shifts) / len(shifts), 3) if shifts else None,
        "mtime": round(mtime, 3),
        # 目标级废弃（meta 里 objects[*].deprecated，如大装甲）：标注保留、不参与训练导出
        "dep_objs": n_dep_obj,
        "n_usable": len(objs) - n_dep_obj,
        # 图级废弃（来自 out/deprecated.json，权威表；整图不参与训练集导出）
        "deprecated": bool(dep_info),
        "dep_by": (dep_info or {}).get("by"),
        "dep_time": (dep_info or {}).get("time"),
        "dep_reason": (dep_info or {}).get("reason"),
    }


def _reviewed_at(meta):
    """最近一次"确认"时间：优先 meta.reviewed_at，其次最后一条人工编辑审计的时间。"""
    t = meta.get("reviewed_at")
    if t:
        return t
    ed = meta.get("review_edits") or []
    return (ed[-1] or {}).get("time") if ed else None


def _is_done(e):
    """已处理完（没有待复核项）：已确认 reviewed 或 背景图（无目标）。"""
    return (e.get("n_review") or 0) == 0 and (e.get("reviewed") or not e.get("n_obj"))


def _needs_review(e):
    """需要人工复核（未被废弃且还有待复核目标）。"""
    return bool(e.get("n_review")) and not e.get("deprecated")


def _no_review(e):
    """不需要人工复核：有目标、无待复核项、未废弃（含精修通过/双教师一致/已人工确认）。"""
    return ((e.get("n_review") or 0) == 0 and (e.get("n_obj") or 0) > 0
            and not e.get("deprecated"))


def _hist(values, edges):
    counts = [0] * (len(edges) - 1)
    n_under = 0
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
        else:
            if v < edges[0]:
                n_under += 1
            else:
                counts[-1] += 1
    return {"edges": list(edges), "counts": counts, "under": n_under}


class OutIndex:
    """某个 out 目录的 meta 摘要索引。"""

    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.entries: dict = {}
        self.sig = None
        self.built_at = 0.0
        self.building = False
        self.progress = (0, 0)
        self.error = None
        self._lock = threading.Lock()
        self._facets_cache = None          # (时间戳, facets)，避免状态轮询里重复全表统计

    # ---- 签名与构建 ----
    def signature(self):
        meta_dir = self.out / "meta"
        n, mx = 0, 0.0
        for p in meta_dir.rglob("*.json"):
            n += 1
            try:
                mx = max(mx, p.stat().st_mtime)
            except OSError:
                pass
        return (n, round(mx, 3))

    def ensure(self, force=False, blocking=True):
        sig = self.signature()
        if not force and sig == self.sig and self.built_at:
            return self
        if self.building:
            return self
        if sig[0] <= INDEX_SYNC_LIMIT or blocking:
            self._build(sig)
        else:
            threading.Thread(target=self._build, args=(sig,), daemon=True).start()
        return self

    def _build(self, sig=None):
        with self._lock:
            if self.building:
                return
            self.building = True
        from webui import deprecate as dep_mod
        dep = dep_mod.load(self.out)                    # 废弃表（sidecar，权威）
        try:
            files = sorted((self.out / "meta").rglob("*.json"))
            total = len(files)
            self.progress = (0, total)
            entries = {}
            for i, p in enumerate(files, 1):
                try:
                    meta = json.loads(p.read_text(encoding="utf-8"))
                    key = meta.get("key") or str(p.relative_to(self.out / "meta").with_suffix(""))
                    entries[key] = _entry_from_meta(meta, p.stat().st_mtime, dep.get(key))
                except Exception as exc:
                    LOG.warning("索引跳过 %s: %s", p, exc)
                if i % 500 == 0:
                    self.progress = (i, total)
            self.entries = entries
            self.sig = sig or self.signature()
            self.built_at = time.time()
            self.error = None
            self.progress = (total, total)
            LOG.info("索引完成 %s: %d 条 (%.2fs)", rel_to_root(self.out), len(entries),
                     time.time() - self.built_at + 0.0)
        except Exception as exc:
            self.error = str(exc)
            LOG.exception("建索引失败 %s", self.out)
        finally:
            self.building = False

    # ---- 查询 ----
    def list(self, flt="all", q="", sort="priority", order="asc", offset=0, limit=200, ensure=True):
        if ensure:
            self.ensure()
        items = list(self.entries.values())
        if flt and flt != "all":
            items = [e for e in items if _match_filter(e, flt)]
        if q:
            ql = q.lower()
            items = [e for e in items if ql in (e["key"] or "").lower()
                     or ql in (e.get("image") or "").lower()]

        def sort_key(e):
            if sort == "plate_w":
                return (e["plate_w"], e["key"])
            if sort == "score":
                return (e["score"], e["key"])
            if sort == "n_obj":
                return (e["n_obj"], e["key"])
            if sort == "size":
                return (e["size"][0] * e["size"][1], e["key"])
            if sort == "mtime":
                return (e["mtime"], e["key"])
            if sort == "key":
                return (e["key"],)
            return (e["priority"], -e["plate_w"], e["key"])      # priority 默认

        items.sort(key=sort_key, reverse=(order == "desc"))
        # 队列分区（稳定排序，组内保持上面的排序结果，两种 order 都成立）：
        #   ① 最近确认的 N 张（最新的在最上面，方便回看/撤销）
        #   ② 待处理（按 priority 等原排序）
        #   ③ 其余已处理完（已确认 reviewed / 背景图）——"甩到队列最后"
        #   ④ 已废弃（永远最后，"废弃之前"即归档段之上）
        recent = self.recent_keys(RECENT_CONFIRMED)
        head, rest = [], []
        for e in items:
            (head if e["key"] in recent else rest).append(e)
        if head:
            head.sort(key=lambda e: e.get("reviewed_at") or "", reverse=True)
        rest.sort(key=lambda e: 2 if e.get("deprecated") else (1 if _is_done(e) else 0))
        items = head + rest
        total = len(items)
        return {
            "total": total,
            "offset": int(offset),
            "limit": int(limit),
            "items": items[int(offset):int(offset) + int(limit)],
            "facets": self.facets(),
        }

    def recent_keys(self, n=RECENT_CONFIRMED):
        """最近确认的 N 个 key（按确认时间倒序），用于把队列顶部限制成一小段"最近操作"。"""
        done = [e for e in self.entries.values()
                if e.get("reviewed_at") and not e.get("deprecated") and _is_done(e)]
        done.sort(key=lambda e: e["reviewed_at"], reverse=True)
        return {e["key"] for e in done[:max(0, int(n))]}

    def facets_cached(self, ttl=5.0):
        """带短 TTL 的 facets（顶部栏每 2s 轮询 /api/state，不值得每次都全表统计）。"""
        now = time.time()
        if self._facets_cache and now - self._facets_cache[0] < ttl:
            return self._facets_cache[1]
        f = self.facets()
        self._facets_cache = (now, f)
        return f

    def facets(self):
        cnt = Counter()
        for e in self.entries.values():
            cnt["all"] += 1
            if e["n_obj"] == 0:
                cnt["background"] += 1
            if e["n_review"]:
                cnt["review"] += 1
            if e["reviewed"]:
                cnt["reviewed"] += 1
            if e["edited"]:
                cnt["edited"] += 1
            for f in e["flags"]:
                cnt[f] += 1
        n_dep = sum(1 for e in self.entries.values() if e.get("deprecated"))
        cnt["deprecated"] = n_dep
        cnt["active"] = len(self.entries) - n_dep
        cnt["obj_deprecated"] = sum(1 for e in self.entries.values() if (e.get("dep_objs") or 0) > 0)
        cnt["all_deprecated"] = sum(1 for e in self.entries.values()
                                    if e["n_obj"] > 0 and (e.get("n_usable") or 0) == 0)
        # 需要复核 / 不需要复核（互斥口径）：后者＝有目标、无待复核项、未废弃
        cnt["need_review"] = sum(1 for e in self.entries.values() if _needs_review(e))
        cnt["no_review"] = sum(1 for e in self.entries.values() if _no_review(e))
        cnt["holdout"] = sum(1 for e in self.entries.values() if e.get("holdout"))
        out = {k: int(cnt.get(k, 0)) for k in
               ("all", "review", "reviewed", "edited", "background", "agree", "conflict",
                "no_match", "refine_failed", "refine_skipped", "tiny", "deprecated", "active",
                "obj_deprecated", "all_deprecated", "need_review", "no_review", "holdout")}
        for nm in ("B", "LB"):            # 编号维度：含该类目标的图片数
            out["num_" + nm] = sum(1 for e in self.entries.values()
                                   if (e.get("nums") or {}).get(nm, 0) > 0)
        return out


def _match_filter(e, flt):
    if flt == "review":
        return e["n_review"] > 0
    if flt == "background":
        return e["n_obj"] == 0
    if flt == "reviewed":
        return e["reviewed"]
    if flt == "edited":
        return e["edited"]
    if flt == "deprecated":
        return bool(e.get("deprecated"))
    if flt == "active":
        return not e.get("deprecated")
    if flt == "obj_deprecated":
        return (e.get("dep_objs") or 0) > 0
    if flt == "all_deprecated":
        return e["n_obj"] > 0 and (e.get("n_usable") or 0) == 0
    if flt == "no_review":
        return _no_review(e)
    if flt == "need_review":
        return _needs_review(e)
    if flt == "holdout":                       # 仅验证集来源（防"机器预习"的那批图）
        return bool(e.get("holdout"))
    if flt.startswith("num_"):
        return (e.get("nums") or {}).get(flt[4:], 0) > 0      # 按编号名筛选（如 num_LB / num_B）
    return flt in e["flags"]


_INDEXES: dict = {}
_INDEX_LOCK = threading.Lock()


def get_index(out_dir: Path, force=False, blocking=True):
    """获取（必要时构建）索引。

    blocking=False 时，条目数超过阈值的大目录改为后台构建并立即返回（前端据 building 轮询），
    避免 21k 规模下首个 HTTP 请求被阻塞一分钟。
    """
    key = str(out_dir.resolve())
    with _INDEX_LOCK:
        idx = _INDEXES.get(key)
        if idx is None or force:
            idx = OutIndex(out_dir)
            _INDEXES[key] = idx
    return idx.ensure(force=force, blocking=blocking)


def invalidate_index(out_dir: Path):
    with _INDEX_LOCK:
        _INDEXES.pop(str(Path(out_dir).resolve()), None)


def refresh_index_entry(out_dir: Path, key: str, mtime=None):
    """只刷新单个 key 的索引条目（编辑/废弃后调用），避免 21k 规模下整库重建。

    * 从磁盘重读该 key 的 meta 与废弃标记，替换条目内容；
    * 同步推进索引签名中的 mtime，使后续请求不会判定为“需要重建”。
    """
    from webui import deprecate as dep_mod
    out_dir = Path(out_dir)
    with _INDEX_LOCK:
        idx = _INDEXES.get(str(out_dir.resolve()))
    if idx is None or not idx.sig:
        return False
    try:
        meta = read_meta(out_dir, key)
        p = out_dir / "meta" / (str(key) + ".json")
        mt = p.stat().st_mtime if p.exists() else time.time()
        idx.entries[key] = _entry_from_meta(meta, mt, dep_mod.load(out_dir).get(key))
        idx._facets_cache = None
    except Exception as exc:
        LOG.warning("刷新索引条目失败 %s/%s: %s", rel_to_root(out_dir), key, exc)
        return False
    n, mx = idx.sig
    idx.sig = (n, round(max(mx, mtime or mt), 3))
    return True


# --------------------------------------------------------------------------- #
# 输出目录 / 图片目录扫描
# --------------------------------------------------------------------------- #
_DIRS_CACHE = {"t": 0.0, "outs": [], "images": []}


def scan_dirs(force=False):
    if not force and time.time() - _DIRS_CACHE["t"] < 10:
        return _DIRS_CACHE["outs"], _DIRS_CACHE["images"]
    outs, images = [], []
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name in {"webbuild"}:
            continue
        if (p / "meta").is_dir():
            sig = OutIndex(p).signature()
            outs.append({"path": p.name, "n_meta": sig[0], "mtime": sig[1]})
        if p.name in {"RawPic", "BaseLine"}:
            images.append(p.name)
        if p.name == "RawPic":
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in sub.iterdir() if f.is_file()):
                    images.append("%s/%s" % (p.name, sub.name))
    _DIRS_CACHE.update(t=time.time(), outs=outs, images=images)
    return outs, images


def out_defaults():
    return {
        "images": rel_to_root(C.DEFAULT_IMAGES), "out": rel_to_root(C.DEFAULT_OUT),
        "limit": 1000, "sample": "stratified", "workers": 2, "expand": C.ROI_EXPAND,
        "min_roi_side": C.ROI_MIN_SIDE, "conf": C.TEACHER_CONF, "max_det": C.TEACHER_MAX_DET,
        "min_refine_plate_w": C.MIN_REFINE_PLATE_W,
        "refine_extra_thresholds": ",".join(str(v) for v in C.REFINE_THRESHOLDS_EXTRA),
        "refine_thresholds_base": list(C.REFINE_THRESHOLDS_BASE),
        "agree_iou": C.AGREE_IOU, "agree_kpt_pct": C.AGREE_KPT_PCT,
        "consensus": C.CONSENSUS, "class_mode": C.CLASS_MODE,
        "panel_limit": C.PANEL_LIMIT, "sheet_cols": C.SHEET_COLS,
        "teachers": [t["label"] for t in C.TEACHERS],
        "tiny_plate_w": C.TINY_PLATE_W,
    }


def out_summary(out_dir: Path):
    """轻量摘要（不建索引）：meta 文件数/时间戳 + 缓存的 stats.json 关键指标。"""
    sig = OutIndex(out_dir).signature()
    stats_path = out_dir / "stats.json"
    brief = None
    if stats_path.exists():
        try:
            s = json.loads(stats_path.read_text(encoding="utf-8"))
            brief = {"images": s.get("images", {}), "objects": s.get("objects", {}),
                     "refine_accept_rate": s.get("refine_accept_rate"),
                     "consistency": s.get("consistency", {}),
                     "mtime": round(stats_path.stat().st_mtime, 3)}
        except Exception:
            brief = None
    cfg = out_dir / "run_config.json"
    run_args = None
    if cfg.exists():
        try:
            run_args = json.loads(cfg.read_text(encoding="utf-8")).get("args")
        except Exception:
            run_args = None
    return {"path": rel_to_root(out_dir), "n_meta": sig[0], "mtime": sig[1],
            "stats": brief, "run_args": run_args}


# --------------------------------------------------------------------------- #
# 预览图渲染
# --------------------------------------------------------------------------- #
_PREVIEW_CACHE = LRU(maxsize=64)
_PREVIEW_STATS = {"rendered": 0, "ms": 0.0}


def resolve_image_path(meta: dict, out_dir: Path):
    """定位 meta 对应的原图：优先 meta.path，其次按 rel/key 在常见图片目录下找。"""
    cands = []
    if meta.get("path"):
        cands.append(Path(meta["path"]))
    rel = meta.get("rel") or meta.get("key")
    if rel:
        for base in (C.DEFAULT_IMAGES, C.BASELINE_DIR, ROOT):
            cands.append(Path(base) / rel)
    stems = [Path(rel).stem] if rel else []
    for p in cands:
        if p.is_file():
            try:
                return safe_path(p)
            except ApiError:
                raise
    if stems:
        for base in (C.DEFAULT_IMAGES, C.BASELINE_DIR):
            if not base.is_dir():
                continue
            for f in base.rglob(stems[0] + ".*"):
                if f.suffix.lower() in IMG_EXTS:
                    return f
    raise ApiError("找不到原图: %s" % (meta.get("rel") or meta.get("path")), 404)


def read_meta(out_dir: Path, key: str) -> dict:
    p = out_dir / "meta" / (str(key) + ".json")
    if not p.exists():
        raise ApiError("meta 不存在: %s" % key, 404)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ApiError("meta 解析失败: %s" % exc, 500)


def render_preview(path: Path, max_side=1600, quality=85):
    st = path.stat()
    ck = (str(path), round(st.st_mtime, 3), int(max_side), int(quality))
    hit = _PREVIEW_CACHE.get(ck)
    if hit is not None:
        return hit
    t0 = time.time()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ApiError("图像解码失败: %s" % rel_to_root(path), 500)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = 1.0
    if max_side and max(h, w) > int(max_side):
        scale = float(max_side) / float(max(h, w))
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ApiError("图像编码失败", 500)
    data = buf.tobytes()
    _PREVIEW_CACHE.put(ck, (data, [w, h], [int(w * scale), int(h * scale)]))
    _PREVIEW_STATS["rendered"] += 1
    _PREVIEW_STATS["ms"] = 0.9 * _PREVIEW_STATS["ms"] + 0.1 * (time.time() - t0) * 1000
    return data, [w, h], [int(w * scale), int(h * scale)]


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
_STATS_CACHE: dict = {}


def get_stats(out_dir: Path, force=False):
    from autolabel import stats as stats_mod
    sig = OutIndex(out_dir).signature()
    ck = str(out_dir.resolve())
    c = _STATS_CACHE.get(ck)
    if not force and c and c["sig"] == sig:
        return c["data"]
    data = stats_mod.collect(out_dir)
    data["_generated"] = round(time.time(), 3)
    _STATS_CACHE[ck] = {"sig": sig, "data": data}
    return data


def stats_refresh(report, out_dir: Path, panels=0, cols=6):
    """后台任务：重算 stats.json/report.md，并按需重出拼版图。

    参数顺序与 api.submit 约定一致（report 回调在最前）。
    """
    from autolabel import run as run_mod
    from autolabel import stats as stats_mod
    report(0.1, "重算统计")
    data = stats_mod.write(out_dir)
    _STATS_CACHE.pop(str(out_dir.resolve()), None)
    invalidate_index(out_dir)
    if panels:
        report(0.4, "生成抽检面板")
        cfg = out_dir / "run_config.json"
        images = None
        if cfg.exists():
            try:
                images = json.loads(cfg.read_text(encoding="utf-8"))["args"]["images"]
            except Exception:
                images = None
        images = images or rel_to_root(C.DEFAULT_IMAGES)
        run_mod.build_panels(out_dir, images, int(panels), int(cols), tag="check")
        report(0.8, "生成复核面板")
        run_mod.build_panels(out_dir, images, int(panels), int(cols), tag="review")
    report(1.0, "完成")
    return {"stats": data, "out": rel_to_root(out_dir)}


# --------------------------------------------------------------------------- #
# 端点实现（由 server.py 路由调用）
# --------------------------------------------------------------------------- #
def api_state(ctx):
    import webui.jobs as jobs
    outs, images = scan_dirs()
    out = ctx.q("out") or (outs[0]["path"] if outs else rel_to_root(C.DEFAULT_OUT))
    cur = safe_path(out, must_exist=False)
    info = {"ok": True, "version": _version(), "root": str(ROOT), "port": ctx.server_port,
            "outs": outs, "images_dirs": images, "defaults": out_defaults(),
            "current_out": rel_to_root(cur), "out_exists": cur.exists(),
            "local": bool(getattr(ctx, "local", True)),
            "job": jobs.status(), "preview": dict(_PREVIEW_STATS)}
    if cur.exists() and (cur / "meta").is_dir():
        info.update(out_summary(cur))
        from webui import deprecate as dep_mod
        info["deprecated"] = dep_mod.summary(cur)
        idx = _INDEXES.get(str(cur.resolve()))
        info["index"] = {"building": bool(idx and idx.building),
                         "n": len(idx.entries) if idx else 0}
        if idx and idx.entries:
            info["facets"] = idx.facets_cached()      # 供顶部栏显示"待复核/无需复核"
    return info


def api_images(ctx):
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT))
    idx = get_index(out, blocking=False)
    res = idx.list(flt=ctx.q("filter", "all"), q=ctx.q("q", ""), sort=ctx.q("sort", "priority"),
                   order=ctx.q("order", "asc"), offset=ctx.qi("offset", 0), limit=ctx.qi("limit", 200),
                   ensure=False)
    res.update({"ok": True, "out": rel_to_root(out),
                "index": {"building": idx.building, "n": len(idx.entries),
                          "built_at": round(idx.built_at, 3), "error": idx.error}})
    return res


def api_meta(ctx):
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT))
    key = ctx.q("key")
    if not key:
        raise ApiError("缺少 key 参数")
    meta = read_meta(out, key)
    try:
        img = resolve_image_path(meta, out)
        meta["_image_found"] = True
        meta["_image_path"] = rel_to_root(img)
    except ApiError as exc:
        meta["_image_found"] = False
        meta["_image_path"] = None
        meta["_image_error"] = str(exc)
    from webui import deprecate as dep_mod
    meta["deprecated"] = key in dep_mod.load(out)        # 以 sidecar 为准
    meta["ok"] = True
    meta["out"] = rel_to_root(out)
    meta["image_url"] = "/api/image?out=%s&key=%s" % (rel_to_root(out), key)
    return meta


def api_image(ctx):
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT))
    key = ctx.q("key")
    if not key:
        raise ApiError("缺少 key 参数")
    meta = read_meta(out, key)
    img = resolve_image_path(meta, out)
    data, wh, rwh = render_preview(img, max_side=ctx.qi("max", 1600), quality=ctx.qi("q", 85))
    return (200, "image/jpeg", data,
            {"X-Image-Size": "%dx%d" % (wh[0], wh[1], ),
             "X-Preview-Size": "%dx%d" % (rwh[0], rwh[1])})


def api_asset(ctx):
    """通用文件读取（限工作目录内）：用于查看 figures/ 拼版图、report.md 等。"""
    p = safe_path(ctx.q("path"))
    ext = p.suffix.lower()
    ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".md": "text/markdown; charset=utf-8", ".csv": "text/csv; charset=utf-8",
             ".json": "application/json; charset=utf-8", ".yaml": "text/yaml; charset=utf-8",
             ".yml": "text/yaml; charset=utf-8", ".log": "text/plain; charset=utf-8",
             ".txt": "text/plain; charset=utf-8"}.get(ext, "application/octet-stream")
    if ctx.qb("download", False):
        return (200, ctype, p.read_bytes(), {"Content-Disposition": 'attachment; filename="%s"' % p.name})
    return (200, ctype, p.read_bytes(), {"Cache-Control": "no-store"})


def api_stats(ctx):
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT))
    idx = get_index(out, blocking=False)
    entries = list(idx.entries.values())
    plate_w = [e["plate_w"] for e in entries if e["plate_w"]]
    shift = [e["shift_pct"] for e in entries if e.get("shift_pct") is not None]
    colors, nums, review_kinds, flags = Counter(), Counter(), Counter(), Counter()
    for e in entries:
        colors.update(e.get("colors") or {})
        nums.update(e.get("nums") or {})
        review_kinds.update(e.get("review_kinds") or {})
        flags.update({k: 1 for k in e["flags"]})        # 按图片计数的 flags 分布
    return {"ok": True, "out": rel_to_root(out),
            "stats": get_stats(out, force=ctx.qb("force", False)),
            "index": {"n": len(entries), "building": idx.building,
                      "deprecated": sum(1 for e in entries if e.get("deprecated")),
                      "dep_objects": sum(e.get("dep_objs") or 0 for e in entries),
                      "usable_objects": sum(e.get("n_usable") or 0 for e in entries),
                      "reviewed": sum(1 for e in entries if e.get("reviewed")),
                      "edited": sum(1 for e in entries if e.get("edited")),
                      "hist": {
                          "plate_w": _hist(plate_w, [0, 12, 24, 48, 96, 192, 384, 768, 1536, 10 ** 6]),
                          "shift_pct": _hist(shift, [0, 1, 2, 4, 6, 8, 10, 15, 20, 10 ** 6]),
                          "flags": dict(flags), "review_kind": dict(review_kinds),
                          "colors": dict(colors), "nums": dict(nums),
                      },
                      "total_objects": sum(e["n_obj"] for e in entries)}}


def api_stats_refresh(ctx):
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT))
    panels = int(ctx.body.get("panels", 0) or 0)
    cols = int(ctx.body.get("cols", C.SHEET_COLS) or C.SHEET_COLS)
    task = submit("stats", stats_refresh, out, panels=panels, cols=cols)
    return {"ok": True, "task": dict(task)}


def api_task(ctx):
    return {"ok": True, "task": dict(task_status(ctx.q("id")))}


def api_save(ctx):
    import webui.edits as edits
    return edits.save(ctx.body or {})


def api_restore(ctx):
    import webui.edits as edits
    return edits.restore(ctx.body or {})


def api_backups(ctx):
    import webui.edits as edits
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT))
    key = ctx.q("key")
    if not key:
        raise ApiError("缺少 key 参数")
    return {"ok": True, "key": key, "backups": edits.list_backups(out, key)}


def api_job_start(ctx):
    import webui.jobs as jobs
    return jobs.start(ctx.body or {})


def api_job_stop(ctx):
    import webui.jobs as jobs
    return jobs.stop()


def api_job_status(ctx):
    import webui.jobs as jobs
    return jobs.status()


def api_job_log(ctx):
    import webui.jobs as jobs
    return jobs.log(since=ctx.qi("since", 0), limit=ctx.qi("limit", 400))


def api_deprecate(ctx):
    """标记/取消废弃（单张或批量）。body: {out,key|keys,deprecated,reviewer,reason}。"""
    import webui.deprecate as dep_mod
    body = ctx.body or {}
    out = safe_path(body.get("out") or rel_to_root(C.DEFAULT_OUT), must_exist=True)
    keys = body.get("keys")
    if not keys:
        one = body.get("key")
        if not one:
            raise ApiError("缺少 key 或 keys")
        keys = [one]
    keys = [str(k).strip() for k in keys if str(k).strip()]
    if not keys:
        raise ApiError("key 列表为空")
    deprecated = bool(body.get("deprecated", True))
    reviewer = body.get("reviewer") or "webui"
    reason = body.get("reason") or ""
    if len(keys) == 1 and not body.get("bulk"):
        res = dep_mod.set_flag(out, keys[0], deprecated, reviewer=reviewer, reason=reason)
    else:
        res = dep_mod.bulk(out, keys, deprecated, reviewer=reviewer, reason=reason)
    res.update({"ok": True, "summary": dep_mod.summary(out)})
    return res


def api_deprecated_list(ctx):
    import webui.deprecate as dep_mod
    out = safe_path(ctx.q("out") or rel_to_root(C.DEFAULT_OUT), must_exist=True)
    return {"ok": True, "out": rel_to_root(out), "items": dep_mod.load(out),
            "summary": dep_mod.summary(out)}


def api_deprecate_sync(ctx):
    """把 sidecar 的废弃标记重新镜像进 meta（流水线重跑后使用）。"""
    import webui.deprecate as dep_mod
    out = safe_path((ctx.body or {}).get("out") or rel_to_root(C.DEFAULT_OUT), must_exist=True)
    return dict({"ok": True}, **dep_mod.sync_mirror(out))


def api_export_preflight(ctx):
    import webui.dataset as dataset
    return dataset.preflight(ctx.body or {})


def api_export_run(ctx):
    import webui.dataset as dataset
    return dataset.run_export(ctx.body or {})


def api_export_status(ctx):
    return {"ok": True, "task": dict(task_status(ctx.q("id")))}


def _version():
    from webui import __version__
    return __version__
