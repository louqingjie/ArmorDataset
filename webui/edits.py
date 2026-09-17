# -*- coding: utf-8 -*-
"""人工修正的服务端校验与回写。

约定:
  * 前端提交"修改后的四点（原图像素）"与操作类型，服务端负责校验、备份、回写；
  * 回写完全复用 `labelio.write_edited_labels` → txt 13 字段格式、meta schema、
    原子写语义与流水线一致，不会产生第二套格式；
  * 每次保存前把 meta json 与 labels txt 备份到 `out/backup/<key>/<ts>_*`（保留最近 10 份），
    并在 meta 里记录 `review_edit`（本目标最后一次修改）与 `review_edits`（审计历史）。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

import make_table as MT
from autolabel import config as C
from autolabel import labelio
from autolabel.geom import quad_size

from webui import api

KEEP_BACKUPS = 10
KEEP_AUDIT = 20
REVIEW_PREFIXES = labelio.REVIEW_FLAG_PREFIXES
# 界面保存会重建 meta，这些由其它工具（废弃标记 / 目标级废弃 / 恢复）写入的字段需原样保留
PRESERVE_META_KEYS = ("deprecated", "dep_by", "dep_time", "dep_reason",
                      "object_deprecations", "restored_from", "restored_at")


# --------------------------------------------------------------------------- #
# 校验与转换
# --------------------------------------------------------------------------- #
def validate_quad(points, img_w, img_h):
    arr = np.asarray(points, np.float32)
    if arr.shape != (4, 2):
        raise api.ApiError("四角点必须是 4 个 [x, y]，收到 %s" % (list(arr.shape),))
    if not np.isfinite(arr).all():
        raise api.ApiError("四角点包含非法数值")
    arr = arr.copy()
    arr[:, 0] = np.clip(arr[:, 0], 0, float(img_w))
    arr[:, 1] = np.clip(arr[:, 1], 0, float(img_h))
    x, y = arr[:, 0], arr[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 4.0:
        raise api.ApiError("四点退化（面积 %.1f px² 过小）" % area)
    return arr


def to_internal(obj, img_w, img_h):
    """meta 目标结构 -> labelio 需要的内部结构（补 ndarray 与默认字段）。"""
    q = np.asarray(obj.get("quad_final_px"), np.float32).reshape(4, 2)
    qc = obj.get("quad_coarse_px")
    qc = np.asarray(qc, np.float32).reshape(4, 2) if qc else q.copy()
    out = {k: v for k, v in dict(obj).items()}
    out.pop("quad_final_px", None)
    out.pop("quad_coarse_px", None)
    out.pop("review", None)                       # 由 labelio 依据 flags 重新计算
    out["quad_final"] = q
    out["quad_coarse"] = qc
    out["flags"] = list(obj.get("flags") or [])
    out["source"] = obj.get("source") or "manual"
    out["reason"] = obj.get("reason")
    pw, bar = quad_size(q)
    out["plate_w"] = round(float(pw), 2)
    out["bar_len"] = round(float(bar), 2)
    return out


# --------------------------------------------------------------------------- #
# 备份
# --------------------------------------------------------------------------- #
def backup_dir(out_dir: Path, key: str) -> Path:
    return Path(out_dir) / "backup" / str(key)


def new_ts():
    """毫秒级时间戳：同一秒内多次备份/恢复也不会互相覆盖。"""
    t = time.time()
    return "%s-%03d" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(t)), int(t * 1000) % 1000)


def make_backup(out_dir: Path, key: str, kind: str = "manual"):
    """备份 meta + labels；文件名带来源标记（manual/auto），保留最新 KEEP-1 份 + 最旧 1 份。"""
    d = backup_dir(out_dir, key)
    d.mkdir(parents=True, exist_ok=True)
    ts = new_ts()
    bid = "%s_%s" % (ts, kind)                    # 备份标识（列表/恢复都用它）
    src_meta = Path(out_dir) / "meta" / (key + ".json")
    src_lab = Path(out_dir) / "labels" / (key + ".txt")
    if not src_meta.exists():
        raise api.ApiError("meta 不存在，无法备份: %s" % key, 404)
    meta_dst = d / ("%s_meta.json" % bid)
    n = 1
    while meta_dst.exists():                      # 极端情况下再退避避免覆盖
        n += 1
        bid = "%s-%d_%s" % (new_ts(), n, kind)
        meta_dst = d / ("%s_meta.json" % bid)
    shutil.copy2(src_meta, meta_dst)
    lab_dst = None
    if src_lab.exists():
        lab_dst = d / ("%s_labels.txt" % bid)
        shutil.copy2(src_lab, lab_dst)
    metas = sorted(d.glob("*_meta.json"))
    if len(metas) > KEEP_BACKUPS:
        # 保留最新 KEEP-1 份 + 最旧 1 份：自动保存会频繁产生快照，但要始终留得住
        # 「本次编辑前」的原始状态（否则恢复只能回到几步之前）
        for old in metas[1:-(KEEP_BACKUPS - 1)]:
            try:
                old.unlink()
                (d / old.name.replace("_meta.json", "_labels.txt")).unlink(missing_ok=True)
            except OSError:
                pass
    return {"ts": bid, "kind": kind, "dir": api.rel_to_root(d),
            "meta": api.rel_to_root(meta_dst),
            "labels": api.rel_to_root(lab_dst) if lab_dst else None}


def list_backups(out_dir: Path, key: str):
    d = backup_dir(out_dir, key)
    if not d.is_dir():
        return []
    metas = sorted(d.glob("*_meta.json"), reverse=True)
    res = []
    for m in metas:
        lab = d / m.name.replace("_meta.json", "_labels.txt")
        bid = m.name[:-len("_meta.json")]
        res.append({"ts": bid, "kind": "auto" if bid.endswith("_auto") else "manual",
                    "meta": api.rel_to_root(m),
                    "labels": api.rel_to_root(lab) if lab.exists() else None,
                    "mtime": round(m.stat().st_mtime, 3)})
    return res


# --------------------------------------------------------------------------- #
# 保存
# --------------------------------------------------------------------------- #
def save(payload):
    out = api.safe_path(payload.get("out") or api.rel_to_root(C.DEFAULT_OUT), must_exist=True)
    key = str(payload.get("key") or "").strip()
    if not key:
        raise api.ApiError("缺少 key")
    meta = api.read_meta(out, key)
    h, w = int(meta["size"][0]), int(meta["size"][1])
    objs = list(meta.get("objects") or [])
    actions = {}
    for a in payload.get("objects") or []:
        if a.get("index") is None:
            continue
        actions[int(a["index"])] = a
    reviewer = str(payload.get("reviewer") or "webui")[:40]
    auto = bool(payload.get("auto"))              # 自动保存（前端 debounce 触发）
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    bk = make_backup(out, key, kind="auto" if auto else "manual")

    edits, kept = [], []
    if payload.get("background"):
        edits.append({"action": "background", "removed": len(objs), "time": now})
    else:
        for i, o in enumerate(objs):
            a = actions.get(i)
            if a and a.get("action") == "delete":
                edits.append({"action": "delete", "index": i, "time": now,
                              "before_px": o.get("quad_final_px")})
                continue
            rec = dict(o)
            if a:
                changed = {}
                if a.get("quad_final_px"):
                    q = validate_quad(a["quad_final_px"], w, h)
                    before = rec.get("quad_final_px")
                    rec["quad_final_px"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in q]
                    changed["quad"] = {"before_px": before, "after_px": rec["quad_final_px"]}
                if a.get("color") is not None:
                    ci = int(a["color"])
                    if not 0 <= ci <= 3:
                        raise api.ApiError("颜色索引非法: %s" % ci)
                    rec["color"], rec["color_name"] = ci, MT.COLOR_NAMES.get(ci, "?")
                    changed["color"] = ci
                if a.get("num") is not None:
                    ni = int(a["num"])
                    if not 0 <= ni <= 8:
                        raise api.ApiError("编号索引非法: %s" % ni)
                    rec["num"], rec["num_name"] = ni, MT.NUM_NAMES.get(ni, "?")
                    changed["num"] = ni
                if a.get("clear_review"):
                    rec["flags"] = [f for f in (rec.get("flags") or [])
                                    if not any(f.startswith(p) for p in REVIEW_PREFIXES)]
                    if "reviewed" not in rec["flags"]:
                        rec["flags"].append("reviewed")
                    rec["reason"] = None
                    changed["clear_review"] = True
                if changed:
                    rec["review_edit"] = {"reviewer": reviewer, "time": now,
                                          "action": "update", **changed}
                    edits.append({"action": "update", "index": i, "time": now,
                                  "reviewer": reviewer, **changed})
            kept.append(rec)

    internal = [to_internal(o, w, h) for o in kept]
    audit = list(meta.get("review_edits") or [])
    audit.append({"reviewer": reviewer, "time": now, "auto": auto,
                  "n_edits": len(edits), "edits": edits[:12]})
    # 回写是"重建 meta"，需显式带上其它模块写入的字段（废弃标记、恢复记录等），
    # 否则一次界面保存就会把它们抹掉
    extra = {"review_edits": audit[-KEEP_AUDIT:],
             "n_dep_obj": sum(1 for o in kept if o.get("deprecated"))}
    for k in PRESERVE_META_KEYS:
        if k in meta:
            extra[k] = meta[k]
    res = labelio.write_edited_labels(
        out, key, w, h, internal,
        extra_meta=extra,
        image_name=meta.get("image"), image_path=meta.get("path"), rel=meta.get("rel"),
        teachers=meta.get("teachers") or [], run_config=meta.get("config") or {},
        mode=meta.get("class_mode") or C.CLASS_MODE)

    _sync_after_write(out, key)
    new_meta = api.read_meta(out, key)
    return {"ok": True, "key": key, "n_obj": res["n_obj"], "n_review": res["n_review"],
            "edits": edits, "backup": bk, "auto": auto, "meta": new_meta,
            "reviewed": [f == "reviewed" for o in new_meta["objects"] for f in (o.get("flags") or [])].count(True)}


def restore(payload):
    """恢复到最近一次（或指定 ts）的备份。"""
    out = api.safe_path(payload.get("out") or api.rel_to_root(C.DEFAULT_OUT), must_exist=True)
    key = str(payload.get("key") or "").strip()
    if not key:
        raise api.ApiError("缺少 key")
    backups = list_backups(out, key)
    if not backups:
        raise api.ApiError("该图没有可恢复的备份", 404)
    ts = str(payload.get("ts") or "").strip()
    pick = None
    for b in backups:
        if not ts or b["ts"] == ts:
            pick = b
            break
    if pick is None:
        raise api.ApiError("备份版本不存在: %s" % ts, 404)
    # 先选定目标备份，再为当前状态留一份现场（否则会取到刚生成的快照，等于没恢复）
    src_meta = api.safe_path(pick["meta"])
    cur = make_backup(out, key)
    shutil.copy2(src_meta, out / "meta" / (key + ".json"))
    lab_src = api.safe_path(pick["labels"]) if pick["labels"] else None
    lab_dst = out / "labels" / (key + ".txt")
    if lab_src and lab_src.exists():
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lab_src, lab_dst)
    elif lab_dst.exists():
        lab_dst.unlink()
    labelio.patch_meta(out, key, {"restored_from": pick["ts"],
                                  "restored_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _sync_after_write(out, key)
    return {"ok": True, "key": key, "restored_from": pick["ts"], "backup": cur,
            "meta": api.read_meta(out, key), "backups": list_backups(out, key)}


def _sync_after_write(out: Path, key=None):
    """回写后同步复核清单，并只刷新该 key 的索引条目（避免整库重建）。"""
    try:
        from autolabel import run as run_mod
        # 单 key 走增量（毫秒级）；全量重写要读 21k meta，交互式保存不能走那条路
        if key:
            run_mod.update_review_list(out, [key], log=lambda *a, **k: api.LOG.info(" ".join(str(x) for x in a)))
        else:
            run_mod.write_review_list(out, log=lambda *a, **k: api.LOG.info(" ".join(str(x) for x in a)))
    except Exception:
        api.LOG.warning("同步 review_list.csv 失败", exc_info=True)
    if key:
        api.refresh_index_entry(out, key)
    else:
        api.invalidate_index(out)
    api._STATS_CACHE.pop(str(Path(out).resolve()), None)
