# -*- coding: utf-8 -*-
"""废弃标记：被废弃的图片不参与训练集导出。

存储设计:
  * 权威数据存 `out/deprecated.json`（sidecar）—— 因为流水线重跑（`autolabel.run`）会整份重写
    `meta/*.json`，只有 sidecar 能在重跑后仍保留废弃标记；
  * 同时把 `deprecated: true/false` 镜像写入 `meta/<key>.json`，便于直接查看/离线核对
    （镜像丢失时可用 `sync_mirror()` 或 CLI `--sync-deprecated` 补写）。

对外接口: load / set_flag / bulk / sync_mirror / summary
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from autolabel import labelio

from webui import api

FILENAME = "deprecated.json"
_cache = {}          # out -> (mtime, items)


def path_of(out_dir) -> Path:
    return Path(out_dir) / FILENAME


def load(out_dir) -> dict:
    """读取废弃表（按 mtime 缓存）。返回 {key: {by, time, reason}}。"""
    p = path_of(out_dir)
    if not p.exists():
        return {}
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    ck = str(Path(out_dir).resolve())
    c = _cache.get(ck)
    if c and c[0] == mt:
        return c[1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        items = dict(data.get("items") or {})
    except Exception as exc:
        api.LOG.warning("废弃表解析失败 %s: %s", p, exc)
        items = {}
    _cache[ck] = (mt, items)
    return items


def save(out_dir, items) -> dict:
    payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "count": len(items), "items": items}
    labelio._atomic_write(path_of(out_dir), json.dumps(payload, ensure_ascii=False, indent=1))
    _cache.pop(str(Path(out_dir).resolve()), None)
    return payload


def set_flag(out_dir, key, deprecated, reviewer="webui", reason="", mirror=True):
    """标记/取消废弃单张图。返回 {'key','deprecated','total'}。"""
    out_dir = Path(out_dir)
    items = dict(load(out_dir))
    if deprecated:
        items[key] = {"by": str(reviewer)[:40],
                      "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "reason": str(reason or "")[:120]}
    else:
        items.pop(key, None)
    save(out_dir, items)
    if mirror:
        try:
            labelio.patch_meta(out_dir, key, {"deprecated": bool(deprecated)})
        except Exception:
            api.LOG.warning("镜像 deprecated 到 meta 失败: %s", key, exc_info=True)
    _invalidate(out_dir, [key])
    return {"key": key, "deprecated": bool(deprecated), "total": len(items)}


def bulk(out_dir, keys, deprecated, reviewer="webui", reason="", mirror=True):
    out_dir = Path(out_dir)
    items = dict(load(out_dir))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for k in keys:
        k = str(k).strip()
        if not k:
            continue
        if deprecated:
            items[k] = {"by": str(reviewer)[:40], "time": now, "reason": str(reason or "")[:120]}
        else:
            items.pop(k, None)
        if mirror:
            try:
                labelio.patch_meta(out_dir, k, {"deprecated": bool(deprecated)})
            except Exception:
                pass
        n += 1
    save(out_dir, items)
    _invalidate(out_dir, [str(k.strip()) for k in keys if str(k).strip()])
    return {"n": n, "deprecated": bool(deprecated), "total": len(items)}


def sync_mirror(out_dir):
    """把 sidecar 里的废弃标记重新写回 meta（流水线重跑后可调用）。"""
    out_dir = Path(out_dir)
    items = load(out_dir)
    ok = fail = 0
    for k in items:
        try:
            labelio.patch_meta(out_dir, k, {"deprecated": True})
            ok += 1
        except Exception:
            fail += 1
    return {"synced": ok, "failed": fail, "total": len(items)}


def summary(out_dir):
    items = load(out_dir)
    by = {}
    for v in items.values():
        by[v.get("by") or "?"] = by.get(v.get("by") or "?", 0) + 1
    return {"count": len(items), "by": by,
            "updated": (json.loads(path_of(out_dir).read_text(encoding="utf-8")).get("updated")
                        if path_of(out_dir).exists() else None)}


def _invalidate(out_dir, keys=None):
    """只刷新被改动的索引条目；统计缓存置空（下次打开仪表盘按需重算）。"""
    if keys:
        for k in keys:
            api.refresh_index_entry(out_dir, k)
    else:
        api.invalidate_index(out_dir)
    api._STATS_CACHE.pop(str(Path(out_dir).resolve()), None)
