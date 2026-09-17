# -*- coding: utf-8 -*-
"""端到端自检：对已启动的 Web 服务做接口与流程回归。

用法:
    ./run_webui.sh &                       # 或 python -m webui.server
    python -m webui.selftest               # 默认 http://127.0.0.1:8765
    python -m webui.selftest --base http://127.0.0.1:8790 --skip-job

覆盖: 状态/列表/meta/预览图/静态资源 → 任务子进程跑通 BaseLine → 编辑回写与备份/恢复
      → 训练集导出（含 data.yaml） → 统计重算。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = []


def call(base, path, params=None, body=None, method=None, raw=False, timeout=120):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if body is not None else "GET"))
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            if raw:
                return resp.status, payload, dict(resp.headers)
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError("HTTP %s %s -> %s" % (exc.code, path, detail))


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print("  %s %-46s %s" % ("PASS" if cond else "FAIL", name, detail))
    return bool(cond)


def wait_task(base, tid, timeout=300, key="task"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = call(base, "/api/task", {"id": tid})[key]
        if t["status"] != "running":
            return t
        time.sleep(1.0)
    return {"status": "timeout"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m webui.selftest")
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="AutoLabel_webui_test", help="自检用输出目录")
    ap.add_argument("--export-dest", default="TrainSet_selftest")
    ap.add_argument("--skip-job", action="store_true", help="跳过任务子进程测试")
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    out_dir = ROOT / args.out
    print("== 自检目标 %s ==" % base)

    # 1. 服务与静态资源
    try:
        st = call(base, "/api/state", {"out": args.out})
        check("state 接口", st.get("ok") and st.get("port"), "port=%s v%s" % (st.get("port"), st.get("version")))
        check("默认参数下发", all(k in st.get("defaults", {}) for k in ("expand", "conf", "class_mode", "teachers")))
    except Exception as exc:
        check("state 接口", False, str(exc))
        return 1
    for path, ctype in (("/", "text/html"), ("/static/app.js", "javascript"), ("/static/style.css", "text/css")):
        try:
            status, body, hdr = call(base, path, raw=True, timeout=30)
            check("静态资源 %s" % path, status == 200 and ctype in hdr.get("Content-Type", ""),
                  "%d bytes" % len(body))
        except Exception as exc:
            check("静态资源 %s" % path, False, str(exc))

    # 2. 任务（BaseLine 11 张）
    if not args.skip_job:
        try:
            r = call(base, "/api/job/start", body={
                "images": "BaseLine", "out": args.out, "limit": 0, "no_val": True,
                "workers": 1, "panels": 11, "sheet_cols": 4, "max_det": 20,
            })
            jid = r["job"]["id"]
            t0 = time.time()
            while time.time() - t0 < 240:
                s = call(base, "/api/job/status")["job"]
                if not s["running"]:
                    break
                time.sleep(1.5)
            s = call(base, "/api/job/status")["job"]
            check("任务跑通 BaseLine", s["status"] == "done" and s["returncode"] == 0,
                  "状态=%s rc=%s 进度=%s" % (s["status"], s["returncode"], s["progress"].get("done")))
            log = call(base, "/api/job/log", {"since": 0, "limit": 500})
            check("日志增量返回", log["lines"] and log["next"] > 0, "%d 行" % len(log["lines"]))
            check("进度聚合", (s["progress"].get("done") or 0) >= 11, "done=%s" % s["progress"].get("done"))
            n_meta = len(list((out_dir / "meta").rglob("*.json"))) if (out_dir / "meta").is_dir() else 0
            check("标签落盘", n_meta == 11, "meta=%d" % n_meta)
        except Exception as exc:
            check("任务跑通 BaseLine", False, str(exc))

    # 3. 列表 / meta / 预览图
    try:
        r = call(base, "/api/images", {"out": args.out, "filter": "all", "limit": 3})
        check("列表接口", r["total"] >= 11 and len(r["items"]) == 3, "total=%d" % r["total"])
        key = r["items"][0]["key"]
    except Exception as exc:
        check("列表接口", False, str(exc))
        return 1
    try:
        m = call(base, "/api/meta", {"out": args.out, "key": key})
        o = m["objects"][0]
        check("meta 结构", m["ok"] and len(o["quad_final_px"]) == 4 and "flags" in o,
              "key=%s n_obj=%d" % (key, m["n_obj"]))
    except Exception as exc:
        check("meta 结构", False, str(exc))
        return 1
    try:
        status, body, hdr = call(base, "/api/image", {"out": args.out, "key": key, "max": 640}, raw=True, timeout=60)
        check("预览图渲染", body[:2] == b"\xff\xd8" and "X-Preview-Size" in hdr,
              "%s / %s" % (hdr.get("X-Image-Size"), hdr.get("X-Preview-Size")))
    except Exception as exc:
        check("预览图渲染", False, str(exc))

    # 4. 编辑回写 + 备份 + 恢复
    try:
        meta_before = call(base, "/api/meta", {"out": args.out, "key": key})
        q = [list(p) for p in meta_before["objects"][0]["quad_final_px"]]
        q[0][0] += 5.0
        q[0][1] += 3.0
        r = call(base, "/api/save", body={
            "out": args.out, "key": key, "reviewer": "selftest",
            "objects": [{"index": 0, "action": "update", "quad_final_px": q, "clear_review": True}],
        })
        new_q = r["meta"]["objects"][0]["quad_final_px"]
        check("编辑回写", abs(new_q[0][0] - q[0][0]) < 0.02 and r["meta"]["objects"][0].get("review_edit"),
              "四角更新 + review_edit 已记录")
        labels_path = ROOT / args.out / "labels" / (key + ".txt")
        txt = labels_path.read_text().splitlines()[0].split()
        expect = q[0][0] / meta_before["size"][1]
        check("YOLO 标签同步更新", abs(float(txt[5]) - expect) < 5e-4, "字段数=%d kp0_x=%.4f" % (len(txt), float(txt[5])))
        check("自动备份", (ROOT / r["backup"]["meta"]).exists(), r["backup"]["meta"])
        bk = call(base, "/api/backups", {"out": args.out, "key": key})
        check("备份列表", len(bk["backups"]) >= 1, "%d 份" % len(bk["backups"]))
        before_txt = (ROOT / r["backup"]["labels"]).read_text() if r["backup"]["labels"] else ""
        call(base, "/api/restore", body={"out": args.out, "key": key})
        after_txt = labels_path.read_text()
        check("恢复备份", after_txt == before_txt, "labels 与备份一致")
    except Exception as exc:
        check("编辑回写", False, str(exc))

    # 5. 导出
    try:
        pf = call(base, "/api/export/preflight", body={
            "out": args.out, "dest": args.export_dest, "filter": "all",
            "object_filter": "keep_all", "class_mode": "single", "image_mode": "symlink", "val_ratio": 0.2,
        })
        check("导出预检", pf["n_images"] >= 11 and pf["n_objects"] >= 11,
              "图=%d 目标=%d" % (pf["n_images"], pf["n_objects"]))
        r = call(base, "/api/export/run", body={
            "out": args.out, "dest": args.export_dest, "filter": "all", "object_filter": "keep_all",
            "class_mode": "single", "image_mode": "symlink", "val_ratio": 0.2, "overwrite": True,
        })
        t = wait_task(base, r["task"]["id"])
        yaml_path = ROOT / args.export_dest / "data.yaml"
        check("训练集导出", t["status"] == "done" and yaml_path.exists(),
              "状态=%s 图=%s" % (t["status"], t.get("result", {}).get("counts")))
        if yaml_path.exists():
            text = yaml_path.read_text()
            check("data.yaml 内容", "kpt_shape: [4, 2]" in text and "train:" in text and "names:" in text,
                  yaml_path.name)
        n_lab = len(list((ROOT / args.export_dest / "labels" / "train").glob("*.txt")))
        check("导出标签文件", n_lab >= 1, "labels/train=%d" % n_lab)
    except Exception as exc:
        check("训练集导出", False, str(exc))

    # 6. 统计重算
    try:
        r = call(base, "/api/stats/refresh", body={"out": args.out, "panels": 0})
        t = wait_task(base, r["task"]["id"])
        stats = json.loads((ROOT / args.out / "stats.json").read_text())
        check("统计重算", t["status"] == "done" and stats["images"]["total"] >= 11,
              "images=%d objects=%d" % (stats["images"]["total"], stats["objects"]["total"]))
        d = call(base, "/api/stats", {"out": args.out})
        check("仪表盘数据", d["index"]["hist"]["plate_w"]["counts"] is not None and d["stats"]["images"]["total"] >= 11,
              "直方图 %d 桶" % len(d["index"]["hist"]["plate_w"]["counts"]))
    except Exception as exc:
        check("统计重算", False, str(exc))

    ok = sum(1 for r in RESULTS if r[0])
    print("\n== 自检结果: %d/%d 通过 ==" % (ok, len(RESULTS)))
    for good, name, detail in RESULTS:
        if not good:
            print("   FAIL %s %s" % (name, detail))
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
