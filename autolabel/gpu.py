# -*- coding: utf-8 -*-
"""CUDA 运行库预加载：让 onnxruntime-gpu 在未设置 LD_LIBRARY_PATH 时也能用上 GPU。

背景：pip 安装的 NVIDIA 运行库位于 `site-packages/nvidia/<pkg>/lib`（或项目内
`third_party/*/nvidia/<pkg>/lib`），该目录**不在动态链接器搜索路径中**，ORT 的
CUDA EP 因此报 `Failed to load library ... libcublasLt.so.12` 并静默回退 CPU。

这里在创建 InferenceSession 之前，用 ctypes 以 RTLD_GLOBAL 预加载这些 .so，
使 provider 的 DT_NEEDED（libcublasLt / libcublas / libcudart / libcurand /
libcufft / libcudnn）能在全局符号表中解析到，从而真正启用 GPU。

开关：环境变量 `AUTOLABEL_GPU=0` 关闭预加载；`AUTOLABEL_CUDA_LIB_DIRS=a:b` 追加搜索目录。
"""
from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from pathlib import Path

# 预加载顺序：底层运行时 → NVRTC/nvJitLink → 数学库 → cuDNN
PRIORITY = (
    "libcudart.so", "libnvrtc.so", "libnvjitlink.so", "libnvfatbin.so",
    "libcublasLt.so", "libcublas.so", "libcufft.so", "libcurand.so",
    "libcusolver.so", "libcusparse.so", "libcusparselt.so", "libnvshmem.so",
    "libcudnn.so", "libcudnn_graph.so", "libcudnn_ops.so", "libcudnn_cnn.so",
    "libcudnn_adv.so", "libcudnn_engines_precompiled.so",
    "libcudnn_engines_runtime_compiled.so", "libcudnn_heuristic.so",
)

_LOADED = None


def candidate_dirs():
    """按优先级返回候选库目录：环境变量 → 项目 third_party → 当前环境 site-packages。"""
    dirs = []
    env = os.environ.get("AUTOLABEL_CUDA_LIB_DIRS")
    if env:
        dirs += [Path(p) for p in env.split(os.pathsep) if p.strip()]
    root = Path(__file__).resolve().parent.parent
    dirs += sorted((root / "third_party").glob("*/nvidia/*/lib"))
    roots = []
    try:
        roots += list(site.getsitepackages())
    except Exception:
        pass
    roots.append(str(Path(sys.prefix) / "lib" / ("python%d.%d" % sys.version_info[:2]) / "site-packages"))
    for r in roots:
        dirs += sorted((Path(r) / "nvidia").glob("*/lib"))
    dirs += [Path("/usr/local/cuda/lib64")] if Path("/usr/local/cuda/lib64").is_dir() else []
    seen, out = set(), []
    for d in dirs:
        if d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def _pick_libs(dirs):
    """按 PRIORITY 逐个 SONAME 前缀挑选实际文件（先命中的目录优先）。"""
    files, used = [], set()
    for prefix in PRIORITY:
        for d in dirs:
            cands = sorted(glob.glob(str(d / (prefix + "*"))))
            cands = [c for c in cands if c.endswith(".so") or ".so." in c]
            if cands:
                f = cands[0] if not prefix.endswith("Lt.so") else next(
                    (c for c in cands if "Lt" in c), cands[0])
                if f not in used:
                    used.add(f)
                    files.append(f)
                break
    return files


def preload_cuda_libraries(verbose=False, force=False):
    """幂等预加载；返回 {'enabled','dirs','loaded','failed'} 供日志/诊断。"""
    global _LOADED
    if _LOADED is not None and not force:
        return _LOADED
    if str(os.environ.get("AUTOLABEL_GPU", "1")).lower() in ("0", "false", "no", "off"):
        _LOADED = {"enabled": False, "dirs": [], "loaded": [], "failed": []}
        return _LOADED

    dirs = candidate_dirs()
    loaded, failed = [], []
    for f in _pick_libs(dirs):
        try:
            ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
            loaded.append(Path(f).name)
        except OSError as exc:
            failed.append("%s: %s" % (Path(f).name, exc))
    _LOADED = {"enabled": True, "dirs": [str(d) for d in dirs],
               "loaded": loaded, "failed": failed}
    if verbose:
        print("[gpu] 预加载 %d 个库（失败 %d）: %s" % (len(loaded), len(failed), ", ".join(loaded)))
        for msg in failed:
            print("[gpu]   失败:", msg)
    return _LOADED


def describe():
    """诊断信息：预加载结果 + ORT 版本/可用 provider。"""
    info = dict(preload_cuda_libraries())
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["providers"] = ort.get_available_providers()
    except Exception as exc:
        info["onnxruntime_error"] = str(exc)
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), ensure_ascii=False, indent=1))
