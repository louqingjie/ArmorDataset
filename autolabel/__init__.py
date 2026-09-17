"""装甲板自动标注（知识蒸馏伪标签）流水线。

流程: 双教师全图推理 → 1.5× ROI 裁剪 → sp_vision_25 传统视觉角点精修
      → 一致性/门限过滤 → YOLO-pose txt + meta json + 复核清单/可视化/统计报告

用法: python -m autolabel.run --help
"""
__version__ = "1.0.0"

# GPU 可用时预加载 CUDA 运行库（缺库/无 GPU 时静默跳过，自动回退 CPU）
from . import gpu as _gpu          # noqa: E402

_gpu.preload_cuda_libraries()
