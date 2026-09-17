# -*- coding: utf-8 -*-
"""装甲板学生模型训练包。

主线：YOLO26n-pose（kpt_shape=[4,2]，单类 armor）← 双教师伪标签；
对照：YOLOv8n-pose（与 szu2026 同族结构）；
颜色/编号：裁块小 CNN（部署友好，等价 sp_vision_25 的 yolov5.bin + tiny_resnet 链路）。

入口（都用 `--config` 指定配置，默认 training/config.yaml）：
  python -m training.prepare     导出训练集 + 生成模型 yaml + 构建分类头裁块数据集
  python -m training.train_pose  训练姿态学生模型（ultralytics）
  python -m training.train_cls   训练颜色/编号分类头（tiny CNN，导出 ONNX）
  python -m training.eval_kpts   评估：角点误差(px / %板宽)、四边形 IoU、检出率、颜色编号准确率

一键：./run_training.sh        （可用 ./run_training.sh training/config.yaml 指定配置）
"""
