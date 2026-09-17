"""装甲板自动标注 Web 工作台（零新增依赖）。

组成:
  server.py   标准库 ThreadingHTTPServer + 正则路由 + 静态资源 + 端口回退
  api.py      列表/详情/预览图/统计/后台任务等只读与统计接口（内存索引 + LRU 缓存）
  jobs.py     autolabel.run 子进程管理（启动/停止/续跑、进度聚合、日志增量）
  edits.py    人工修正校验与标签回写（备份 + review_edit 审计）
  dataset.py  训练集导出（过滤/划分/软链/class-mode 重写/data.yaml）
  static/     前端单页（原生 HTML/CSS/JS/Canvas，离线可用）

启动: ./run_webui.sh            (等价于 python -m webui.server --port 8765)
"""
__version__ = "1.0.0"
