"""内置 PDF 导入插件（plugins/importer）。

* ``supervisor.ImportWorkerPlugin`` —— 常驻导入 supervisor：每轮探测清单/磁盘
  变动，有待导入文件时以**子进程**拉起导入引擎（段错误隔离），做崩溃归因与
  毒文件黑名单，并向 ``/import/`` 路由暴露状态。原 ``jindai/import_worker.py``。
* ``import_lists`` —— 导入清单 / 磁盘登记 / 盖章状态机 / 毒文件清单
  （裸 asyncpg，**不依赖 FastAPI、不依赖 ORM**）。
* ``import_engine`` —— stage 链构造 + 文件级并发 + 轮内进度快照
  （**只允许在子进程跑**：pymupdf 段错误必须隔离，见 ``supervisor.py``）。
* ``import_cli`` —— argparse 入口（``python -m plugins.importer.import_cli``；
  运维/看门狗实际拉起的是 ``batch_operation/batch_import_direct.py`` 进程）。
* ``fill_nontitle`` —— 登记后回填 title/authors/item_type（幂等；原
  ``batch_operation/fill_nontitle_rows.py``，CLI：
  ``python -m plugins.importer.fill_nontitle``）。

2026-10-30 按"导入领域全部归到导入插件"把 ``jindai/maintenance/`` 下的
``import_lists`` / ``import_engine`` / ``import_cli`` / ``fill_nontitle``
整体搬进本包；核心 ``jindai/maintenance.py`` 随之退回**单文件**，只剩库内维护
（``/api/v2/maintenance/*`` 路由、worker 任务表、embedding 补算、OCR 任务）。
`jindai/` 核心不再 import 本插件，依赖方向保持 plugins → jindai。

worker 的启停由 app lifespan → ``PluginManager.startup()/shutdown()`` 统一管理。
"""

from .supervisor import ImportWorkerPlugin

__all__ = ["ImportWorkerPlugin"]
