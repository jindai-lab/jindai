"""内置 PDF 导入插件（plugins/importer）。

* ``supervisor.ImportWorkerPlugin`` —— 常驻导入 supervisor：每轮探测清单/磁盘
  变动，有待导入文件时以**子进程**拉起导入引擎（段错误隔离），做崩溃归因与
  毒文件黑名单，并向 ``/import/`` 路由暴露状态。原 ``jindai/import_worker.py``。

worker 的启停由 app lifespan → ``PluginManager.startup()/shutdown()`` 统一管理。
"""

from .supervisor import ImportWorkerPlugin

__all__ = ["ImportWorkerPlugin"]
