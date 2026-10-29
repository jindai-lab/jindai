"""内置回收站插件（plugins/trash）。

按功能把回收站相关组件收拢为一个内置插件：

* ``core`` —— 删除文献记录时把关联文件搬进 ``<storage>/.trash/<批次>/``
  （纯文件系统操作，不碰数据库；原 ``jindai/trash.py``）。其它插件
  （如 bibliography）直接 ``from plugins.trash import move_to_trash``。
* ``purge.TrashPurgeWorkerPlugin`` —— 常驻清空 worker：每周一 06:00 清空回收站
  （安全阀：批次内仍有 ``file_metadata`` 引用则整批跳过；原
  ``jindai/trash_purge.py``）。

worker 的启停由 app lifespan → ``PluginManager.startup()/shutdown()`` 统一管理。
"""

from .core import (MANIFEST_LOG, TRASH_DIRNAME, is_trash_path, move_to_trash,
                   trash_root)
from .purge import TrashPurgeWorkerPlugin

__all__ = [
    "TrashPurgeWorkerPlugin",
    "TRASH_DIRNAME",
    "MANIFEST_LOG",
    "is_trash_path",
    "move_to_trash",
    "trash_root",
]
