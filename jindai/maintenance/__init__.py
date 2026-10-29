"""库维护领域包：库内数据维护 + 磁盘→库的导入采集（2026-10-29 包化）。

原先是四个顶层散件（``jindai/maintenance.py``、``jindai/import_lists.py``、
``jindai/import_engine.py``、``jindai/import_cli.py``），现在按"同一领域、各自
单一职责"收进一个包：

- ``manager.py``       ``MaintenanceManager`` + ``maintenance_manager`` 单例
  （``/api/v2/maintenance/*`` 路由、worker 任务表、embedding 补算、OCR 任务）；
- ``import_lists.py``  导入清单 / 磁盘登记 / 盖章状态机 / 毒文件清单
  （裸 asyncpg，**不依赖 FastAPI、不依赖 ORM**）；
- ``import_engine.py`` stage 链构造 + 文件级并发 + 轮内进度快照
  （**只允许在子进程跑**：pymupdf 段错误必须隔离，见
  ``plugins/importer/supervisor.py``）；
- ``import_cli.py``    argparse 入口（``python -m jindai.maintenance.import_cli``；
  运维/看门狗实际拉起的是 ``batch_operation/batch_import_direct.py`` 进程）；
- ``fill_nontitle.py`` 登记后回填空 title/authors/item_type（幂等；原
  ``batch_operation/fill_nontitle_rows.py``，2026-10-29 迁入，核心包不再
  subprocess 拉运维脚本）。

运行时目录（报表/锁/状态/审计清单）的唯一来源在 ``jindai/paths.py``
（``REPORTS``，可用 config.yaml 的 ``import_reports_dir`` 覆盖），本包与
``plugins/trash`` 都从那里取，不再各自拼路径。

⚠️ 本 ``__init__`` 刻意**不** import ``manager``：导入引擎的 CLI 子进程只用得到
``import_lists`` / ``import_engine``，不该被 FastAPI + watchdog 那套拖起来。
``maintenance_manager`` 用 PEP 562 惰性导出，
``from jindai.maintenance import maintenance_manager`` 的写法保持不变。
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期
    from .manager import MaintenanceManager, maintenance_manager

__all__ = ["MaintenanceManager", "maintenance_manager"]

# 惰性导出的名字 -> 真实模块（PEP 562）
_LAZY = {"MaintenanceManager", "maintenance_manager"}


def __getattr__(name: str) -> Any:
    """按需导入 manager：避免 CLI 子进程被 web 侧依赖污染。"""
    if name in _LAZY:
        from . import manager
        return getattr(manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(__all__)
