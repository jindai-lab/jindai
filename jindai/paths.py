"""运行时路径的唯一来源（报表 / 锁 / 状态 / 审计清单）。

这些目录在**运行时**才存在，与代码无关；默认沿用历史位置
``<项目根>/batch_operation/reports``（容器内 ``/app/batch_operation/reports`` ≡
宿主 ``~/jindai-lab/jindai/batch_operation/reports``，bind mount 同一目录）——那里
有人工维护的 ``import_skip.txt`` / ``import_defer.txt``、监控 grep 的 ``*.log``、
毒文件与在飞标记，所以默认值刻意不动。要整体迁走只改 config.yaml：

    import_reports_dir: /app/var/reports      # 绝对路径

谁在用：``jindai/maintenance/import_lists.py``（清单缓存/盖章报表）、
``jindai/maintenance/import_engine.py``（stage/run 日志、锁、inflight、状态快照）、
``jindai/maintenance/fill_nontitle.py``、``plugins/trash/core.py``（审计清单）。

⚠️ 本模块刻意只依赖 pathlib（config 也是延迟导入），不引入 asyncpg/ORM，
   这样 ``plugins/trash/core.py`` 仍然可以脱离数据库单独测试。
"""
from __future__ import annotations

from pathlib import Path

# 项目根：本文件在 <root>/jindai/paths.py，故 parents[1]。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPORTS = PROJECT_ROOT / "batch_operation" / "reports"


def _resolve_reports_dir() -> Path:
    """config.import_reports_dir（绝对路径）优先，否则用历史默认位置。"""
    try:
        from .config import config
        custom = str(getattr(config, "import_reports_dir", "") or "").strip()
    except Exception:  # noqa: BLE001 - 无 config 的离线场景直接落默认位置
        custom = ""
    return Path(custom) if custom else _DEFAULT_REPORTS


REPORTS = _resolve_reports_dir()

__all__ = ["PROJECT_ROOT", "REPORTS"]
