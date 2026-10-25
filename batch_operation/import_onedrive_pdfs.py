"""OneDrive/file_metadata 导入 —— 清单函数 re-export shim（2026-10-25）。

实现已迁至主库 `jindai/import_lists.py`（常驻导入 worker 的一部分）。
本文件保留旧模块名，供存量批处理脚本（classify_imported.py 等）继续
`from import_onedrive_pdfs import ...` 不受影响。

原文件里的 TaskDBO 克隆路线（clone_pipeline / main，经 task_dbo.pipeline
反序列化跑导入）已删除：那是 Calibre 时代的重复导入实现，2026-10-18 起
被 direct-stages 引擎取代，现在连薄壳都走 jindai.import_cli。
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/app")

from jindai.import_lists import (  # noqa: E402,F401
    IMPORT_STATE_TABLE,
    LIST_SQL,
    MISSING_PDF_SQL,
    PARA_TOTAL_SQL,
    REPORTS,
    SCAN_ITEM_TYPE,
    TOOLBOOK_RE,
    apply_skip_defer,
    build_file_list,
    cached_disk_inventory,
    count_covered_pages,
    ensure_import_state_table,
    get_conn,
    pending_counts,
    read_skip_file,
    record_import_state,
    scan_disk_upsert,
    stats,
    upsert_disk,
    walk_disk,
)

# 旧模块的 _COVERED_PAGES_SQL / _list_sql / _STATE_FILTER / _STATE_ORDER
# 私有名也 re-export（部分脚本引用）。
from jindai.import_lists import (  # noqa: E402,F401
    _COVERED_PAGES_SQL,
    _STATE_FILTER,
    _STATE_ORDER,
    _list_sql,
)

import logging  # noqa: E402

log = logging.getLogger("jindai.import")
