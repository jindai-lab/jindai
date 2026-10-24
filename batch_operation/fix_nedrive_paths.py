"""清理 neDrive/ 拼写变体路径的陈旧行（缺首字母 O 的死路径副本）。

判据（2026-10-24 实证，44 行全部满足）：
  * path LIKE 'neDrive/%'，盘上无该目录，文件实际在 OneDrive/ 同相对路径；
  * OneDrive 正主行存在且 import_state='complete'，正文段落是 neDrive 侧的超集
    （1720 段中 1712 逐字重复，8 段独有为重切分变体/OCR 垃圾，无真实内容）。

动作：先导出整行 TSV 备份到 reports/，再按 FK 顺序删
text_embeddings → paragraph → page_skip_state → embedding_pending_queue →
file_dataset → file_metadata（单事务，失败整体回滚）。

Dry-run 默认；--apply 才写。可安全重跑（目标行删完即 0 命中）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/app")
import asyncio  # noqa: E402

import common as C  # noqa: E402  (bootstrap: APP_ROOT on sys.path)

log = C.setup_logging("fix_nedrive_paths",
                      "batch_operation/reports/fix_nedrive_paths.log")

FIND_SQL = """
    SELECT n.id, n.path, t.id AS twin_id, t.path AS twin_path,
           st.status AS twin_status
    FROM file_metadata n
    JOIN file_metadata t ON t.path = 'OneDrive/' || substr(n.path, 9)
    LEFT JOIN file_import_state st ON st.source_id = t.id
    WHERE n.path LIKE 'neDrive/%'
"""
DELETE_SQLS = [
    ("text_embeddings", "DELETE FROM text_embeddings WHERE source_id = ANY($1::uuid[])"),
    ("paragraph", "DELETE FROM paragraph WHERE source_id = ANY($1::uuid[])"),
    ("page_skip_state", "DELETE FROM page_skip_state WHERE source_id = ANY($1::uuid[])"),
    ("embedding_pending_queue",
     "DELETE FROM embedding_pending_queue WHERE source_id = ANY($1::uuid[])"),
    ("file_dataset", "DELETE FROM file_dataset WHERE file_id = ANY($1::uuid[])"),
    ("file_metadata", "DELETE FROM file_metadata WHERE id = ANY($1::uuid[])"),
]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup",
                    default="batch_operation/reports/neDrive_rows_backup.tsv")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        rows = await conn.fetch(FIND_SQL)
        rows = [r for r in rows if r["twin_status"] == "complete"]
        no_twin = await conn.fetch(
            "SELECT path FROM file_metadata WHERE path LIKE 'neDrive/%'")
        log.info("neDrive rows with complete twin: %d (raw neDrive rows: %d)",
                 len(rows), len(no_twin))
        if not rows:
            log.info("nothing to do")
            return 0
        for r in rows[:5]:
            log.info("  %s -> twin %s", r["path"], r["twin_status"])
        if len(rows) > 5:
            log.info("  ... and %d more", len(rows) - 5)

        ids = [r["id"] for r in rows]
        backup = Path(args.backup)
        backup.parent.mkdir(parents=True, exist_ok=True)
        with open(backup, "w", encoding="utf-8") as f:
            f.write("# neDrive stale rows deleted by fix_nedrive_paths.py\n")
            f.write("# id\tpath\ttwin_id\ttwin_path\n")
            for r in rows:
                f.write(f"{r['id']}\t{r['path']}\t{r['twin_id']}\t{r['twin_path']}\n")
        log.info("backup: %s (%d rows)", backup, len(rows))

        if not args.apply:
            log.info("dry-run only, re-run with --apply to delete")
            return 0

        async with conn.transaction():
            for name, sql in DELETE_SQLS:
                tag = await conn.execute(sql, ids)
                log.info("  %s: %s", name, tag)
        log.info("done: deleted %d stale neDrive rows", len(ids))
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
