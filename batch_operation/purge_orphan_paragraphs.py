"""Purge orphan paragraphs (source file gone from file_metadata) and their queue rows.

An orphan is a paragraph whose ``source_id`` has no ``file_metadata`` row —
the file was deleted (or replaced by a path migration) but paragraph.source_id
is a bare UUID with no FK, so the rows stayed behind forever: 4,117,146 rows
as of 2026-09-29. They can never receive embeddings (embeddings FK points at
file_metadata) and they pollute every missing-embedding statistic.

Plan (idempotent, resumable — counts are reported per phase):
  1. Materialize orphan ids once into an UNLOGGED table (single anti-join scan).
  2. DELETE embedding_pending_queue rows for those ids (PK probe, fast).
  3. DELETE paragraph rows for those ids (PK probe across 32 partitions).
  4. Drop the scratch table; print before/after totals.

  python batch_operation/purge_orphan_paragraphs.py --dry-run   # counts only
  python batch_operation/purge_orphan_paragraphs.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import common as C

log = C.setup_logging("purge_orphans", "batch_operation/reports/purge_orphans.log")

SCRATCH = "orphan_para_ids"
COLLECT_SQL = f"""
    CREATE UNLOGGED TABLE {SCRATCH} AS
    SELECT p.id FROM paragraph p
    LEFT JOIN file_metadata f ON f.id = p.source_id
    WHERE f.id IS NULL
"""
COUNT_SQL = f"SELECT count(*) FROM {SCRATCH}"
DEL_QUEUE_SQL = f"DELETE FROM embedding_pending_queue WHERE id IN (SELECT id FROM {SCRATCH})"
DEL_PARA_SQL = f"DELETE FROM paragraph WHERE id IN (SELECT id FROM {SCRATCH})"
DROP_SQL = f"DROP TABLE IF EXISTS {SCRATCH}"
TOTAL_PARA_SQL = "SELECT count(*) FROM paragraph"
TOTAL_QUEUE_SQL = "SELECT count(*) FROM embedding_pending_queue"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="collect + count only (default when --apply absent)")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        await conn.execute(DROP_SQL)  # idempotent re-run
        pre_p = await conn.fetchval(TOTAL_PARA_SQL)
        pre_q = await conn.fetchval(TOTAL_QUEUE_SQL)
        log.info("before: paragraphs=%s queue=%s", f"{pre_p:,}", f"{pre_q:,}")

        t0 = time.time()
        await conn.execute(COLLECT_SQL)
        n = await conn.fetchval(COUNT_SQL)
        log.info("orphans materialized: %s ids in %.1fs", f"{n:,}", time.time() - t0)
        if not n:
            await conn.execute(DROP_SQL)
            log.info("nothing to do")
            return 0
        if not args.apply:
            summary = {"orphan_paragraphs": n, "applied": False}
            log.info("dry-run summary: %s", json.dumps(summary))
            await conn.execute(DROP_SQL)
            return 0

        t0 = time.time()
        dq = await conn.execute(DEL_QUEUE_SQL)
        log.info("queue rows deleted: %s (%.1fs)", dq, time.time() - t0)
        t0 = time.time()
        dp = await conn.execute(DEL_PARA_SQL)
        log.info("paragraph rows deleted: %s (%.1fs)", dp, time.time() - t0)
        await conn.execute(DROP_SQL)

        post_p = await conn.fetchval(TOTAL_PARA_SQL)
        post_q = await conn.fetchval(TOTAL_QUEUE_SQL)
        summary = {
            "orphan_paragraphs_found": n,
            "queue_deleted": int(dq.split()[-1]),
            "paragraphs_deleted": int(dp.split()[-1]),
            "paragraphs_before": pre_p, "paragraphs_after": post_p,
            "queue_before": pre_q, "queue_after": post_q,
            "applied": True,
        }
        log.info("done: %s", json.dumps(summary))
        C.dump_json("batch_operation/reports/purge_orphans.json", summary)
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
