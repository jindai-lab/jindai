"""Drain embedding_pending_queue rows that can never be processed.

Queue states:
  stale  = paragraph already fully embedded (consumer should have consumed it)
  orphan = queue row whose paragraph row no longer exists (queue has no FK)

Deletes run in small LIMIT'ed batches so no statement ever holds a long
lock on the 35M-row queue table.

  python batch_operation/clean_queue.py --dry-run          # count only
  python batch_operation/clean_queue.py --what both --batch 20000
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import common as C

log = C.setup_logging("clean_queue", "batch_operation/reports/clean_queue.log")

STALE_SQL = """
    DELETE FROM embedding_pending_queue
    WHERE ctid IN (
        SELECT ctid FROM embedding_pending_queue
        WHERE EXISTS (SELECT 1 FROM text_embeddings e
                      WHERE e.id = embedding_pending_queue.id
                        AND e.source_id = embedding_pending_queue.source_id)
        LIMIT $1)
"""
ORPHAN_SQL = """
    DELETE FROM embedding_pending_queue
    WHERE ctid IN (
        SELECT ctid FROM embedding_pending_queue
        WHERE NOT EXISTS (SELECT 1 FROM paragraph p
                          WHERE p.id = embedding_pending_queue.id
                            AND p.source_id = embedding_pending_queue.source_id)
        LIMIT $1)
"""
COUNT_STALE_SQL = """
    SELECT count(*) FROM embedding_pending_queue q
    WHERE EXISTS (SELECT 1 FROM text_embeddings e
                  WHERE e.id = q.id AND e.source_id = q.source_id)
"""
COUNT_ORPHAN_SQL = """
    SELECT count(*) FROM embedding_pending_queue q
    WHERE NOT EXISTS (SELECT 1 FROM paragraph p
                      WHERE p.id = q.id AND p.source_id = q.source_id)
"""


async def drain(conn, sql, batch: int, label: str, max_batches: int) -> int:
    total = 0
    rounds = 0
    while True:
        n = await conn.execute(sql, batch)
        n = int(n.split()[-1])
        total += n
        rounds += 1
        if n < batch or (max_batches and rounds >= max_batches):
            break
        if rounds % 20 == 0:
            log.info("%s: removed %s so far...", label, f"{total:,}")
    return total


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["stale", "orphan", "both"], default="both")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        if args.dry_run:
            if args.what in ("stale", "both"):
                log.info("stale rows: %s", f"{await conn.fetchval(COUNT_STALE_SQL):,}")
            if args.what in ("orphan", "both"):
                log.info("orphan rows: %s", f"{await conn.fetchval(COUNT_ORPHAN_SQL):,}")
            log.info("queue total: %s",
                     f"{await conn.fetchval('SELECT count(*) FROM embedding_pending_queue'):,}")
            return 0
        removed = {}
        if args.what in ("stale", "both"):
            removed["stale"] = await drain(
                conn, STALE_SQL, args.batch, "stale", args.max_batches
            )
        if args.what in ("orphan", "both"):
            removed["orphan"] = await drain(
                conn, ORPHAN_SQL, args.batch, "orphan", args.max_batches
            )
        log.info("removed: %s", removed)
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
