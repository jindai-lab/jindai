"""One-shot status report for the jindai batch pipeline.

Shows: endpoint config + live API check, OneDrive PDFs still lacking
paragraphs, paragraphs still lacking embeddings, and queue health.

  python batch_operation/stats.py            # ~1 min (default checks)
  python batch_operation/stats.py --deep     # + queue stale/orphan breakdown
  python batch_operation/stats.py --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import common as C

log = C.setup_logging("stats")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", action="store_true",
                    help="also compute queue stale/orphan breakdown (slow, ~10min)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    conn = await C.get_conn()
    try:
        t = time.time()
        out["embedding_endpoint"] = {
            "url": C.EMBED_URL,
            "model": C.EMBED_MODEL,
            "dims": C.DIMS,
            "key_configured": bool(C.EMBED_KEY),
        }
        # live API check
        ec = C.EmbeddingClient(timeout=30, retries=1)
        try:
            t0 = time.time()
            vecs = await ec.embed(["ping"])
            out["api_check"] = {
                "ok": True, "dims": len(vecs[0]),
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as ex:  # noqa: BLE001
            out["api_check"] = {"ok": False, "error": str(ex)}
        finally:
            await ec.aclose()

        out["onedrive_pdfs_without_paragraph"] = await conn.fetchval("""
            SELECT count(*) FROM file_metadata f
            WHERE f.path LIKE 'OneDrive/%' AND lower(f.path) LIKE '%.pdf'
              AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
        """)
        out["paragraphs_without_embedding"] = await conn.fetchval("""
            SELECT count(*) FROM paragraph p
            WHERE NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
        """)
        out["onedrive_paragraphs_without_embedding"] = await conn.fetchval("""
            SELECT count(*) FROM paragraph p
            WHERE EXISTS (SELECT 1 FROM file_metadata f
                          WHERE f.id = p.source_id AND f.path LIKE 'OneDrive/%')
              AND NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
        """)
        out["embedding_rows"] = await conn.fetchval(
            "SELECT count(*) FROM text_embeddings"
        )
        out["queue_total"] = await conn.fetchval(
            "SELECT count(*) FROM embedding_pending_queue"
        )
        out["default_elapsed_s"] = round(time.time() - t, 1)

        if args.deep:
            t = time.time()
            out["queue_stale"] = await conn.fetchval("""
                SELECT count(*) FROM embedding_pending_queue q
                WHERE EXISTS (SELECT 1 FROM text_embeddings e
                              WHERE e.id = q.id AND e.source_id = q.source_id)
            """)
            out["queue_orphan"] = await conn.fetchval("""
                SELECT count(*) FROM embedding_pending_queue q
                WHERE NOT EXISTS (SELECT 1 FROM paragraph p
                                  WHERE p.id = q.id AND p.source_id = q.source_id)
            """)
            out["deep_elapsed_s"] = round(time.time() - t, 1)
    finally:
        await conn.close()

    for k, v in out.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            log.info("%-42s %s", k, f"{v:,}" if isinstance(v, int) else v)
        else:
            log.info("%-42s %s", k, json.dumps(v, ensure_ascii=False)
                     if isinstance(v, (dict, list)) else v)
    if args.json:
        C.dump_json(args.json, out)
    healthy = out.get("api_check", {}).get("ok")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
