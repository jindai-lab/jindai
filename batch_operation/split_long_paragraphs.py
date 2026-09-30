"""Split over-long paragraphs into <=2000-char pieces at the last punctuation.

Rule (user spec): a paragraph longer than LIMIT chars is cut into pieces such
that each piece is at most LIMIT chars, and the cut falls immediately after the
LAST punctuation mark inside the LIMIT window. If the window contains no
punctuation at all, fall back to a hard cut at exactly LIMIT chars.

Scope: paragraphs of source files under OneDrive/ (the PDF import), all of
which come from PDF extraction. Idempotent: split pieces are <= LIMIT so a
re-run only picks up whatever is still too long (e.g. from an import that is
still running).

Per batch (transactional, 500 paragraphs):
  1. DELETE old text_embeddings rows for the old paragraph ids (content
     changed -> vectors must be recomputed).
  2. DELETE old embedding_pending_queue rows for those ids.
  3. DELETE the old paragraph rows.
  4. INSERT the new pieces as brand-new paragraph rows (fresh uuids, every
     other column copied verbatim: source_page/outline/lang/extdata/...)
     plus queue rows so bge_m3 picks them up.

  python batch_operation/split_long_paragraphs.py --dry-run
  python batch_operation/split_long_paragraphs.py --apply --limit 2   # 2 batches
  python batch_operation/split_long_paragraphs.py --apply             # all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid

import common as C

log = C.setup_logging("split_long", "batch_operation/reports/split_long_paragraphs.log")

LIMIT = 2000
BATCH = 500
TARGETS = "split_targets_long"
PUNCT = set(".,;:!?)]}\"，。、；：！？…）】》」』”’")

COLS = ("id", "source_id", "source_page", "author", "pdate", "outline",
        "content", "pagenum", "lang", "keywords")

ENSURE_SQL = f"""
    DROP TABLE IF EXISTS {TARGETS};
    CREATE UNLOGGED TABLE {TARGETS} AS
    SELECT p.id FROM paragraph p
    JOIN file_metadata f ON f.id = p.source_id
    WHERE f.path LIKE 'OneDrive/%' AND length(p.content) > {LIMIT}
"""
COUNT_SQL = f"SELECT count(*) FROM {TARGETS}"
IDS_SQL = f"""
    SELECT id FROM {TARGETS}
    WHERE ($1::uuid IS NULL OR id > $1)
    ORDER BY id LIMIT {BATCH}
"""
FETCH_SQL = f"""
    SELECT {', '.join('p.' + c for c in COLS)}, p.extdata::text
    FROM paragraph p WHERE p.id = ANY($1)
"""
INS_PARA_SQL = """
    INSERT INTO paragraph (id, source_id, source_page, author, pdate, outline,
                           content, pagenum, lang, keywords, extdata)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
"""
INS_QUEUE_SQL = "INSERT INTO embedding_pending_queue (id, source_id) VALUES ($1, $2)"
DEL_PARA_SQL = "DELETE FROM paragraph WHERE id = ANY($1)"
DEL_EMB_SQL = "DELETE FROM text_embeddings WHERE id = ANY($1)"
DEL_QUEUE_SQL = "DELETE FROM embedding_pending_queue WHERE id = ANY($1)"


def split_long(text: str, limit: int = LIMIT) -> tuple[list[str], int]:
    """Greedy cut at the last punctuation inside each limit-sized window."""
    pieces: list[str] = []
    rest = text
    hard = 0
    while len(rest) > limit:
        window = rest[:limit]
        cut = -1
        for i, ch in enumerate(window):
            if ch in PUNCT:
                cut = i
        if cut < 0:
            cut = limit - 1          # no punctuation anywhere -> hard cut
            hard += 1
        pieces.append(rest[:cut + 1])
        rest = rest[cut + 1:]
    if rest:
        pieces.append(rest)
    return pieces, hard


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max batches (0 = all)")
    args = ap.parse_args()

    conn = await C.get_conn()
    stats = {"scanned": 0, "split": 0, "pieces": 0, "hard_cuts": 0,
             "emb_deleted": 0, "queue_deleted": 0, "queue_added": 0,
             "max_piece": 0, "batches": 0, "applied": bool(args.apply)}
    try:
        t0 = time.time()
        await conn.execute(ENSURE_SQL)
        total = await conn.fetchval(COUNT_SQL)
        log.info("candidates >%d chars under OneDrive: %s (%.0fs)",
                 LIMIT, f"{total:,}", time.time() - t0)
        if not total:
            log.info("nothing to split")
            return 0
        if not args.apply:
            # precise dry-run: page ids from the small target table, batch-fetch
            cursor = None
            while True:
                ids = [r["id"] for r in await conn.fetch(IDS_SQL, cursor)]
                if not ids:
                    break
                cursor = ids[-1]
                rows = await conn.fetch(FETCH_SQL, ids)
                for r in rows:
                    pieces, hard = split_long(r["content"])
                    stats["scanned"] += 1
                    stats["pieces"] += len(pieces)
                    stats["hard_cuts"] += hard
                    stats["max_piece"] = max(stats["max_piece"],
                                             max(len(p) for p in pieces))
            stats["split"] = stats["scanned"]
            log.info("dry-run: %s", json.dumps(stats))
            return 0

        cursor = None
        done_batches = 0
        sample_shown = False
        while True:
            ids = [r["id"] for r in await conn.fetch(IDS_SQL, cursor)]
            if not ids:
                break
            cursor = ids[-1]
            rows = await conn.fetch(FETCH_SQL, ids)
            if not rows:
                continue
            old_ids, new_ids = [], []
            para_args = []
            for r in rows:
                pieces, hard = split_long(r["content"])
                if len(pieces) == 1:
                    continue  # <= LIMIT already (target built a while ago)
                stats["scanned"] += 1
                stats["split"] += 1
                stats["pieces"] += len(pieces)
                stats["hard_cuts"] += hard
                stats["max_piece"] = max(stats["max_piece"],
                                         max(len(p) for p in pieces))
                old_ids.append(r["id"])
                if not sample_shown:
                    log.info("sample cut of %s (%d chars -> %d pieces): "
                             "p1 head=%r tail=%r | p2 head=%r",
                             r["id"], len(r["content"]), len(pieces),
                             pieces[0][:60], pieces[0][-60:], pieces[1][:60])
                    sample_shown = True
                for piece in pieces:
                    nid = uuid.uuid4()
                    new_ids.append(nid)
                    para_args.append((
                        nid, r["source_id"], r["source_page"], r["author"],
                        r["pdate"], r["outline"], piece, r["pagenum"],
                        r["lang"], r["keywords"], r["extdata"]))
            if old_ids:
                async with conn.transaction():
                    m = await conn.execute(DEL_EMB_SQL, old_ids)
                    stats["emb_deleted"] += int(m.split()[-1])
                    m = await conn.execute(DEL_QUEUE_SQL, old_ids)
                    stats["queue_deleted"] += int(m.split()[-1])
                    await conn.execute(DEL_PARA_SQL, old_ids)
                    await conn.executemany(INS_PARA_SQL, para_args)
                    # queue is filled by trigger trigger_paragraph_insert;
                    # read back the real count instead of inserting ourselves
                    n = await conn.fetchval(
                        "SELECT count(*) FROM embedding_pending_queue"
                        " WHERE id = ANY($1)", [a[0] for a in para_args])
                    stats["queue_added"] += n
                stats["batches"] += 1
                done_batches += 1
                log.info("batch %d: split=%d new=%d emb_del=%d | totals %s",
                         done_batches, stats["split"], stats["pieces"],
                         stats["emb_deleted"],
                         json.dumps(stats, default=str))
                C.dump_json("batch_operation/reports/split_long_paragraphs.json",
                            stats)
            if args.limit and done_batches >= args.limit:
                log.info("--limit %d reached", args.limit)
                break
        stats["finished"] = True
        C.dump_json("batch_operation/reports/split_long_paragraphs.json", stats)
        log.info("finished: %s", json.dumps(stats))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
