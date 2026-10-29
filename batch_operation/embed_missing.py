"""Embed every paragraph that is still missing an embedding.

Scope (default ``all``): every paragraph whose id has no row in
text_embeddings -- today 35.27M paragraphs (11.0M under OneDrive/),
including paragraphs imported after this file snapshot (re-run to catch up).

Design:
  * discovers work with the authoritative anti-join
    (paragraph LEFT-ANTI text_embeddings) grouped per file, then walks
    files largest-first;
  * per file, pages paragraphs by keyset (id) so a 2M-paragraph book
    never blows memory; chunks with the backend's own get_chunks (200/50);
  * encodes pages in parallel batches against the configured backend
    (embedding_api_base: local -> in-process BAAI/bge-m3 on the GPU), inserts
    ON CONFLICT DO NOTHING, verifies the
    per-paragraph chunk count, then consumes embedding_pending_queue rows
    (same semantics as jindai/maintenance/manager.py::consume_embedding_queue),
    so the UI queue gauge drains truthfully and re-runs are resumable.

Run (in container, cwd /app):
  python batch_operation/embed_missing.py --dry-run
  python batch_operation/embed_missing.py --scope onedrive --limit-files 5
  python batch_operation/embed_missing.py --scope all --inflight 4 --batch 256
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import common as C
from jindai.embedutils import EmbeddingClient

log = C.setup_logging("embed_missing", "batch_operation/reports/embed_missing.log")

INSERT_SQL = C.INSERT_EMBEDDING_SQL
MISSING_BY_FILE_SQL = """
    SELECT p.source_id, count(*)::bigint AS n
    FROM paragraph p
    WHERE NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
    GROUP BY 1
"""
PAGE_SQL = """
    SELECT p.id, p.content
    FROM paragraph p
    WHERE p.source_id = $1
      AND ($2::uuid IS NULL OR p.id > $2)
      AND NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
    ORDER BY p.id
    LIMIT $3
"""
COUNT_CHUNKS_SQL = """
    SELECT id, count(*) AS n FROM text_embeddings
    WHERE id = ANY($1::uuid[]) GROUP BY id
"""
DELETE_QUEUE_SQL = """
    DELETE FROM embedding_pending_queue WHERE id = ANY($1::uuid[])
"""
ENSURE_QUEUE_SQL = """
    INSERT INTO embedding_pending_queue (id, source_id, created_at)
    SELECT id, source_id, now() FROM paragraph WHERE source_id = $1
    ON CONFLICT DO NOTHING
"""
FINAL_LEFT_SQL = """
    SELECT count(*) FROM paragraph p
    WHERE NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
      AND ($1::text IS NULL OR EXISTS (
          SELECT 1 FROM file_metadata f
          WHERE f.id = p.source_id AND f.path LIKE $2))
"""

PAGE_SIZE = 5000       # paragraphs fetched per keyset round-trip
WAVE = 2048            # chunks held in memory per encode/insert wave
LOG_PARAS = 200_000    # per-file heartbeat interval (paragraphs)
CHUNKS_PER_PARA_EST = 1.9


async def collect_missing(conn, scope: str, limit_files: int) -> tuple:
    """Return ([(file_id, path, missing_paragraphs)] desc, total_missing)."""
    rows = await conn.fetch(MISSING_BY_FILE_SQL)
    missing = {r["source_id"]: r["n"] for r in rows}
    total = sum(missing.values())
    log.info(
        "paragraphs missing embeddings: %s across %d files",
        f"{total:,}", len(missing),
    )
    if scope == "onedrive":
        path_rows = await conn.fetch(
            "SELECT id, path FROM file_metadata WHERE path LIKE $1", C.ONEDRIVE_LIKE
        )
    else:
        path_rows = await conn.fetch("SELECT id, path FROM file_metadata")
    paths = {r["id"]: r["path"] for r in path_rows}
    files = [(fid, paths[fid], n) for fid, n in missing.items() if fid in paths]
    orphans = total - sum(n for _, _, n in files)
    if orphans:
        log.warning(
            "%s missing paragraphs belong to file ids absent from file_metadata "
            "(deleted files?) and are out of scope", f"{orphans:,}",
        )
    files.sort(key=lambda x: -x[2])
    if limit_files:
        files = files[:limit_files]
    return files, total


async def embed_wave(conn, sem, embedder: EmbeddingClient, bulk: list,
                     args) -> None:
    """Encode + insert one wave of (pid, sid, chunk_id, text) tuples."""
    if not bulk:
        return
    batches = [bulk[i:i + args.batch] for i in range(0, len(bulk), args.batch)]

    async def emb(texts):
        async with sem:
            return await embedder.encode(texts)

    vec_lists = await asyncio.gather(*[emb([b[3] for b in bt]) for bt in batches])
    recs = [
        (b[0], b[1], b[2], json.dumps(vec))
        for bt, vecs in zip(batches, vec_lists)
        for b, vec in zip(bt, vecs)
    ]
    if recs:
        await conn.executemany(INSERT_SQL, recs)


async def process_file(conn, sem, embedder: EmbeddingClient, fid, path,
                       args, prog: C.Progress) -> tuple:
    """Embed all missing paragraphs of one file (keyset-paged).

    Returns (paragraphs, chunks, incomplete_paragraphs).
    """
    if not args.no_queue:
        await conn.execute(ENSURE_QUEUE_SQL, fid)

    n_para = n_chunk = n_incomplete = 0
    last_id = None
    heartbeat = 0
    while True:
        rows = await conn.fetch(PAGE_SQL, fid, last_id, PAGE_SIZE)
        if not rows:
            break
        last_id = rows[-1]["id"]

        bulk: list[tuple] = []
        expected: dict = {}   # paragraph uuid -> expected chunk count
        for r in rows:
            chunks = C.get_chunks(r["content"])
            expected[r["id"]] = len(chunks)
            for ci, chunk in enumerate(chunks, 1):
                bulk.append((str(r["id"]), str(fid), ci, chunk))

        for ws in range(0, len(bulk), WAVE):
            await embed_wave(conn, sem, embedder, bulk[ws:ws + WAVE], args)

        # verify per-paragraph coverage, then consume queue rows for
        # paragraphs that are actually complete
        pids = list(expected)
        counts = {r["id"]: r["n"] for r in await conn.fetch(COUNT_CHUNKS_SQL, pids)}
        complete = [pid for pid in pids if counts.get(pid, 0) >= expected[pid]]
        incomplete = [pid for pid in pids if counts.get(pid, 0) < expected[pid]]
        if complete:
            await conn.execute(DELETE_QUEUE_SQL, complete)

        n_para += len(rows)
        n_chunk += len(bulk)
        n_incomplete += len(incomplete)
        prog.add(len(bulk))
        heartbeat += len(rows)
        if incomplete:
            log.warning(
                "%s: %d/%d paragraphs incomplete after page (kept in queue)",
                path, len(incomplete), len(pids),
            )
        if heartbeat >= LOG_PARAS:
            heartbeat = 0
            log.info("  ... %s paragraphs done | %s", f"{n_para:,}", prog.line())
    if n_incomplete:
        log.warning("%s: total incomplete paragraphs %d", path, n_incomplete)
    return n_para, n_chunk, n_incomplete


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", choices=["all", "onedrive"], default="all")
    ap.add_argument("--batch", type=int, default=256, help="texts per API request")
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-queue", action="store_true",
                    help="do not touch embedding_pending_queue")
    ap.add_argument("--report", default="batch_operation/reports/embed_missing.json")
    args = ap.parse_args()

    t0 = time.time()
    conn = await C.get_conn()
    try:
        files, total_missing = await collect_missing(conn, args.scope, args.limit_files)
        log.info(
            "scope=%s files=%d missing_paragraphs=%s (~%s chunks @%.1f/para)",
            args.scope, len(files), f"{total_missing:,}",
            f"{int(total_missing * CHUNKS_PER_PARA_EST):,}", CHUNKS_PER_PARA_EST,
        )
        if args.dry_run:
            for _fid, path, n in files[:20]:
                log.info("  %8d  %s", n, path)
            return 0

        embedder = EmbeddingClient(timeout=300.0)
        sem = asyncio.Semaphore(args.inflight)
        prog = C.Progress(log, unit="chunks")
        prog.set_total(int(total_missing * CHUNKS_PER_PARA_EST))
        done_files = para_done = chunk_done = incomplete_done = 0
        failed_files = []
        try:
            for fid, path, _est in files:
                try:
                    n_para, n_chunk, n_inc = await process_file(
                        conn, sem, embedder, fid, path, args, prog
                    )
                except Exception as ex:  # noqa: BLE001 - keep going per file
                    log.error("file failed: %s (%s: %s)", path, type(ex).__name__, ex)
                    failed_files.append(path)
                    continue
                done_files += 1
                para_done += n_para
                chunk_done += n_chunk
                incomplete_done += n_inc
                if done_files % 20 == 0 or done_files == len(files):
                    log.info(
                        "files %d/%d | paragraphs %s | chunks %s | %s",
                        done_files, len(files), f"{para_done:,}",
                        f"{chunk_done:,}", prog.line(),
                    )
        finally:
            await embedder.aclose()

        left = await conn.fetchval(
            FINAL_LEFT_SQL,
            None if args.scope == "all" else "x",
            C.ONEDRIVE_LIKE,
        )
        summary = {
            "scope": args.scope,
            "files_total": len(files),
            "files_done": done_files,
            "files_failed": len(failed_files),
            "failed_sample": failed_files[:20],
            "paragraphs_processed": para_done,
            "chunks_written": chunk_done,
            "incomplete_paragraphs": incomplete_done,
            "paragraphs_still_missing": left,
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }
        C.dump_json(args.report, summary)
        log.info("summary: %s", json.dumps(summary, ensure_ascii=False))
        return 0 if left == 0 and not failed_files else 3
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
