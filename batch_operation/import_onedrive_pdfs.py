"""Import OneDrive PDFs that have no paragraph yet, via the stored task.

Executes the existing ``导入 OneDrive PDF`` task (task_dbo) with the app's
own Task machinery: CalibreDataSource enumerates OneDrive/*.pdf from the
Calibre library, PDFDataSource resumes per file (skip_existed), then the
LanguageDetect / Reparagraph / WordCut / SaveParagraph chain runs as usual.
Every inserted paragraph auto-enqueues into embedding_pending_queue via the
DB trigger, so embed_missing.py can pick the work up afterwards.

Safe to re-run (the pipeline resumes per file, fully imported files are
skipped).

  python batch_operation/import_onedrive_pdfs.py --dry-run
  python batch_operation/import_onedrive_pdfs.py --task "导入 OneDrive PDF" -n 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import common as C

log = C.setup_logging(
    "import_onedrive", "batch_operation/reports/import_onedrive.log"
)

MISSING_PDF_SQL = """
    SELECT count(*) FROM file_metadata f
    WHERE f.path LIKE $1 AND lower(f.path) LIKE '%.pdf'
      AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
"""
PARA_TOTAL_SQL = "SELECT count(*)::bigint FROM paragraph"


async def stats(conn) -> dict:
    missing = await conn.fetchval(MISSING_PDF_SQL, C.ONEDRIVE_LIKE)
    paragraphs = await conn.fetchval(PARA_TOTAL_SQL)
    return {"onedrive_pdfs_without_paragraph": missing, "paragraph_total": paragraphs}


async def progress_watcher(interval: int):
    conn = await C.get_conn()
    t0 = time.time()
    try:
        while True:
            await asyncio.sleep(interval)
            row = await conn.fetchrow(
                """
                SELECT (SELECT count(*)::bigint FROM file_metadata f
                          WHERE f.path LIKE $1
                            AND NOT EXISTS (SELECT 1 FROM paragraph p
                                            WHERE p.source_id = f.id)) AS files_left,
                       (SELECT count(*)::bigint FROM paragraph) AS paras
                """,
                C.ONEDRIVE_LIKE,
            )
            log.info(
                "[progress] files w/o paragraph=%s  paragraphs=%s  elapsed=%.1fmin",
                row["files_left"], f"{row['paras']:,}",
                (time.time() - t0) / 60,
            )
    finally:
        await conn.close()


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="导入 OneDrive PDF")
    ap.add_argument("-n", "--concurrent", type=int, default=0,
                    help="override worker concurrency (0 = task default)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress-secs", type=int, default=120)
    ap.add_argument("--report", default="batch_operation/reports/import_onedrive.json")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        pre = await stats(conn)
    finally:
        await conn.close()
    log.info("pre: files without paragraph=%s, paragraph total=%s",
             pre["onedrive_pdfs_without_paragraph"], f"{pre['paragraph_total']:,}")
    if args.dry_run:
        return 0

    # --- load task & plugins exactly like `python -m jindai task ...` -----
    from jindai import Task
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins

    dbo = await TaskDBO.get(args.task)
    if not dbo:
        log.error("task %r not found", args.task)
        return 2
    _init_plugins()

    logfile = open(args.report + ".log", "a", encoding="utf-8")

    def logfn(*x):
        msg = " ".join(str(i) for i in x)
        print(msg, flush=True)
        print(msg, file=logfile, flush=True)

    task = Task.from_dbo(dbo, verbose=args.verbose, log=logfn, use_tqdm=False)
    if args.concurrent > 0:
        task.concurrent = args.concurrent

    watcher = asyncio.create_task(progress_watcher(args.progress_secs))
    t0 = time.time()
    try:
        result = await task.execute_async()
    finally:
        watcher.cancel()
        logfile.close()

    conn = await C.get_conn()
    try:
        post = await stats(conn)
    finally:
        await conn.close()

    summary = {
        "task": args.task,
        "pre": pre,
        "post": post,
        "imported_files": (
            pre["onedrive_pdfs_without_paragraph"] - post["onedrive_pdfs_without_paragraph"]
        ),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "result": result,
    }
    C.dump_json(args.report, summary)
    log.info("summary: %s", json.dumps(summary, ensure_ascii=False, default=str))
    if isinstance(result, dict) and result.get("__exception__"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
