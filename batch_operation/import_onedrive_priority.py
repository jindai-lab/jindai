#!/usr/bin/env python3
"""Priority import: OneDrive PDFs that already have a text layer (no OCR).

Reads with_text list from detect_textlayer.py output, clones the stored
"导入 OneDrive PDF" task, overrides CalibreDataSource.only_files to point at
the list file, and executes the clone. Falls back gracefully.

Usage:
  python import_onedrive_priority.py --list batch_operation/reports/with_text.txt
  python import_onedrive_priority.py --list ... --chunk 2000 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time

sys.path.insert(0, "/app")
import common as C

log = C.setup_logging(
    "import_priority", "batch_operation/reports/import_priority.log"
)


def read_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]


async def write_list_file(paths: list[str], target: str) -> None:
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("\n".join(paths))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-json", default="batch_operation/reports/textlayer_split.json")
    ap.add_argument("--list", default="", help="text file with paths; overrides --split-json")
    ap.add_argument("--task", default="导入 OneDrive PDF")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-secs", type=int, default=120)
    args = ap.parse_args()

    if args.list:
        todo = read_list(args.list)
    else:
        with open(args.split_json) as fh:
            data = json.load(fh)
        todo = data.get("with_text", [])
    if not todo:
        log.info("nothing to import")
        return 0
    log.info("priority import: %d files with text layer", len(todo))

    from jindai import Task
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins
    from jindai.storage import storage

    dbo = await TaskDBO.get(args.task)
    if not dbo:
        log.error("task %r not found", args.task)
        return 2
    _init_plugins()

    # Drop files that already have paragraphs (idempotent resume)
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            """SELECT f.path FROM file_metadata f
               WHERE f.path LIKE 'OneDrive/%' AND lower(f.path) LIKE '%.pdf'
                 AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)"""
        )
    finally:
        await conn.close()
    missing = {r["path"] for r in rows}
    todo = [p for p in todo if p in missing]
    log.info("after resume filter: %d files still need import", len(todo))
    if not todo:
        return 0

    listfile_abs = "/storage/.batch/with_text_priority.txt"
    os.makedirs(os.path.dirname(listfile_abs), exist_ok=True)
    await write_list_file(todo, listfile_abs)
    log.info("list file written: %s (%d lines)", listfile_abs, len(todo))

    if args.dry_run:
        return 0

    # Clone task; override CalibreDataSource.only_files
    pipeline = copy.deepcopy(dbo.pipeline)
    overriden = False
    for stage in pipeline:
        if "CalibreDataSource" in stage:
            stage["CalibreDataSource"]["only_files"] = "storage_relative:.batch/with_text_priority.txt"
            overriden = True
    if not overriden:
        log.error("CalibreDataSource stage not found in task pipeline")
        return 2

    # only_files path must be readable inside container; storage root is /storage.
    # jindai paths in pipeline params are plain strings — use the /storage path.
    for stage in pipeline:
        if "CalibreDataSource" in stage:
            stage["CalibreDataSource"]["only_files"] = "/storage/.batch/with_text_priority.txt"

    from types import SimpleNamespace
    clone = SimpleNamespace(
        name=dbo.name + " (priority)",
        pipeline=pipeline,
        concurrent=dbo.concurrent,
        resume_next=getattr(dbo, "resume_next", False),
        params=getattr(dbo, "params", {}),
    )

    task = Task.from_dbo(clone, verbose=False, use_tqdm=False,
                         log=lambda *x: log.info(" ".join(str(i) for i in x)))
    t0 = time.time()
    await task.execute_async()
    log.info("priority import finished in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

