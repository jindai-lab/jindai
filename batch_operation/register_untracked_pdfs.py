"""把磁盘上从未登记的 PDF 全部 upsert 进 file_metadata（2026-10-09 用户指令：全部补，除 Tool Books）。

来源：`reports/pdf_disk_paths.txt`（≤6h 的全盘清单缓存；过期则现场 os.walk 全树 10–18 分钟）。
item_type 按顶层目录：Journals/Series → `proceedings`（合订本）、Zotero → `journalArticle`、
其余（Authors/…/`..`）→ `book`、Albums → `multimedia`（2026-10-09 用户定）。顺带写 size_bytes（逐文件 stat）。
幂等：`ON CONFLICT (path) DO NOTHING`；dry-run 默认。

  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/register_untracked_pdfs.py [--apply] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time

import common as C

log = C.setup_logging("register_untracked",
                      "batch_operation/reports/register_untracked.log")
OUT = "batch_operation/reports/register_untracked.json"

ITEM_TYPE = {"Journals": "proceedings", "Series": "proceedings",
             "Zotero": "journalArticle", "Albums": "multimedia"}
BATCH = 2000


def disk_list(base: str) -> list[str]:
    cache = "batch_operation/reports/pdf_disk_paths.txt"
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 3600:
        with open(cache, encoding="utf-8") as f:
            out = [l.strip() for l in f if l.strip().lower().endswith(".pdf")]
        log.info("[disk] cache hit: %d paths", len(out))
        return out
    out = []
    for dirpath, dns, fns in os.walk(base):
        # 隐藏目录（含回收站 .trash/）不登记：那里是被删除记录的文件，
        # 每周清空一次，绝不能重新入队。
        dns[:] = [d for d in dns if not d.startswith(".")]
        for fn in fns:
            if fn.lower().endswith(".pdf"):
                out.append(os.path.relpath(os.path.join(dirpath, fn), base)
                           .replace(os.sep, "/"))
    log.info("[disk] walked %d paths", len(out))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from jindai.storage import storage
    base = storage.safe_join("OneDrive")[: -len("OneDrive")]

    paths = disk_list(base)
    conn = await C.get_conn()
    existing = {r["path"] for r in await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf'")}
    new = [p for p in paths
           if p not in existing and not p.startswith("Tool Books/")
           and not p.startswith(".trash/")]
    skipped_tool = len([p for p in paths
                        if p.startswith("Tool Books/") and p not in existing])
    skipped_trash = len([p for p in paths
                         if p.startswith(".trash/") and p not in existing])
    if args.limit:
        new = new[:args.limit]
    by_dir = collections.Counter(p.split("/")[0] for p in new)
    log.info("[plan] %d to insert (tool_books excluded); by dir: %s", len(new),
             json.dumps(dict(by_dir), ensure_ascii=False))

    applied = {"inserted": 0, "failed": 0, "tool_books_excluded": skipped_tool,
               "trash_excluded": skipped_trash}
    if args.apply:
        rows = []
        stat_fail = 0
        for p in new:
            try:
                sz = os.path.getsize(os.path.join(base, p))
            except OSError:
                sz, stat_fail = 0, 1
            rows.append((p, "pdf", sz, ITEM_TYPE.get(p.split("/")[0], "book")))
        applied["stat_failed"] = stat_fail
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            try:
                res = await conn.executemany(
                    "INSERT INTO file_metadata (path, extension, size_bytes, item_type, import_state) "
                    "VALUES ($1, $2, $3, $4, 'pending') ON CONFLICT (path) DO NOTHING", chunk)
                applied["inserted"] += int(res.split()[-1]) if res else 0
            except Exception as ex:
                applied["failed"] += 1
                log.warning("[ins] batch %d failed: %s", i, str(ex)[:200])
            if (i // BATCH) % 5 == 0:
                log.info("[ins] %d/%d done", i + len(chunk), len(rows))
    after = await conn.fetchval(
        "SELECT count(*) FROM file_metadata WHERE lower(path) LIKE '%.pdf'")
    summary = {"mode": "apply" if args.apply else "dry-run",
               "disk": len(paths), "already": len(existing),
               "plan_by_dir": dict(by_dir), "applied": applied,
               "pdf_rows_after": int(after)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log.info("[done] %s", json.dumps(summary, ensure_ascii=False))
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
