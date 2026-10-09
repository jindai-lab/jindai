"""把 temp_books_ingest 搬进来的 154 个文件登记进 file_metadata（幂等）。"""
import asyncio
import json
import sys

import common as C

log = C.setup_logging("register_moved", "batch_operation/reports/register_moved.log")
ROWS = sys.argv[1] if len(sys.argv) > 1 else "batch_operation/reports/temp_books_rows.json"

SQL = """
    INSERT INTO file_metadata (path, extension, size_bytes, item_type, title, authors)
    VALUES ($1, $2, $3, $4, $5, $6::text[])
    ON CONFLICT (path) DO NOTHING
"""


async def main() -> int:
    rows = json.load(open(ROWS, encoding="utf-8"))
    conn = await C.get_conn()
    ins = 0
    for path, ext, size, item_type, title, authors in rows:
        tag = await conn.execute(SQL, path, ext, size, item_type, title, authors)
        ins += int(tag.split()[-1])
    # 读回核对
    n = 0
    empty = 0
    for path, *_ in rows:
        r = await conn.fetchrow(
            "SELECT title, item_type FROM file_metadata WHERE path=$1", path)
        if r is None:
            n += 1
        elif not (r["title"] or "").strip():
            empty += 1
    log.info("[done] rows=%d inserted=%d missing_after=%d empty_title=%d",
             len(rows), ins, n, empty)
    await conn.close()
    return 0


sys.exit(asyncio.run(main()))
