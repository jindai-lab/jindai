"""size_bytes=0 行全量核验：盘上是否真实存在、真实大小、分布与成因分桶。"""
import json
import sys
from collections import Counter

sys.path.insert(0, "/app")
from jindai.maintenance.import_lists import get_conn  # noqa: E402


async def main() -> None:
    inv = json.load(open("/app/batch_operation/reports/disk_inventory.json",
                         encoding="utf-8"))
    disk = inv.get("sizes", inv)
    conn = await get_conn()
    rows = await conn.fetch(
        "SELECT path, size_bytes, created_at, split_part(path,'/',1) top "
        "FROM file_metadata WHERE lower(path) LIKE '%.pdf'")
    await conn.close()

    zero = [r for r in rows if not r["size_bytes"]]
    pos = [r for r in rows if r["size_bytes"]]
    z_on = [r for r in zero if r["path"] in disk and disk[r["path"]] > 0]
    z_missing = [r for r in zero if r["path"] not in disk]
    z_real0 = [r for r in zero if r["path"] in disk and disk[r["path"]] == 0]
    print(f"pdf_rows={len(rows)} size0={len(zero)} size>0={len(pos)}")
    print(f"size0 -> on_disk_real_size>0: {len(z_on)} | really 0 on disk: "
          f"{len(z_real0)} | NOT on disk: {len(z_missing)}")
    print("--- size0 but real file samples:")
    for r in z_on[:5]:
        print(f"  {r['path'][:80]} | real={disk[r['path']]}")
    print("--- size0 NOT on disk samples:")
    for r in z_missing[:5]:
        print(f"  {r['path'][:80]} | created={r['created_at']:%Y-%m-%d}")
    print("--- size0 by top:", dict(Counter(r["top"] for r in zero).most_common(8)))
    print("--- size0 by created month:",
          dict(Counter(f"{r['created_at']:%Y-%m}" for r in zero).most_common(12)))
    print("--- size>0 by created month:",
          dict(Counter(f"{r['created_at']:%Y-%m}" for r in pos).most_common(12)))

asyncio_run = __import__("asyncio").run
asyncio_run(main())
