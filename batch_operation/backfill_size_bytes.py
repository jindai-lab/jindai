"""回填 file_metadata.size_bytes（size=0 且盘上有真实大小的行，来自 fresh inventory）。"""
import json
import sys

sys.path.insert(0, "/app")
from jindai.maintenance.import_lists import get_conn  # noqa: E402


async def main() -> int:
    inv = json.load(open("/app/batch_operation/reports/disk_inventory.json",
                         encoding="utf-8"))
    disk = inv.get("sizes", inv)
    conn = await get_conn()
    rows = await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND coalesce(size_bytes,0)=0")
    todo = [(disk[r["path"]], r["path"]) for r in rows
            if r["path"] in disk and disk[r["path"]] > 0]
    print(f"to backfill: {len(todo)}")
    n = 0
    for i in range(0, len(todo), 5000):
        chunk = todo[i:i + 5000]
        async with conn.transaction():
            await conn.executemany(
                "UPDATE file_metadata SET size_bytes=$1 "
                "WHERE (size_bytes IS NULL OR size_bytes=0) AND path=$2",
                [(sz, p) for sz, p in chunk])
            n += len(chunk)
    left = await conn.fetchval(
        "SELECT count(*) FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND coalesce(size_bytes,0)=0")
    await conn.close()
    print(f"backfilled={n} size0_left={left}")
    return 0


if __name__ == "__main__":
    sys.exit(__import__("asyncio").run(main()))
