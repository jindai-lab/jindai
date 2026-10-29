"""400 条「盘上存在但 stat=0」行的画像：分布、样例、文件系统形态。"""
import json
import subprocess
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
        "SELECT path, size_bytes, created_at::text ct, import_state "
        "FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND coalesce(size_bytes,0)=0")
    await conn.close()
    zero_on_disk = [r for r in rows
                    if r["path"] in disk and disk[r["path"]] == 0]
    print(f"size0 rows total={len(rows)}, really 0 on disk={len(zero_on_disk)}")
    print("by top:", dict(Counter(r["path"].split("/")[0]
                                  for r in zero_on_disk).most_common(8)))
    print("by created date:", dict(Counter(r["ct"][:10] for r in zero_on_disk)
                                   .most_common(10)))
    print("by import_state:", dict(Counter(r["import_state"] or "NULL"
                                           for r in zero_on_disk)))
    print("--- samples (path | created):")
    for r in zero_on_disk[:12]:
        print(f"  {r['path'][:95]} | {r['ct'][:10]}")

    # 直接看文件系统形态（前6个）：stat + 文件类型 + 是否符号链接/reparse
    print("--- filesystem probe:")
    for r in zero_on_disk[:6]:
        p = "/storage/" + r["path"]
        out = subprocess.run(["stat", "-c", "%F | %s bytes | links=%h", p],
                             capture_output=True, text=True).stdout.strip()
        print(f"  {r['path'][:70]}\n     -> {out}")

asyncio = __import__("asyncio")
asyncio.run(main())
