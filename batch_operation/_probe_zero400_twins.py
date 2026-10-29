"""400 空文件的孪生检查：同名非空文件是否存在于盘上别处/同目录。"""
import json
import os
import sys

sys.path.insert(0, "/app")
from jindai.maintenance.import_lists import get_conn  # noqa: E402


async def main() -> None:
    inv = json.load(open("/app/batch_operation/reports/disk_inventory.json",
                         encoding="utf-8"))
    disk = inv.get("sizes", inv)
    conn = await get_conn()
    rows = await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND coalesce(size_bytes,0)=0")
    await conn.close()
    empties = [r["path"] for r in rows
               if r["path"] in disk and disk[r["path"]] == 0]
    names = {os.path.basename(p).lower() for p in empties}
    twins = [(p, s) for p, s in disk.items()
             if s > 0 and os.path.basename(p).lower() in names]
    print(f"empty={len(empties)} same-name-nonempty-elsewhere={len(twins)}")
    for p, s in twins[:8]:
        print(f"  TWIN {p[:88]} -> {s}")
    # 同目录近似名非空版本（截取前20字符比对）
    dcount = 0
    examples = []
    for p in empties:
        d = os.path.dirname(p)
        base = os.path.basename(p).lower()[:20]
        for q, s in disk.items():
            if s > 0 and os.path.dirname(q) == d and \
                    os.path.basename(q).lower().startswith(base):
                dcount += 1
                if len(examples) < 5:
                    examples.append((p, q, s))
                break
    print(f"empties with a non-empty similar variant in SAME dir: {dcount}")
    for e in examples:
        print(f"  EMPTY {e[0][-70:]}\n    vs   {e[1][-70:]} ({e[2]} bytes)")

asyncio = __import__("asyncio")
asyncio.run(main())
