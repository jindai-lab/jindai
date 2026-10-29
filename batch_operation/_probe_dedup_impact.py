"""查重护栏上线前的误伤率探针：现盘未登记 PDF 中有多少会因同名跳过、样例判读。"""
import asyncio
import json
import sys
import unicodedata

sys.path.insert(0, "/app")
from jindai.maintenance.import_lists import get_conn  # noqa: E402


def key(p: str) -> str:
    return unicodedata.normalize("NFC", p.split("/")[-1]).lower()


async def main() -> None:
    inv = json.load(open("/app/batch_operation/reports/disk_inventory.json",
                         encoding="utf-8"))
    disk = inv.get("sizes", inv)
    conn = await get_conn()
    rows = await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND path NOT LIKE 'Tool Books/%' AND path NOT LIKE '.trash/%'")
    existing = {r["path"] for r in rows}
    await conn.close()
    seen = {}
    for p in existing:
        seen.setdefault(key(p), p)
    cands = [p for p in disk if p not in existing]
    hits = [(p, seen[key(p)]) for p in cands if key(p) in seen]
    print(f"disk={len(disk)} existing={len(existing)} "
          f"candidates={len(cands)} would_skip={len(hits)}")
    for p, t in hits[:15]:
        print(f"  SKIP {p[:80]}  <=>  {t[:80]}")
    for p in cands:
        if key(p) not in seen:
            print(f"  KEEP {p[:90]}")

asyncio.run(main())
