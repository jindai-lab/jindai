"""全库 PDF 磁盘清单：磁盘 *.pdf vs file_metadata 登记情况 → JSON 缓存。

目的（2026-10-09 用户指令）：每小时报告的“剩余 PDF”不再只看已登记行——
未登记的磁盘 PDF（Journals/Series/Zotero/Albums/Supplementary Books…）也要可见。

输出 ~/.hermes/profiles/collector/cache/pdf_disk_inventory.json：
  ts / disk_total / db_pdf_rows / untracked_total / untracked_by_dir /
  rows_missing_file / rows_missing_by_dir

dry-run 默认打印摘要；--write 落缓存。全树走一遍 CIFS 约 10–18 分钟，
由 gpu_cpu_monitor.py 在缓存过期(>24h)时后台拉起，勿在整点任务里同步等它。
  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/pdf_disk_inventory.py --write
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import time

import common as C

CACHE = "batch_operation/reports/pdf_disk_inventory.json"   # bind-mounted，宿主 jindai-lab/.../reports/ 下同名文件


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from jindai.storage import storage
    base = storage.safe_join("OneDrive")[: -len("OneDrive")]

    t0 = time.time()
    disk: set[str] = set()
    for dirpath, _d, fns in os.walk(base):
        for fn in fns:
            if fn.lower().endswith(".pdf"):
                disk.add(os.path.relpath(os.path.join(dirpath, fn), base)
                         .replace(os.sep, "/"))
    log_walk = f"[walk] {len(disk)} pdfs in {time.time()-t0:.0f}s"

    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf'")
    db = {r["path"] for r in rows}
    await conn.close()

    untracked = [p for p in disk if p not in db]
    missing = [p for p in db if p not in disk]
    inv = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "disk_total": len(disk),
        "db_pdf_rows": len(db),
        "untracked_total": len(untracked),
        "untracked_by_dir": dict(collections.Counter(
            p.split("/")[0] for p in untracked).most_common()),
        "rows_missing_file": len(missing),
        "rows_missing_by_dir": dict(collections.Counter(
            p.split("/")[0] for p in missing).most_common(10)),
    }
    print(log_walk)
    print(json.dumps(inv, ensure_ascii=False, indent=1))
    if args.write:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        tmp = CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(inv, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CACHE)          # 原子替换，监控读不到半截文件
        print("written:", CACHE)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
