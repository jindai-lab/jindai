#!/usr/bin/env python
"""classify_imported.py — 把 import_state='imported' 的行落实为 complete / partial。

背景：`imported` = 有段落但无盖章的 2026-10-16 机制上线前旧导入（2.2 万本）。
它们一直留在导入清单里（清单只排除 complete），每轮白走一遍。

本脚本对每本开 PDF 读 page_count，用与 PDFDataSource.reimport_gaps /
import_onedrive_pdfs._COVERED_PAGES_SQL **完全一致**的覆盖判据盖章：
  complete = 覆盖页数 >= 总页数  → 下轮清单直接出队
  partial  = 还有缺口            → 留在清单，导入器按页集补（不重 OCR）
  unknown  = PDF 打不开/0 页      → 不动，仍 imported，最后列出来

盖章走 import_onedrive_pdfs.record_import_state()（同批 upsert
file_import_state + 冗余 UPDATE file_metadata.import_state），
与导入器自己盖的章同一套写入，无口径漂移。

--apply 写前把原状态导 TSV：reports/imported_classify_backup_<ts>.tsv
回滚 = 对该 TSV 的 id 逐行 UPDATE import_state='imported' + DELETE file_import_state。

用法（容器内）：
  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/classify_imported.py [--limit N] [-j 8] [--apply]
"""
import argparse
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")

import logging  # noqa: E402
logging.disable(logging.CRITICAL)

import common as C  # noqa: E402
from jindai.maintenance.import_lists import (  # noqa: E402
    IMPORT_STATE_TABLE, count_covered_pages, ensure_import_state_table,
    record_import_state,
)

import pymupdf  # noqa: E402

REPORTS = "/app/batch_operation/reports"
SELECT_SQL = ("SELECT id, path FROM file_metadata "
              "WHERE import_state='imported' AND lower(path) LIKE '%.pdf' "
              "ORDER BY path")


def page_count(path: str):
    """线程池里跑：开 PDF 读页数。失败/0 页返回 None。

    `../app/` 陈旧前缀（44 行，DB path 与真实位置不符但文件在盘）先剥掉：
    真实文件就是去前缀后的 `/storage/OneDrive/...`。
    """
    try:
        rel = path[6:] if path.startswith("../app/") else path
        doc = pymupdf.open("/storage/" + rel)
        n = doc.page_count
        doc.close()
        return n or None
    except Exception:
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="写入盖章结果；不带则只算分布（dry-run）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 本（调试）")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="并发开 PDF 线程数")
    args = ap.parse_args()

    conn = await C.get_conn()
    await ensure_import_state_table(conn)
    rows = await conn.fetch(SELECT_SQL)
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    if not n:
        print("no 'imported' rows — nothing to do")
        return
    print(f"target rows: {n}  mode={'APPLY' if args.apply else 'dry-run'} "
          f"jobs={args.jobs}", flush=True)

    if args.apply:
        # 写前备份原状态（一次性导全量，非分块内的 rows——limit 下也导全量，
        # 回滚才能覆盖本轮真正改过的行）
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{REPORTS}/imported_classify_backup_{ts}.tsv"
        with open(bak, "w", encoding="utf-8") as f:
            for r in await conn.fetch(
                    "SELECT id, path, import_state FROM file_metadata "
                    "WHERE import_state='imported' AND lower(path) LIKE '%.pdf' "
                    "ORDER BY path"):
                f.write(f"{r['id']}\t{r['path']}\t{r['import_state']}\n")
        print("backup:", bak, flush=True)

    counts = {"complete": 0, "partial": 0, "unknown": 0}
    unknown_samples = []
    loop = asyncio.get_running_loop()
    ex = ThreadPoolExecutor(max_workers=args.jobs)
    chunk = 400
    t0 = time.time()
    for i in range(0, n, chunk):
        part = rows[i:i + chunk]
        # 开 PDF 全部并行（瓶颈在 CIFS 读）
        pcs = await loop.run_in_executor(
            ex, lambda ps=[r["path"] for r in part]: [page_count(p) for p in ps])
        if args.apply:
            for r, pc in zip(part, pcs):
                if not pc:
                    counts["unknown"] += 1
                    if len(unknown_samples) < 20:
                        unknown_samples.append(r["path"])
                    continue
                covered = await count_covered_pages(conn, r["id"])
                st = await record_import_state(conn, r["id"], pc, covered)
                counts[st] = counts.get(st, 0) + 1
        else:
            for r, pc in zip(part, pcs):
                if not pc:
                    counts["unknown"] += 1
                    if len(unknown_samples) < 20:
                        unknown_samples.append(r["path"])
                    continue
                covered = await count_covered_pages(conn, r["id"])
                st = "complete" if covered >= pc else "partial"
                counts[st] += 1
        done = min(i + chunk, n)
        el = time.time() - t0
        eta = (el / done) * (n - done) if done else 0
        print(f"[progress] {done}/{n}  complete={counts['complete']} "
              f"partial={counts['partial']} unknown={counts['unknown']}  "
              f"elapsed={el/60:.1f}min eta={eta/60:.1f}min", flush=True)

    print(f"DONE mode={'APPLY' if args.apply else 'dry-run'} "
          f"total={n} complete={counts['complete']} partial={counts['partial']} "
          f"unknown={counts['unknown']} "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    if counts["unknown"]:
        print("unknown 样本（PDF 打不开/0 页，保持 imported）:")
        for p in unknown_samples:
            print("  " + p)
    # 读回分布
    dist = await conn.fetch(
        "SELECT import_state, count(*) c FROM file_metadata "
        "WHERE import_state IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")
    print("STATES:", " | ".join(f"{r['import_state']}={r['c']}" for r in dist),
          flush=True)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
