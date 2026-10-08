"""Import OneDrive PDFs that have no paragraph yet — file_metadata 驱动（2026-10-08 起）。

原 Calibre 路由（CalibreDataSource 枚举 metadata.db → PDFDataSource）已废止：
Calibre 停用后本脚本改为**只依赖 jindai 的 file_metadata**——

1. （可选，默认开）磁盘扫一遍 OneDrive 下的 *.pdf，把表里还没有的文件 upsert 进
   file_metadata（path 唯一；新文件此后唯一的登记入口，替代 Calibre 枚举）。
2. SELECT file_metadata 中**所有 PDF 行**（2026-10-08 用户指令：解除 OneDrive/白名单限制、导入全部 file_metadata PDF，仍排除 `Tool Books/` 与 `../` 遗留行；原为 `path LIKE 'OneDrive/%' AND lower(path) LIKE '%.pdf'`）
   的全部路径作为导入清单（已导入的由 skip_existed + reimport_gaps 按页集跳过，
   与旧路由“每轮枚举全量、按页集续传”语义一致）。
3. 按既定“工具书不导入”规则用关键词正则剔除词典类（单独计数、单独落盘），
   再剔除磁盘上已不存在的陈旧路径（交给日后的 fix_path_drift 处理）。
4. 克隆存好的任务管线、**去掉 CalibreDataSource（若仍在）**，把清单注入
   PDFDataSource.content，其余 LanguageDetect/Reparagraph/DatasetFromField/
   SaveParagraph 链不变；dataset_name 书库 由 PDFDataSource 自己挂。

统计口径与监控一致（jindai_progress.sh 的 pending 就是 file_metadata 反连接）。

  python batch_operation/import_onedrive_pdfs.py --dry-run          # 只报清单统计
  python batch_operation/import_onedrive_pdfs.py --no-scan          # 跳过磁盘 upsert
  python batch_operation/import_onedrive_pdfs.py -n 4 --progress-secs 60
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import common as C

log = C.setup_logging(
    "import_onedrive", "batch_operation/reports/import_onedrive.log"
)

MISSING_PDF_SQL = """
    SELECT count(*) FROM file_metadata f
    WHERE lower(f.path) LIKE '%.pdf'
      AND f.path NOT LIKE 'Tool Books/%'
      AND f.path NOT LIKE '../%'
      AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
"""
PARA_TOTAL_SQL = "SELECT count(*)::bigint FROM paragraph"
LIST_SQL = """
    SELECT path FROM file_metadata
    WHERE lower(path) LIKE '%.pdf'
      AND path NOT LIKE 'Tool Books/%'
      AND path NOT LIKE '../%'
    ORDER BY path
"""
EXISTING_ONEDRIVE_SQL = "SELECT path FROM file_metadata WHERE path LIKE $1"

# 工具书（词典/辞书/大部头参考书）——沿用 references/backlog-inventory.md 的关键词表，
# 站规：不导入（不排序到队尾，直接不进批次）。
TOOLBOOK_RE = re.compile(
    r"词典|辭典|字典|大典|辞海|辞源|百科|名物|lexicon|lexica|dictionary|dictionar|"
    r"dictionarium|glossar|vocabular|lexikon|lexique|thesaurus|concordance|"
    r"wörterbuch|woerterbuch|liddell|scott|georges|bekker", re.I)


async def stats(conn) -> dict:
    missing = await conn.fetchval(MISSING_PDF_SQL)
    paragraphs = await conn.fetchval(PARA_TOTAL_SQL)
    return {"onedrive_pdfs_without_paragraph": missing, "paragraph_total": paragraphs}


async def progress_watcher(interval: int):
    conn = await C.get_conn()
    t0 = time.time()
    try:
        while True:
            await asyncio.sleep(interval)
            row = await conn.fetchrow(
                """
                SELECT (SELECT count(*)::bigint FROM file_metadata f
                          WHERE lower(f.path) LIKE '%.pdf'
                            AND f.path NOT LIKE 'Tool Books/%'
                            AND f.path NOT LIKE '../%'
                            AND NOT EXISTS (SELECT 1 FROM paragraph p
                                            WHERE p.source_id = f.id)) AS files_left,
                       (SELECT count(*)::bigint FROM paragraph) AS paras
                """
            )
            log.info(
                "[progress] files w/o paragraph=%s  paragraphs=%s  elapsed=%.1fmin",
                row["files_left"], f"{row['paras']:,}",
                (time.time() - t0) / 60,
            )
    finally:
        await conn.close()


async def scan_disk_upsert(conn) -> int:
    """把磁盘上存在、file_metadata 还没有的 OneDrive PDF 补进表里（幂等）。

    Calibre 停用后没有别的东西会登记新文件——这一步就是新文件进入导入队列的入口。
    """
    from jindai.storage import storage
    root = storage.safe_join("OneDrive")
    disk: dict[str, int] = {}
    n_dirs = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        n_dirs += 1
        for fn in filenames:
            if not fn.lower().endswith(".pdf"):
                continue
            full = os.path.join(dirpath, fn)
            rel = "OneDrive/" + os.path.relpath(full, root).replace(os.sep, "/")
            try:
                disk[rel] = os.path.getsize(full)
            except OSError:
                disk[rel] = 0
        if n_dirs % 2000 == 0:
            log.info("[scan] walked %d dirs, %d pdfs so far", n_dirs, len(disk))

    existing = {r["path"] for r in await conn.fetch(EXISTING_ONEDRIVE_SQL, C.ONEDRIVE_LIKE)}
    new = [(p, disk[p]) for p in disk if p not in existing]
    if new:
        await conn.executemany(
            "INSERT INTO file_metadata (path, extension, size_bytes, item_type) "
            "VALUES ($1, 'pdf', $2, 'book') ON CONFLICT (path) DO NOTHING",
            new,
        )
    # 表里有、磁盘没有的（陈旧路径）单独留档，不在此删行
    stale = sorted(existing - set(disk))
    if stale:
        path = "batch_operation/reports/filemeta_stale_paths.txt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(stale) + "\n")
    log.info(
        "[scan] disk pdfs=%d  existing rows=%d  upserted=%d  stale-on-disk=%d (see reports/filemeta_stale_paths.txt)",
        len(disk), len(existing), len(new), len(stale),
    )
    return len(new)


async def build_file_list(conn) -> tuple[list[str], dict]:
    rows = await conn.fetch(LIST_SQL)
    all_paths = [r["path"] for r in rows]
    toolbooks = [p for p in all_paths if TOOLBOOK_RE.search(p)]
    candidates = [p for p in all_paths if p not in set(toolbooks)]

    from jindai.storage import storage
    missing = [p for p in candidates if not os.path.exists(storage.safe_join(p))]
    keep = [p for p in candidates if p not in set(missing)]

    meta = {
        "filemeta_pdf_rows": len(all_paths),
        "toolbook_excluded": len(toolbooks),
        "missing_on_disk": len(missing),
        "list_size": len(keep),
    }
    os.makedirs("batch_operation/reports", exist_ok=True)
    if toolbooks:
        with open("batch_operation/reports/filemeta_toolbook_excluded.txt",
                  "w", encoding="utf-8") as f:
            f.write("\n".join(toolbooks) + "\n")
    if missing:
        with open("batch_operation/reports/filemeta_missing_on_disk.txt",
                  "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")
    log.info("[list] %s", json.dumps(meta, ensure_ascii=False))
    return keep, meta


def clone_pipeline(dbo, paths: list[str]):
    """克隆存好的任务管线：去掉 CalibreDataSource（若仍在），清单注入 PDFDataSource。"""
    pipeline = copy.deepcopy(dbo.pipeline)
    new_pipeline = []
    injected = False
    for stage in pipeline:
        name = next(iter(stage))
        if "CalibreDataSource" in name:
            log.info("[clone] dropping CalibreDataSource stage (Calibre 路由已废止)")
            continue
        if not injected and "PDFDataSource" in name:
            stage["PDFDataSource"]["content"] = "\n".join(paths)
            injected = True
        new_pipeline.append(stage)
    if not injected:
        raise RuntimeError("no PDFDataSource stage found in task pipeline")
    clone = SimpleNamespace(
        name=getattr(dbo, "name", "导入 OneDrive PDF") + " (filemeta)",
        pipeline=new_pipeline,
        concurrent=getattr(dbo, "concurrent", 1),
        resume_next=getattr(dbo, "resume_next", False),
        params=getattr(dbo, "params", {}),
    )
    return clone


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="导入 OneDrive PDF")
    ap.add_argument("-n", "--concurrent", type=int, default=0,
                    help="override worker concurrency (0 = task default)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the disk→file_metadata upsert scan")
    ap.add_argument("--progress-secs", type=int, default=120)
    ap.add_argument("--report", default="batch_operation/reports/import_onedrive.json")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        pre = await stats(conn)
        if not args.no_scan:
            await scan_disk_upsert(conn)
        keep, list_meta = await build_file_list(conn)
    finally:
        await conn.close()
    log.info("pre: files without paragraph=%s, paragraph total=%s, import list=%s",
             pre["onedrive_pdfs_without_paragraph"], f"{pre['paragraph_total']:,}",
             list_meta)
    if args.dry_run:
        print(json.dumps({"pre": pre, "list": list_meta},
                         ensure_ascii=False, indent=2))
        return 0

    # --- load task & plugins exactly like `python -m jindai task ...` -----
    from jindai import Task
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins

    dbo = await TaskDBO.get(args.task)
    if not dbo:
        log.error("task %r not found", args.task)
        return 2
    _init_plugins()

    clone = clone_pipeline(dbo, keep)
    logfile = open(args.report + ".log", "a", encoding="utf-8")

    def logfn(*x):
        msg = " ".join(str(i) for i in x)
        print(msg, flush=True)
        print(msg, file=logfile, flush=True)

    task = Task.from_dbo(clone, verbose=args.verbose, log=logfn, use_tqdm=False)
    if args.concurrent > 0:
        task.concurrent = args.concurrent

    watcher = asyncio.create_task(progress_watcher(args.progress_secs))
    t0 = time.time()
    try:
        result = await task.execute_async()
    finally:
        watcher.cancel()
        logfile.close()

    conn = await C.get_conn()
    try:
        post = await stats(conn)
    finally:
        await conn.close()

    summary = {
        "task": args.task,
        "source": "file_metadata",
        "list": list_meta,
        "pre": pre,
        "post": post,
        "imported_files": (
            pre["onedrive_pdfs_without_paragraph"] - post["onedrive_pdfs_without_paragraph"]
        ),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "result": result,
    }
    C.dump_json(args.report, summary)
    log.info("summary: %s", json.dumps(summary, ensure_ascii=False, default=str))
    if isinstance(result, dict) and result.get("__exception__"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
