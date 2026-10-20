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
   SaveParagraph 链不变；数据集按 folder_dataset_names() 规则挂
   （2026-10-10：Zotero→Zotero、Journals→报刊--<二级>、Series→特辑--<二级>、
   OneDrive/Supplementary→书库--<作者>，不挂「书库」基础绑定）。

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
      AND f.path NOT LIKE '.trash/%'
      AND f.path NOT LIKE '../%'
      AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
      -- 已盖章 complete 的文件不再算「待导入」：它们要么有段落，要么整本
      -- 都是确认空白页（全在 page_skip_state）。口径见 _STATE_FILTER。
      AND NOT EXISTS (SELECT 1 FROM file_import_state s
                      WHERE s.source_id = f.id AND s.status = 'complete')
"""
PARA_TOTAL_SQL = "SELECT count(*)::bigint FROM paragraph"
# 2026-10-10 用户指令：按文件夹优先级导入
#   Zotero → Series → OneDrive → Journals → Supplementary Books → 其余
# 清单顺序即处理顺序（已导入的靠 skip_existed + reimport_gaps 按页集快速跳过，
# 所以全量排序不会浪费时间，只会让未导入部分按此优先级推进）。
LIST_SQL = """
    SELECT path FROM file_metadata
    WHERE lower(path) LIKE '%.pdf'
      AND path NOT LIKE 'Tool Books/%'
      AND path NOT LIKE '.trash/%'
      AND path NOT LIKE '../%'
      {state_filter}
    ORDER BY CASE split_part(path, '/', 1)
               WHEN 'Zotero'              THEN 1
               WHEN 'Series'              THEN 2
               WHEN 'OneDrive'            THEN 3
               WHEN 'Journals'            THEN 4
               WHEN 'Supplementary Books' THEN 5
               ELSE 6
             END,
             path
"""
# 「已完成」文件直接出队列（2026-10-17 用户指令）：每个文件导入结束后在
# file_import_state 盖一个 status='complete' 的章；下一轮既不打开它的 PDF，
# 也不跑页集续传查询。缺页/从未导入的文件要么没行、要么 status='partial'，
# 照旧每轮重试。判断口径与 PDFDataSource 的 reimport_gaps 完全一致：
# 页有「非空、非 outline 段落」或在 page_skip_state 里 = 已覆盖。
_STATE_FILTER = """AND NOT EXISTS (
          SELECT 1 FROM file_import_state s
          WHERE s.source_id = file_metadata.id AND s.status = 'complete')"""


def _list_sql(exclude_complete: bool) -> str:
    """按是否排除「已完成」文件生成清单 SQL。"""
    return LIST_SQL.format(
        state_filter=_STATE_FILTER if exclude_complete else "")
EXISTING_ONEDRIVE_SQL = "SELECT path FROM file_metadata WHERE path LIKE $1"

# 工具书（词典/辞书/大部头参考书）——沿用 references/backlog-inventory.md 的关键词表，
# 站规：不导入（不排序到队尾，直接不进批次）。
TOOLBOOK_RE = re.compile(
    r"词典|辭典|字典|大典|辞海|辞源|百科|名物|lexicon|lexica|dictionary|dictionar|"
    r"dictionarium|glossar|vocabular|lexikon|lexique|thesaurus|concordance|"
    r"wörterbuch|woerterbuch|liddell|scott|georges|bekker", re.I)

# --- 启动期缓存（2026-10-16）：让「重启导入」从 ~13 分钟降到秒级 ----------
# 每轮启动要做三件分钟级的事：① 磁盘走一遍（~70k PDF，8–9 分钟）；
# ② stats 的反连接 count（3–5 分钟）；③ build_file_list 再对 70k 路径
# os.path.exists 一遍（磁盘第二趟）。看门狗每次 crash/重启都要重来，
# 迭代修复时全耗在这上面。缓存策略：
#   disk_inventory.json  —— 磁盘清单（path→size），TTL 内复用并直接传给
#                           build_file_list（省掉第二趟 stat）；
#   import_stats_cache.json —— 文件/段落数统计，TTL 内复用。
# TTL 到期自动重算，新文件最迟 TTL 后进入队列；两份缓存都在
# batch_operation/reports/ 下，删文件即回退到全量计算。
STATS_CACHE_FILE = "batch_operation/reports/import_stats_cache.json"
DISK_CACHE_FILE = "batch_operation/reports/disk_inventory.json"


def _cache_read(path: str, cache_secs: float) -> dict | None:
    """读 JSON 缓存；cache_secs<=0、文件缺失、超龄或损坏都返回 None。"""
    if cache_secs <= 0:
        return None
    try:
        age = time.time() - os.stat(path).st_mtime
        if age > cache_secs:
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_age"] = age
        return data
    except (OSError, ValueError, TypeError):
        return None


def _cache_write(path: str, obj: dict) -> None:
    """原子写 JSON 缓存（tmp + replace），失败只记日志不致命。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except OSError as ex:
        log.warning("[cache] write %s failed: %s", path, ex)


# --- 文件级导入状态（2026-10-17 用户指令）-----------------------------------
# 每个文件导入结束后盖章：所有页都有「非空、非 outline 段落」或在
# page_skip_state（确认空白/坏页）→ status='complete'，下一轮直接出队列，
# 不再打开 PDF、不再跑页集续传查询；否则 status='partial'（缺页/没导过），
# 照旧每轮重试。判据与 PDFDataSource 的 reimport_gaps 用的 SQL 完全一致。
IMPORT_STATE_TABLE = "file_import_state"

# 页覆盖判据（和 pdfdatasource 的 reimport_gaps 逐字一致）：
_COVERED_PAGES_SQL = """
    SELECT count(DISTINCT source_page) FROM (
        SELECT source_page FROM paragraph
        WHERE source_id = $1 AND source_page IS NOT NULL
          AND outline IS NULL AND btrim(coalesce(content, '')) <> ''
        UNION
        SELECT source_page FROM page_skip_state
        WHERE source_id = $1 AND source_page IS NOT NULL
    ) t
"""


async def ensure_import_state_table(conn) -> None:
    """建 file_import_state（幂等）。批处理自建，不碰 web 的 SQLModel 表。

    `source_id` 是 uuid（对齐 file_metadata.id / paragraph.source_id，
    2026-10-17 首次跑时才发现这两列不是 text）。
    """
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {IMPORT_STATE_TABLE} (
            source_id     uuid PRIMARY KEY,
            status        text NOT NULL,          -- 'complete' | 'partial'
            pages_total   int,
            pages_covered int,
            updated_at    timestamptz NOT NULL DEFAULT now()
        )
    """)


async def count_covered_pages(conn, source_id) -> int:
    """该文件「已覆盖」的页数（有非空段落，或在 page_skip_state）。"""
    return int(await conn.fetchval(_COVERED_PAGES_SQL, source_id) or 0)


async def record_import_state(conn, source_id, pages_total,
                              pages_covered) -> str:
    """盖章：返回 'complete' / 'partial'。pages_total 未知（打不开 PDF）
    时不写状态——留待下一轮重试。"""
    if not pages_total:
        return "unknown"
    status = "complete" if pages_covered >= pages_total else "partial"
    await conn.execute(
        f"""INSERT INTO {IMPORT_STATE_TABLE}
                (source_id, status, pages_total, pages_covered, updated_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (source_id) DO UPDATE
               SET status = EXCLUDED.status,
                   pages_total = EXCLUDED.pages_total,
                   pages_covered = EXCLUDED.pages_covered,
                   updated_at = now()""",
        source_id, status, int(pages_total), int(pages_covered))
    # 冗余到 file_metadata.import_state（2026-10-20）：监控/清单直接查这一列，
    # 不必再做 paragraph 反连接。列不存在（老镜像/未跑迁移）时静默跳过。
    try:
        await conn.execute(
            "UPDATE file_metadata SET import_state = $2 WHERE id = $1::uuid",
            source_id, status)
    except Exception as ex:
        log.warning("[state] import_state 冗余失败（忽略）: %s", str(ex)[:120])
    return status


async def stats(conn, cache_secs: float = 0) -> dict:
    """文件级统计（纯监控口径）。

    ⚠️ 这两个 count 是**分钟级**查询：反连接要对 ~70k 个 file_metadata 行
    逐个探测 paragraph 的 32 个 HASH 分区（每个 `paragraph_N_source_idx`
    index-only scan），实测 3–5 分钟；而启动时跑一次、进度 watcher 每 10
    分钟又跑一次，等于长期占着库。

    因此：`cache_secs > 0` 时把结果写进
    ``reports/import_stats_cache.json`` 并在 TTL 内直接复用（日志
    ``[stats] reuse``）；`cache_secs = 0` 保持旧行为（每次都真查）。
    无论是否启用缓存，真查过一次都会回写缓存文件，方便后续轮次命中。
    """
    if cache_secs > 0:
        cached = _cache_read(STATS_CACHE_FILE, cache_secs)
        if cached is not None:
            log.info("[stats] reuse cached stats (age %.0fs): "
                     "files_without_paragraph=%s paragraphs=%s",
                     cached["_age"], cached["onedrive_pdfs_without_paragraph"],
                     f"{cached['paragraph_total']:,}")
            return {"onedrive_pdfs_without_paragraph":
                        cached["onedrive_pdfs_without_paragraph"],
                    "paragraph_total": cached["paragraph_total"]}
    missing = await conn.fetchval(MISSING_PDF_SQL)
    paragraphs = await conn.fetchval(PARA_TOTAL_SQL)
    out = {"onedrive_pdfs_without_paragraph": missing,
           "paragraph_total": paragraphs}
    _cache_write(STATS_CACHE_FILE, {**out, "ts": time.time()})
    return out


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
                            AND f.path NOT LIKE '.trash/%'
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


# 2026-10-09 用户指令「全部补」：登记入口从 OneDrive 扩到 /storage 全库
# （Tool Books 与隐藏目录除外）；item_type 按顶层目录。
# 隐藏目录剪枝就是回收站 `.trash/`（删除记录时文件的去处）不被重新登记的保障，
# 两个口径（隐藏目录剪枝 + 各 SQL 的 path NOT LIKE '.trash/%'）都要留着。
SCAN_ITEM_TYPE = {"Journals": "proceedings", "Series": "proceedings",
                  "Zotero": "journalArticle", "Albums": "multimedia"}
SCAN_SKIP_DIRS = {"Tool Books"}


async def scan_disk_upsert(conn, cache_secs: float = 0) -> dict:
    """把磁盘上存在、file_metadata 还没有的 PDF 补进表里（幂等，全库扫描）。

    Calibre 停用后没有别的东西会登记新文件——这一步就是新文件进入导入队列的入口。
    2026-10-09 起覆盖 /storage 全部顶层目录（Tool Books 除外）。

    `cache_secs > 0` 时把走盘结果写进 ``reports/disk_inventory.json``
    （path → size），TTL 内复用（日志 ``[scan] reuse disk inventory``）：
    走盘本身 ~70k 个文件、8–9 分钟，是重启最贵的一步。返回磁盘清单，
    调用方把它直接传给 :func:`build_file_list`，省掉第二趟 os.path.exists。

    Returns:
        磁盘上的 PDF 清单 `{相对路径: 字节数}`。
    """
    from jindai.storage import storage
    root = storage.safe_join("OneDrive")[: -len("OneDrive")]
    disk: dict[str, int] = {}
    cached = _cache_read(DISK_CACHE_FILE, cache_secs)
    if cached is not None and isinstance(cached.get("sizes"), dict):
        disk = {p: int(sz) for p, sz in cached["sizes"].items()}
        log.info("[scan] reuse disk inventory cache (age %.0fs, %d pdfs); "
                 "new files will be registered after TTL",
                 cached["_age"], len(disk))
    else:
        n_dirs = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in SCAN_SKIP_DIRS]
            n_dirs += 1
            for fn in filenames:
                if not fn.lower().endswith(".pdf"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                try:
                    disk[rel] = os.path.getsize(full)
                except OSError:
                    disk[rel] = 0
            if n_dirs % 2000 == 0:
                log.info("[scan] walked %d dirs, %d pdfs so far", n_dirs, len(disk))
        _cache_write(DISK_CACHE_FILE, {"ts": time.time(), "sizes": disk})

    existing = {r["path"] for r in await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND path NOT LIKE 'Tool Books/%' AND path NOT LIKE '.trash/%'")}
    new = [(p, disk[p], SCAN_ITEM_TYPE.get(p.split("/")[0], "book"))
           for p in disk if p not in existing]
    if new:
        await conn.executemany(
            "INSERT INTO file_metadata (path, extension, size_bytes, item_type, import_state) "
            "VALUES ($1, 'pdf', $2, $3, 'pending') ON CONFLICT (path) DO NOTHING",
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
    return disk


async def build_file_list(conn, disk: dict | None = None,
                          exclude_complete: bool = True) -> tuple[list[str], dict]:
    """导入清单（按用户指定的目录优先级排序）。

    `disk` 传入 :func:`scan_disk_upsert` 的磁盘清单时，存在性判断直接查
    内存（O(1)）；否则退回去对 ~70k 个路径逐个 ``os.path.exists``——那
    相当于把磁盘再走一趟，分钟级，是启动慢的第二个来源。

    `exclude_complete=True`（默认）把已盖章 complete 的文件排除出清单
    （2026-10-17）：完成过的 PDF 连队列都不进，省掉每轮的打开 + 页集
    续传查询。False = 全量清单（旧行为）。
    """
    await ensure_import_state_table(conn)
    rows = await conn.fetch(_list_sql(exclude_complete))
    all_paths = [r["path"] for r in rows]
    toolbooks = [p for p in all_paths if TOOLBOOK_RE.search(p)]
    candidates = [p for p in all_paths if p not in set(toolbooks)]
    done = 0
    if exclude_complete:
        done = int(await conn.fetchval(
            f"SELECT count(*) FROM {IMPORT_STATE_TABLE} "
            f"WHERE status = 'complete'") or 0)

    from jindai.storage import storage
    if disk is not None:
        missing = [p for p in candidates if p not in disk]
    else:
        missing = [p for p in candidates
                   if not os.path.exists(storage.safe_join(p))]
    keep = [p for p in candidates if p not in set(missing)]

    meta = {
        "filemeta_pdf_rows": len(all_paths),
        "toolbook_excluded": len(toolbooks),
        "missing_on_disk": len(missing),
        "list_size": len(keep),
        "import_state_complete_excluded": done,
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
        disk = None
        if not args.no_scan:
            disk = await scan_disk_upsert(conn)
        keep, list_meta = await build_file_list(conn, disk=disk)
        # 段错误毒文件清单：reports/import_skip.txt（每行一个 path，# 注释）。
        # 看门狗 import_watchdog.sh 崩溃后会自动把肇事文件追加进来再重启。
        skip_file = "batch_operation/reports/import_skip.txt"
        if os.path.exists(skip_file):
            skip = {ln.strip() for ln in open(skip_file, encoding="utf-8")
                    if ln.strip() and not ln.startswith("#")}
            before = len(keep)
            keep = [p_ for p_ in keep if p_ not in skip]
            if before != len(keep):
                log.warning("skip-file: 从清单剔除 %d 个毒文件",
                            before - len(keep))
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
    # 2026-10-10 用户指令：**单个文件出错时跳过该文件继续**，不要让整单停。
    # task.py::_run_chain 在 except 里 `if not self.resume_next: self.alive = False`，
    # resume_next=True 即"continue on error"（jindai/models.py TaskDBO 注释同义）。
    clone.resume_next = True
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
