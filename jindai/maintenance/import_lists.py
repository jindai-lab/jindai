"""PDF 导入清单 / 磁盘登记 / 导入状态盖章（2026-10-25 并入主库，
2026-10-29 收入 ``jindai/maintenance/`` 包）。

原实现是 batch_operation/import_onedrive_pdfs.py（该文件已删除，其调用方
classify_imported.py 等现在直接 ``from jindai.maintenance.import_lists import ...``）。

口径（长期站规，勿改）：
- 清单只含 import_state IN ('pending','partial') + IS NULL 的 PDF 行，
  partial 优先排序，目录优先级 Zotero→Series→OneDrive→Journals→Supplementary；
- Tool Books/ 与 .trash/、../ 遗留行不进清单；
- 完成盖章 = file_import_state.status='complete'（判据与 PDFDataSource 的
  reimport_gaps 逐字一致：页有非空非 outline 段落或在 page_skip_state），
  同时冗余到 file_metadata.import_state 列；
- 磁盘扫描是新文件进入导入队列的唯一入口（Calibre 停用后无其它登记方）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import asyncpg

# 报表 / 缓存 / 锁 / 状态目录的**唯一来源**
REPORTS = Path(__file__).resolve().parent[2] / 'importer'

log = logging.getLogger("jindai.import")

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

LIST_SQL = """
    SELECT path FROM file_metadata
    WHERE lower(path) LIKE '%.pdf'
      AND path NOT LIKE 'Tool Books/%'
      AND path NOT LIKE '.trash/%'
      AND path NOT LIKE '../%'
      {state_filter}
    ORDER BY {state_order}CASE split_part(path, '/', 1)
               WHEN 'Zotero'              THEN 1
               WHEN 'Series'              THEN 2
               WHEN 'OneDrive'            THEN 3
               WHEN 'Journals'            THEN 4
               WHEN 'Supplementary Books' THEN 5
               ELSE 6
             END,
             path
"""
# 清单口径（2026-10-21 用户指令）：只含 pending + partial + 未盖章 NULL 新行；
# complete/imported/excluded 一律不进队列；partial 排前（先补完缺页再做新书）。
_STATE_FILTER = """AND (import_state IN ('pending', 'partial')
                        OR import_state IS NULL)"""
_STATE_ORDER = "CASE WHEN import_state = 'partial' THEN 0 ELSE 1 END, "

# 工具书（词典/辞书/大部头参考书）——站规：不导入（不排队尾，直接不进批次）。
TOOLBOOK_RE = re.compile(
    r"词典|辭典|字典|大典|辞海|辞源|百科|名物|lexicon|lexica|dictionary|dictionar|"
    r"dictionarium|glossar|vocabular|lexikon|lexique|thesaurus|concordance|"
    r"wörterbuch|woerterbuch|liddell|scott|georges|bekker", re.I)

STATS_CACHE_FILE = REPORTS / "import_stats_cache.json"
DISK_CACHE_FILE = REPORTS / "disk_inventory.json"


# --------------------------------------------------------------------------
# asyncpg 连接（与 batch_operation/common.py::get_conn 同 DSN 口径）
# --------------------------------------------------------------------------
def _dsn() -> str:
    from ..config import config
    return config.database.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(dsn=_dsn())


# --------------------------------------------------------------------------
# 启动期缓存：让「重启导入」从 ~13 分钟降到秒级
# --------------------------------------------------------------------------
def _cache_read(path, cache_secs: float) -> dict | None:
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


def _cache_write(path, obj: dict) -> None:
    """原子写 JSON 缓存（tmp + replace），失败只记日志不致命。

    ⚠️ tmp 必须带 pid 后缀：两个并发写者（如引擎轮走盘 + 手动验证走盘）共用
    同一个 `.tmp` 路径时会互踩句柄，replace 进去的还是截断的混合内容
    （2026-10-26 实证：缓存 JSON 恒截断在 2.55MB 处，每轮退化成全盘重走）。
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except OSError as ex:
        log.warning("[cache] write %s failed: %s", path, ex)


def cached_disk_inventory(cache_secs: float) -> dict | None:
    """新鲜的磁盘清单（path→size）或 None。"""
    cached = _cache_read(DISK_CACHE_FILE, cache_secs)
    if cached is not None and isinstance(cached.get("sizes"), dict):
        return {p: int(sz) for p, sz in cached["sizes"].items()}
    return None


# --------------------------------------------------------------------------
# 文件级导入状态（盖章）
# --------------------------------------------------------------------------
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
    """建 file_import_state（幂等）。"""
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
    """盖章：返回 'complete' / 'partial'。pages_total 未知时不写状态。"""
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
    # 冗余到 file_metadata.import_state：监控/清单直接查这一列。
    try:
        await conn.execute(
            "UPDATE file_metadata SET import_state = $2 WHERE id = $1::uuid",
            source_id, status)
    except Exception as ex:
        log.warning("[state] import_state 冗余失败（忽略）: %s", str(ex)[:120])
    return status


# --------------------------------------------------------------------------
# 统计（分钟级反连接，务必带缓存）
# --------------------------------------------------------------------------
async def stats(conn, cache_secs: float = 0) -> dict:
    """文件级统计（纯监控口径）。

    这两个 count 是分钟级查询（~70k 行 × 32 分区 index-only 探测，3–5 分钟），
    cache_secs>0 时走 import_stats_cache.json TTL 复用。
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


async def pending_counts(conn) -> dict:
    """清单口径的 pending/partial 计数（import_state 部分索引，秒回）。"""
    n_partial = int(await conn.fetchval(
        "SELECT count(*) FROM file_metadata "
        "WHERE import_state = 'partial' AND lower(path) LIKE '%.pdf'") or 0)
    n_pending = int(await conn.fetchval(
        "SELECT count(*) FROM file_metadata "
        "WHERE (import_state = 'pending' OR import_state IS NULL) "
        "  AND lower(path) LIKE '%.pdf'") or 0)
    return {"pending": n_pending, "partial": n_partial,
            "total": n_pending + n_partial}


# --------------------------------------------------------------------------
# 磁盘扫描 upsert（新文件进队列的唯一入口）
# --------------------------------------------------------------------------
SCAN_ITEM_TYPE = {"Journals": "proceedings", "Series": "proceedings",
                  "Zotero": "journalArticle", "Albums": "multimedia"}
SCAN_SKIP_DIRS = {"Tool Books"}


def walk_disk(root: str | None = None) -> dict[str, int]:
    """阻塞式走盘：返回 {storage 相对路径: 字节数}（隐藏目录剪枝）。

    CIFS 全 walk 实测 8–15 分钟——调用方（web 进程内）必须放线程池跑，
    绝不能直接在事件循环里调用。
    """
    if root is None:
        from ..storage import storage
        root = storage.safe_join("OneDrive")[: -len("OneDrive")]
    disk: dict[str, int] = {}
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
    return disk


def dedup_key(basename: str) -> str:
    """登记查重键：文件名 NFC 归一 + 小写。

    ⚠️ **不含字节数**（2026-10-26 用户指令）：同一本书的两份拷贝可能一份加了
    批注/书签，大小不同，字节数会漏判；文件名（Calibre 式《书名 - 作者》/
    丛刊卷名）在本库足够唯一。误判方向也有护栏：skipped 只是不登记、文件
    留在盘上，且全部留档 reports/scan_dedup_skipped.txt 可审计。
    """
    import unicodedata
    return unicodedata.normalize("NFC", basename).lower()


async def dedup_new_paths(conn, new_paths):
    """登记前查重：与库里已有 PDF 行**同文件名**（NFC 小写）→ 判为同一文件的
    另一份拷贝，跳过登记。返回 (keep[path 列表], skipped[(path, twin_path)])。

    2026-10-26 教训：盘上重复目录树（Journals/<丛刊> ↔ Series/<丛刊>、
    OneDrive/<中文作者> ↔ OneDrive/<拉丁作者>）被全盘扫描登记出 8,234 条死路径
    空壳行（正主行早已在库）；有此护栏则整批只留档、不登记。同一次批内互为
    拷贝的也只登记首个。skipped 落 ``reports/scan_dedup_skipped.txt``，
    文件留在盘上不删。
    """
    rows = await conn.fetch(
        "SELECT path FROM file_metadata "
        "WHERE lower(path) LIKE '%.pdf' AND path NOT LIKE 'Tool Books/%' "
        "AND path NOT LIKE '.trash/%'")
    seen = {}
    for r in rows:
        seen.setdefault(dedup_key(r["path"].split("/")[-1]), r["path"])
    keep, skipped = [], []
    for pth in new_paths:
        k = dedup_key(pth.split("/")[-1])
        if k in seen:
            skipped.append((pth, seen[k]))
        else:
            keep.append(pth)
            seen.setdefault(k, pth)
    if skipped:
        path = REPORTS / "scan_dedup_skipped.txt"
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {time.strftime('%F %T')} dedup-skipped {len(skipped)} "
                    "files (basename NFC-lower matched an existing row; "
                    "size NOT in key per user rule 2026-10-26)\n")
            for pth, twin in sorted(skipped):
                f.write(f"{pth}\t<=>\t{twin}\n")
        log.info("[scan] dedup: skipped %d files already registered under other "
                 "paths (see reports/scan_dedup_skipped.txt)", len(skipped))
    return keep, skipped


async def upsert_disk(conn, disk: dict[str, int]) -> dict:
    """把磁盘上有、表里没有的 PDF upsert 进 file_metadata（幂等，含查重护栏）。"""
    existing = {r["path"] for r in await conn.fetch(
        "SELECT path FROM file_metadata WHERE lower(path) LIKE '%.pdf' "
        "AND path NOT LIKE 'Tool Books/%' AND path NOT LIKE '.trash/%'")}
    candidates = [pth for pth in disk if pth not in existing]
    keep, dedup_skipped = await dedup_new_paths(conn, candidates)
    new = [(pth, disk[pth], SCAN_ITEM_TYPE.get(pth.split("/")[0], "book"))
           for pth in keep]
    if new:
        await conn.executemany(
            "INSERT INTO file_metadata (path, extension, size_bytes, item_type, import_state) "
            "VALUES ($1, 'pdf', $2, $3, 'pending') ON CONFLICT (path) DO NOTHING",
            new,
        )
    # 表里有、磁盘没有的（陈旧路径）单独留档，不在此删行
    stale = sorted(existing - set(disk))
    if stale:
        path = REPORTS / "filemeta_stale_paths.txt"
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(stale) + "\n")
    out = {"disk_pdfs": len(disk), "existing_rows": len(existing),
           "upserted": len(new), "dedup_skipped": len(dedup_skipped),
           "stale_on_disk": len(stale),
           "new_paths": [p for p, _, _ in new]}
    log.info("[scan] disk pdfs=%d existing rows=%d upserted=%d "
             "dedup-skipped=%d stale-on-disk=%d (see reports/filemeta_stale_paths.txt)",
             out["disk_pdfs"], out["existing_rows"], out["upserted"],
             out["dedup_skipped"], out["stale_on_disk"])
    return out


async def scan_disk_upsert(conn, cache_secs: float = 0,
                           disk: dict[str, int] | None = None) -> dict:
    """走盘（或复用缓存/调用方给的清单）+ upsert。返回磁盘清单。

    ⚠️ 缓存只在**真正走盘后**写（TTL 语义）：命中缓存时若也回写，mtime 永远
    被续期，新文件永远检测不到（2026-10-25 实现迁移时踩过）。
    """
    if disk is None and cache_secs > 0:
        disk = cached_disk_inventory(cache_secs)
        if disk is not None:
            log.info("[scan] reuse disk inventory cache (%d pdfs); "
                     "new files will be registered after TTL", len(disk))
    if disk is None:
        disk = await _walk_disk_async()
        _cache_write(DISK_CACHE_FILE, {"ts": time.time(), "sizes": disk})
    out = await upsert_disk(conn, disk)
    # 站规：任何批量登记后必跑 fill_nontitle（2026-10-26 防再犯：引擎 10:00 登记
    # 8,234 行后无人回填，空标题挂了一上午——auto_ingest 的 fill 只在它自己登记
    # 新行时才跑、且只扫 OneDrive/Zotero，兜不住引擎的全盘登记）。
    if out.get("upserted"):
        import asyncio
        await asyncio.to_thread(_run_fill_nontitle)
    return disk


async def _walk_disk_async() -> dict[str, int]:
    """在线程池里跑阻塞走盘（web 进程内不许阻塞事件循环）。"""
    import asyncio
    return await asyncio.to_thread(walk_disk)


def _run_fill_nontitle() -> None:
    """登记后回填 title/authors/item_type（幂等，只碰空 title 的 PDF 行）。

    实现在 ``.fill_nontitle``（2026-10-29 从 batch_operation/fill_nontitle_rows.py
    迁入）：核心包不再 subprocess 拉运维脚本，也不再依赖 cwd=/app。
    """
    from .fill_nontitle import fill_nontitle_sync

    try:
        summary = fill_nontitle_sync(apply=True)
        log.info("[scan] fill_nontitle updated=%s failed=%s empty_after=%s",
                 summary["applied"]["updated"], summary["applied"]["failed"],
                 summary["empty_title_after"])
    except Exception as ex:  # noqa: BLE001 - 回填失败不阻断导入轮
        log.warning("[scan] fill_nontitle failed: %s", ex)


# --------------------------------------------------------------------------
# 导入清单
# --------------------------------------------------------------------------
def _list_sql(exclude_complete: bool) -> str:
    if exclude_complete:
        return LIST_SQL.format(state_filter=_STATE_FILTER,
                               state_order=_STATE_ORDER)
    return LIST_SQL.format(state_filter="", state_order="")


async def build_file_list(conn, disk: dict | None = None,
                          exclude_complete: bool = True) -> tuple[list[str], dict]:
    """导入清单（按目录优先级排序）。`disk` 给磁盘清单时跳过 exists 第二趟。"""
    await ensure_import_state_table(conn)
    rows = await conn.fetch(_list_sql(exclude_complete))
    all_paths = [r["path"] for r in rows]
    toolbooks = [p for p in all_paths if TOOLBOOK_RE.search(p)]
    candidates = [p for p in all_paths if p not in set(toolbooks)]
    done = 0
    n_partial = n_pending = 0
    if exclude_complete:
        done = int(await conn.fetchval(
            f"SELECT count(*) FROM {IMPORT_STATE_TABLE} "
            f"WHERE status = 'complete'") or 0)
        pc = await pending_counts(conn)
        n_partial, n_pending = pc["partial"], pc["pending"]

    from ..storage import storage
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
        "list_partial": n_partial,
        "list_pending": n_pending,
    }
    os.makedirs(REPORTS, exist_ok=True)
    if toolbooks:
        with open(REPORTS / "filemeta_toolbook_excluded.txt",
                  "w", encoding="utf-8") as f:
            f.write("\n".join(toolbooks) + "\n")
    if missing:
        with open(REPORTS / "filemeta_missing_on_disk.txt",
                  "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")
    log.info("[list] %s", json.dumps(meta, ensure_ascii=False))
    return keep, meta


# --------------------------------------------------------------------------
# 毒文件黑名单 / 暂缓清单（清单过滤，语义分离）
# --------------------------------------------------------------------------
SKIP_FILE = REPORTS / "import_skip.txt"
DEFER_FILE = REPORTS / "import_defer.txt"


def read_skip_file() -> set[str]:
    if not SKIP_FILE.exists():
        return set()
    return {ln.strip() for ln in open(SKIP_FILE, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")}


def apply_skip_defer(keep: list[str]) -> tuple[list[str], dict]:
    """剔除毒文件黑名单 + 暂缓清单（前缀以 / 结尾为前缀匹配）。"""
    skip = read_skip_file()
    if skip:
        before = len(keep)
        keep = [p for p in keep if p not in skip]
        if before != len(keep):
            log.warning("skip-file: 从清单剔除 %d 个毒文件", before - len(keep))
    if DEFER_FILE.exists():
        prefixes, exact = [], set()
        for ln in open(DEFER_FILE, encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.endswith("/"):
                prefixes.append(ln)
            else:
                exact.add(ln)
        before = len(keep)
        keep = [p for p in keep
                if p not in exact and not any(p.startswith(x) for x in prefixes)]
        if before != len(keep):
            log.warning("defer-file: 从清单暂缓剔除 %d 个文件（%s）",
                        before - len(keep), DEFER_FILE)
    return keep, {"skip_count": len(skip), "list_after_filters": len(keep)}
