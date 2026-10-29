"""PDF 导入引擎：直接按模块构造 stage 链，不读 TaskDBO（2026-10-25 并入主库，
2026-10-30 从 ``jindai/maintenance/`` 移入 ``plugins/importer/``）。

原实现在 batch_operation/batch_import_direct.py（2026-10-18 引擎）。现在那份脚本
只剩进程入口（``from plugins.importer.import_cli import main``），保留它是因为
进程名含 ``batch_import_direct``，``gpu_cpu_monitor`` / ``auto_ingest_scoped`` 的
ps grep 依赖它；引擎实现全部在本模块。

管线（就地构造，不查 task_dbo.pipeline）：
    PDFDataSource(文本层抽取 + 扫描页 OCR + 页集续传)
      -> LanguageDetect()            语种检测
      -> Reparagraph()               连缀断行成段
      -> Condition(lang in zh/ja)    CJK 去空白（RepeatWhile x4 RegexReplace）
      -> Condition(content.strip())  西文连字符修复 -> WordCut -> 批量落库

行为契约（改代码前必读，均有 2026 实证事故支撑）：
- 文件级并发（每 worker 独立 stage 链与缓冲，_dbsession 非并发安全）；
- 「一个文件完全处理完才换下一个」：页未盖全原地重跑补缺页，上限
  IMPORT_MAX_ATTEMPTS，到顶仍缺页记 partial 交下轮；
- 盖章前必须先 flush 落库（count_covered_pages 只认已提交行）；
- 在飞标记（import_direct_inflight/wN）行尾必须带换行（看门狗 cat 收集）；
- bulk insert 失败降级逐段隔离毒段；terms 词表失败只降级不回滚段落。
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import regex as re2  # \p{Other} 等 Unicode 属性

from .import_lists import (
    REPORTS, apply_skip_defer, build_file_list, cached_disk_inventory,
    count_covered_pages, get_conn, record_import_state, scan_disk_upsert,
    stats,
)

log = logging.getLogger("jindai.import")

# stage 日志落点（监控/排障 grep 的是这两个文件，路径勿改）
STAGE_LOG = REPORTS / "import_onedrive.json.log"
RUN_LOG = REPORTS / "import_onedrive.log"
INFLIGHT_DIR = REPORTS / "import_direct_inflight"
# 与旧引擎共用同一把锁：常驻 worker 拉起的子进程与手工导入绝不允许同时跑
LOCK_FILE = REPORTS / "import_fast.lock"
SKIP_FILE = REPORTS / "import_skip.txt"
ONLY_LIST = REPORTS / "import_direct_only.txt"
# 轮内进度快照（常驻 worker 的 /import/ API 读它显示当前导入状态）
STATUS_FILE = REPORTS / "import_status.json"

# dataset_name 只用于 PDFDataSource 内部 Dataset.get(...).id（传给跨页重排器，
# 本链 reparagraphize=False 不启用）。文件级数据集绑定早已在登记时挂好。
PDF_PARAMS: dict[str, Any] = {
    "dataset_name": "书库",
    "lang": "auto",
    "skip_existed": True,
    "reimport_gaps": True,      # 页集续传（本轮的唯一续传口径）
    "reparagraphize": False,    # 跨页重排关：连缀交给链上的 Reparagraph
    "ocr_fallback": True,       # 扫描页 OCR
    "ocr_workers": 4,
    "ocr_mode": "pdf",
    "clean_text": True,
    "remove_garbled": True,
    "garbled_threshold": 0.3,
    "fix_ocr_errors": True,
}

# 就地续跑上限：文件没盖满页时原地重跑（代价很小），防毒页卡死可环境变量调。
MAX_INLINE_ATTEMPTS = int(os.environ.get("IMPORT_MAX_ATTEMPTS", "3"))

# PDFDataSource 每个单文件实例自己会打 "1/1 importing <path>"；worker 已打过
# 带全局进度的 "N/M importing <path>"，把前者滤掉。
_FETCH_IMPORTING_RE = re2.compile(r"\b1/1 importing ")


def make_logfn(logfile: Path | str | None = None):
    """stage/数据源日志：stdout（看门狗→import_onedrive.log）+ json.log。"""
    logfile = Path(logfile) if logfile else STAGE_LOG
    fh = open(logfile, "a", encoding="utf-8")

    def logfn(*x):
        msg = " ".join(str(i) for i in x)
        if _FETCH_IMPORTING_RE.search(msg):
            return
        print(msg, flush=True)
        print(msg, file=fh, flush=True)

    logfn.close = fh.close
    return logfn


# ---------------------------------------------------------------------------
# 批量版 SaveParagraph（内容清洗 + 关键词 600B 截断 + Terms.store）
# ---------------------------------------------------------------------------
def _make_saver_class(batch_default: int):
    """构建 BatchSaveParagraph（SaveParagraph 的批量落库替身）。

    行为对齐 plugins/pipelines/basics.py::SaveParagraph：
    - content 过滤 \\p{Other}
    - keywords 600 UTF-8 字节截断 + 空值过滤
    - Terms.store(keywords) —— 检索词表
    区别：不逐段 commit，先缓冲 batch 段再一次事务 add_all 落库。
    """
    from jindai.pipeline import PipelineStage
    from jindai.models import Paragraph, Terms, get_db_session

    class BatchSaveParagraph(PipelineStage):
        """批量版 SaveParagraph：缓冲 + 每批一次事务；段尾 flush。"""

        def __init__(self, batch: int = batch_default) -> None:
            super().__init__()
            self.batch = max(1, int(batch))
            self.buf: list = []
            self.kw: set = set()
            self.saved = 0
            self.failed = 0

        async def resolve(self, paragraph: Paragraph) -> Paragraph:
            paragraph.content = re2.sub(r"\p{Other}", " ", paragraph.content or "")
            if paragraph.keywords:
                paragraph.keywords = [
                    kw for kw in paragraph.keywords
                    if kw and len(kw.encode("utf-8")) <= 600
                ]
                self.kw.update(paragraph.keywords)
            if paragraph.id:
                # 导入路径不会带 id（from_dict 忽略 id）；防御性逐条落库。
                await self._insert_one(paragraph)
                return paragraph
            self.buf.append(paragraph)
            if len(self.buf) >= self.batch:
                await self.flush()
            return paragraph

        async def _insert_one(self, p: Paragraph) -> None:
            async with get_db_session() as s:
                await s.merge(p)
                await s.commit()

        async def flush(self) -> None:
            if not self.buf:
                return
            buf, self.buf = self.buf, []
            kws, self.kw = sorted(self.kw), set()
            ok = 0
            try:
                async with get_db_session() as session:
                    session.add_all(buf)
                    await session.flush()
                    await session.commit()
                ok = len(buf)
            except Exception as ex:
                self.log(
                    f"[flush] bulk insert failed "
                    f"({type(ex).__name__}: {str(ex)[:300]}), row-by-row fallback"
                )
                consec = 0
                for p in buf:
                    try:
                        async with get_db_session() as s:
                            s.add(p)
                            await s.commit()
                        ok += 1
                        consec = 0
                    except Exception as ex2:
                        self.failed += 1
                        consec += 1
                        if self.failed <= 5:
                            self.log(
                                f"[flush] row dropped: {type(ex2).__name__}: "
                                f"{str(ex2)[:300]}"
                            )
                        if consec >= 10:
                            self.log("[flush] 10 consecutive row failures "
                                     "(DB down?) -> abort this batch")
                            break
            if kws:
                try:
                    await Terms.store(kws)
                except Exception as ex:
                    # terms 只是词表，丢了退化但不丢正文；绝不回滚段落
                    self.log(
                        f"[flush] terms store failed: "
                        f"{type(ex).__name__}: {str(ex)[:200]}"
                    )
            self.saved += ok

        def clear(self) -> None:
            self.buf = []
            self.kw = set()

    return BatchSaveParagraph


def build_chain(batch: int, logfn=None):
    """按插件模块就地构造 PDF 导入链。

    Returns:
        (BatchSaveParagraph 类, Pipeline 实例, saver 实例)
    """
    from jindai.pipeline import Pipeline
    from plugins.pipelines.basics import (
        LanguageDetect, RegexReplace, Reparagraph, WordCut,
    )
    from plugins.pipelines.controls import Condition, RepeatWhile

    BatchSaveParagraph = _make_saver_class(batch)
    saver = BatchSaveParagraph(batch)

    # CJK：去掉所有空白（OCR/PDF 的中文断行残留）
    cjk_clean = Condition(
        cond="lang in ('zhs', 'zht', 'ja')",
        iftrue=[RepeatWhile(
            pipeline=[RegexReplace(pattern=r"\s+", replacement="")],
            times=4,
        )],
        iffalse=[],
    )
    # 正文清洗 → 检索词 → 落库（含 Terms.store）
    save_if_content = Condition(
        cond="content.strip()",
        iftrue=[
            RegexReplace(pattern=r"([a-z])-\s([a-z])", replacement=r"\1\2"),
            WordCut(for_search=True),
            saver,
        ],
        iffalse=[],
    )

    pipeline = Pipeline(
        [LanguageDetect(), Reparagraph(), cjk_clean, save_if_content],
        log=logfn or (lambda *x: log.info(" ".join(str(i) for i in x))),
    )
    pipeline.gctx = {}
    return BatchSaveParagraph, pipeline, saver


def find_saver(stages):
    """在 stage 树里递归找 BatchSaveParagraph 实例（藏在 Condition.iftrue 里）。"""
    for s in stages:
        if type(s).__name__ == "BatchSaveParagraph":
            return s
        for attr in ("iftrue", "iffalse", "pipeline"):
            sub = getattr(s, attr, None)
            if sub is not None and hasattr(sub, "stages"):
                r = find_saver(sub.stages)
                if r is not None:
                    return r
    return None


async def run_chain(paragraph, stage) -> None:
    """把一个段落沿 stage 链深度优先推到底（Task._run_chain 的直跑版）。"""
    if paragraph is None or stage is None:
        return
    agen = stage.flow(paragraph)
    try:
        async for result, nxt in agen:
            if result is None:
                continue
            await run_chain(result, nxt)
    finally:
        await agen.aclose()


async def import_one(path: str, head, ds_params: dict, logfn) -> int | None:
    """单文件：直接构造 PDFDataSource 调 fetch()，段落走内存链。"""
    from plugins.datasources.pdfdatasource import PDFDataSource

    params = dict(ds_params)
    params["content"] = path
    ds = PDFDataSource(**params)
    # log 必须在 apply_params 之前挂上：apply_params 第一行就打
    # "File Paths: ..."，晚挂会让它落到 root logger。
    ds.log = logfn
    ds.gctx = {}
    ds.apply_params(**params)
    async for para in ds.fetch():
        await run_chain(para, head)
    return getattr(ds, "last_page_count", None)


async def worker(wid, files, counter, total, batch, ds_params, logfn,
                 state, store_state=True) -> None:
    """一个文件级 worker：独立 stage 链 + 独立缓冲，共享文件队列。"""
    _cls, pipeline, saver = build_chain(batch, logfn)
    head = pipeline.stages[0]
    assert saver is not None

    os.makedirs(INFLIGHT_DIR, exist_ok=True)
    marker = os.path.join(INFLIGHT_DIR, f"w{wid}")
    my_saved = 0
    my_failed_rows = 0
    state_conn: Any = None

    async def flush_stats() -> None:
        """落库缓冲 + 更新计数。盖章前必须先调用（否则数不到本文件的段落）。"""
        nonlocal my_saved, my_failed_rows
        try:
            await saver.flush()
        except Exception as ex:
            logfn(f"[error] flush failed, dropping buffer "
                  f"({saver.saved} saved so far): {type(ex).__name__}: {ex}")
            saver.clear()
        state["saved"] += saver.saved - my_saved
        my_saved = saver.saved
        state["failed_rows"] += saver.failed - my_failed_rows
        my_failed_rows = saver.failed

    while True:
        try:
            path = files.popleft()
        except IndexError:
            if state_conn is not None:
                await state_conn.close()
            return
        pos = next(counter)
        try:
            # 行尾必须带换行：看门狗用 `cat w*` 收集在飞集合，缺换行会把
            # worker 的路径拼成一行，交集拉黑永远匹配不上（2026-10-25 实证）。
            with open(marker, "w", encoding="utf-8") as f:
                f.write(path + "\n")
        except OSError:
            pass
        try:
            logfn(f"{pos}/{total} importing {path}")
            pages_total = None
            for attempt in range(1, MAX_INLINE_ATTEMPTS + 1):
                pages_total = await import_one(path, head, ds_params, logfn)
                if not store_state or not pages_total:
                    break
                # 必须先 flush 再数覆盖页：缓冲里未落库的段落不算覆盖，
                # 旧版在 flush 前盖章 → 段落数 < batch 的文件整本被误判
                # 0/N partial（2026-10-23 实测修复）。
                await flush_stats()
                if state_conn is None:
                    state_conn = await get_conn()
                try:
                    sid = await state_conn.fetchval(
                        "SELECT id FROM file_metadata WHERE path = $1", path)
                    if not sid:
                        break
                    covered = await count_covered_pages(state_conn, sid)
                    if covered >= pages_total:
                        st = await record_import_state(
                            state_conn, sid, pages_total, covered)
                        state[st] = state.get(st, 0) + 1
                        logfn(f"[state] complete {covered}/{pages_total} 页: {path}")
                        break
                    if attempt >= MAX_INLINE_ATTEMPTS:
                        st = await record_import_state(
                            state_conn, sid, pages_total, covered)
                        state[st] = state.get(st, 0) + 1
                        logfn(f"[state] partial {covered}/{pages_total} 页"
                              f"（已就地重试 {attempt} 次，下轮续）: {path}")
                        break
                    logfn(f"[retry {attempt}/{MAX_INLINE_ATTEMPTS}] "
                          f"{covered}/{pages_total} 页未盖全，原地续跑: {path}")
                except Exception as ex:
                    logfn(f"[state] 盖章失败（不影响导入）: {path} :: "
                          f"{type(ex).__name__}: {ex}")
                    break
            state["files"] += 1
        except Exception as ex:
            state["failed"] += 1
            logfn(f"[skip] file failed, next file: {path} :: "
                  f"{type(ex).__name__}: {ex}")
            import traceback
            logfn("".join(traceback.format_exception(type(ex), ex,
                                                     ex.__traceback__)))
        finally:
            await flush_stats()
            try:
                os.unlink(marker)
            except OSError:
                pass


async def progress_watcher(interval, state, t0, count_secs=1800):
    conn = await get_conn()
    last_ins: int | None = None
    last_t = time.time()
    last_files_left: int | None = None
    last_count_t = 0.0
    try:
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            # 单轮失败不许杀死整个 watcher（2026-10-24 实证：上一版异常后
            # 静默死亡，[progress] 断更 9 小时）。失败记 traceback，下 tick 重连。
            try:
                if last_files_left is None or now - last_count_t >= count_secs:
                    last_files_left = await conn.fetchval(
                        """
                        SELECT count(*) FROM file_metadata f
                        WHERE lower(f.path) LIKE '%.pdf'
                          AND f.path NOT LIKE 'Tool Books/%'
                          AND f.path NOT LIKE '.trash/%'
                          AND f.path NOT LIKE '../%'
                          AND NOT EXISTS (SELECT 1 FROM paragraph p
                                          WHERE p.source_id = f.id)
                          AND NOT EXISTS (SELECT 1 FROM file_import_state s
                                          WHERE s.source_id = f.id
                                            AND s.status = 'complete')
                        """
                    )
                    last_count_t = now
                    left_str = str(last_files_left)
                else:
                    left_str = f"{last_files_left} (cached)"
                ins = await conn.fetchval(
                    "SELECT coalesce(sum(n_tup_ins), 0) FROM pg_stat_user_tables "
                    "WHERE relname LIKE 'paragraph%'"
                )
                # asyncpg 把 sum(bigint) 返回为 Decimal，统一转 int 再算速率。
                ins = int(ins or 0)
                rate = ""
                if last_ins is not None and now > last_t:
                    rate = f"  paras+{(ins - last_ins) / (now - last_t) * 60:,.0f}/min"
                last_ins, last_t = ins, now
                log.info(
                    "[progress] files w/o paragraph=%s%s  files_done=%s failed=%s "
                    "saved=%s  elapsed=%.1fmin",
                    left_str, rate, state["files"], state["failed"],
                    state["saved"], (time.time() - t0) / 60,
                )
            except Exception:
                log.exception(
                    "[progress] watcher iteration failed — 重建连接, 下个 tick 重试")
                try:
                    await conn.close()
                except Exception:
                    pass
                try:
                    conn = await get_conn()
                except Exception:
                    log.exception("[progress] watcher 重连失败, 下个 tick 再试")
    finally:
        await conn.close()


def acquire_lock() -> bool:
    if os.path.exists(LOCK_FILE):
        try:
            pid = int(open(LOCK_FILE).read().strip())
            os.kill(pid, 0)
            # 僵尸进程（state=Z）不算「有人在跑」——先杀外层看门狗时 worker
            # 常变僵尸（2026-10-20 实证），不识别会让锁永久卡死。
            try:
                with open(f"/proc/{pid}/stat") as f:
                    state = f.read().rsplit(")", 1)[1].split()[0]
                if state == "Z":
                    log.warning("lock holder %d is a zombie — taking over the lock", pid)
            except (OSError, IndexError):
                pass  # /proc 消失 = 进程已退，照常接管
            else:
                if state != "Z":
                    return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock() -> None:
    try:
        os.unlink(LOCK_FILE)
    except OSError:
        pass


async def run_round(keep: list[str], workers: int = 4, batch: int = 200,
                    logfn=None, store_state: bool = True,
                    progress_secs: int = 600,
                    progress_count_secs: int = 1800) -> dict:
    """跑一轮导入（给定文件清单）。返回 summary dict。

    崩溃隔离：本函数在调用方进程内跑；pymupdf 段错误会杀死整个进程——
    常驻 ImportWorkerPlugin 以子进程方式调用 CLI 壳获得隔离
    （见 plugins/importer/supervisor.py）。
    """
    if logfn is None:
        logfn = make_logfn()
    files = deque(keep)
    counter = itertools.count(1)
    total = len(keep)
    state = {"files": 0, "failed": 0, "saved": 0, "failed_rows": 0,
             "complete": 0, "partial": 0}

    t0 = time.time()

    async def status_writer():
        """每 30s 把轮内进度快照写盘，供常驻 worker 的 /import/ API 读取。"""
        while True:
            await asyncio.sleep(30)
            try:
                inflight = []
                if INFLIGHT_DIR.exists():
                    for m in sorted(INFLIGHT_DIR.glob("w*")):
                        try:
                            p = m.read_text(encoding="utf-8").strip()
                            if p:
                                inflight.append(p)
                        except OSError:
                            pass
                snap = {"ts": time.time(), "total": total, "inflight": inflight,
                        "elapsed_min": round((time.time() - t0) / 60, 1), **state}
                tmp = str(STATUS_FILE) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snap, f, ensure_ascii=False)
                os.replace(tmp, STATUS_FILE)
            except Exception:
                log.debug("[status] snapshot write failed", exc_info=True)

    status_task = asyncio.create_task(status_writer())
    watcher = asyncio.create_task(
        progress_watcher(progress_secs, state, t0, count_secs=progress_count_secs)
    )
    workers_tasks = [
        asyncio.create_task(worker(
            i, files, counter, total, batch, PDF_PARAMS, logfn,
            state, store_state=store_state,
        ))
        for i in range(max(1, workers))
    ]
    log.info("start: %d files, %d workers, batch=%d, engine=resident",
             total, max(1, workers), batch)
    try:
        await asyncio.gather(*workers_tasks)
    finally:
        watcher.cancel()
        status_task.cancel()
        try:
            os.unlink(STATUS_FILE)
        except OSError:
            pass
        try:
            logfn.close()
        except Exception:
            pass

    conn = await get_conn()
    try:
        post = await stats(conn)
    finally:
        await conn.close()

    summary = {
        "engine": "plugins.importer.import_engine (resident worker)",
        "workers": workers,
        "batch": batch,
        "list_size": total,
        "post": post,
        "files_processed": state["files"],
        "files_failed": state["failed"],
        "marked_complete": state["complete"],
        "marked_partial": state["partial"],
        "paragraphs_saved": state["saved"],
        "rows_dropped": state["failed_rows"],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    return summary


async def prepare_list(no_scan: bool = False, only: str = "",
                       scan_cache_secs: float = 3600,
                       stats_cache_secs: float = 3600) -> tuple[list[str], dict]:
    """登记新文件（磁盘 upsert）→ 建清单 → 应用 skip/defer 过滤。"""
    conn = await get_conn()
    disk = None
    try:
        pre = await stats(conn, cache_secs=stats_cache_secs)
        if only:
            only_paths = [ln.strip() for ln in open(only, encoding="utf-8")
                          if ln.strip() and not ln.startswith("#")]
            q = ("SELECT path FROM file_metadata WHERE path = ANY($1::text[]) "
                 "AND lower(path) LIKE '%.pdf'")
            known = {r["path"] for r in await conn.fetch(q, only_paths)}
            keep = [p for p in only_paths if p in known]
            list_meta = {"only_list": only, "only_requested": len(only_paths),
                         "only_known": len(keep)}
        else:
            if not no_scan:
                # 走盘是阻塞操作，但本函数只在子进程/启动期调用（不在 web 事件
                # 循环里），scan_disk_upsert 内部已走线程池。
                disk = await scan_disk_upsert(conn, cache_secs=scan_cache_secs)
            keep, list_meta = await build_file_list(
                conn, disk=disk, exclude_complete=True)
    finally:
        await conn.close()
    keep, filt = apply_skip_defer(keep)
    list_meta.update(filt)
    list_meta["pre_files_without_paragraph"] = pre["onedrive_pdfs_without_paragraph"]
    return keep, list_meta
