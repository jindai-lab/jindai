"""批量 PDF 导入 —— 直接构造管线模块，**不读 Task / TaskDBO**（2026-10-18 用户指令）。

与 `batch_import_fast.py` 的区别只有一处，但是关键的一处：stage 链不再从
`task_dbo.pipeline`（「导入 OneDrive PDF」任务）反序列化，而是**按模块代码
就地构造**：

    PDFDataSource(文本层抽取 + 扫描页 OCR + 页集续传)
      -> LanguageDetect()            语种检测
      -> Reparagraph()               连缀断行成段
      -> Condition(lang in zh/ja)    CJK 去空白（RepeatWhile x4 RegexReplace）
      -> Condition(content.strip())  西文连字符修复 -> WordCut(for_search=True) -> 批量落库

`WordCut` 产出 `paragraph.keywords`，`BatchSaveParagraph` 落库时同时
`Terms.store(keywords)` —— 即「建立检索词」。**DatasetFromField 不再出现在
链里**：数据集在 file_metadata 登记时已按文件夹规则挂好（`file_dataset`），
逐段再挂一次只会重复 upsert。PDFDataSource 内部对缺失的 (path, dataset)
仍会补挂（运行内缓存、只补缺），已有绑定不会重写。

其余行为与 fast 引擎一致：
- 清单/磁盘登记/工具书与磁盘缺失过滤/毒文件跳过 —— 复用
  `import_onedrive_pdfs.py` 的同一套函数（口径完全一致）。
- `BatchSaveParagraph`：每 --batch 段一次事务；bulk 失败降级逐段隔离毒段；
  段尾强制 flush。
- 文件级并发（--workers，每 worker 独立 stage 链与缓冲）。
- 页集续传：PDFDataSource(reimport_gaps=True)，中断只损失未 flush 的缓冲。
- 盖章 file_import_state（complete/partial），--no-state 关闭。

  python batch_operation/batch_import_direct.py --dry-run --no-scan
  python batch_operation/batch_import_direct.py --limit 8 --no-scan -w 4
  python batch_operation/batch_import_direct.py --selftest        # 只构造链，不碰库
  python batch_operation/batch_import_direct.py                   # 全量（含磁盘登记）
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
from collections import deque
from typing import Any

import regex as re  # \p{Other} 等 Unicode 属性需要 regex 模块

sys.path.insert(0, "/app")

import common as C  # noqa: E402

log = C.setup_logging("import_direct", "batch_operation/reports/import_direct.log")

# stage 日志落点与旧引擎一致（监控/排障 grep 的是这个文件）
STAGE_LOG = "batch_operation/reports/import_onedrive.json.log"
INFLIGHT_DIR = "batch_operation/reports/import_direct_inflight"
# 与 batch_import_fast 共用同一把锁：两个导入器绝不允许同时跑
LOCK_FILE = "batch_operation/reports/import_fast.lock"

# ---------------------------------------------------------------------------
# 管线参数（直接写在代码里，不查 task_dbo）
# ---------------------------------------------------------------------------
# dataset_name 只用于 PDFDataSource 内部 Dataset.get(...).id（传给跨页重排器，
# 本链 reparagraphize=False 不启用）。文件级数据集绑定早就有了，这里不重挂。
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

# 就地续跑上限（2026-10-23 用户指令：完全处理完才换下一个）：文件没盖满页时
# 原地重跑（reimport_gaps 只补缺页，skip_existed 跳过已有页，代价很小），
# 盖满记 complete，到顶仍缺页才记 partial 交下轮。防毒页卡死：可用环境变量调。
MAX_INLINE_ATTEMPTS = int(os.environ.get("IMPORT_MAX_ATTEMPTS", "3"))

# PDFDataSource 每个单文件实例自己会打 "1/1 importing <path>"；
# worker 已经打过带全局进度的 "N/M importing <path>"，把前者滤掉。
_FETCH_IMPORTING_RE = re.compile(r"\b1/1 importing ")


def make_logfn(logfile):
    """stage/数据源日志：stdout（看门狗→import_onedrive.log）+ json.log。"""
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
            paragraph.content = re.sub(r"\p{Other}", " ", paragraph.content or "")
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


# ---------------------------------------------------------------------------
# 直接按模块构造 stage 链（不读 task_dbo）
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 驱动：把一个段落沿 stage 链深度优先推到底（Task._run_chain 的直跑版）
# ---------------------------------------------------------------------------
async def run_chain(paragraph, stage) -> None:
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


async def worker(wid, files, counter, total, batch, ds_params, args, logfn,
                 state, store_state) -> None:
    """一个文件级 worker：独立 stage 链 + 独立缓冲，共享文件队列。"""
    import import_onedrive_pdfs as legacy

    _cls, pipeline, saver = build_chain(batch, logfn)
    head = pipeline.stages[0]
    assert saver is not None

    os.makedirs(INFLIGHT_DIR, exist_ok=True)
    marker = os.path.join(INFLIGHT_DIR, f"w{wid}")
    my_saved = 0
    my_failed_rows = 0
    state_conn: Any = None

    async def flush_stats() -> None:
        """落库缓冲 + 更新计数。盖章前必须先调用（否则数不到本文件的段落），
        finally 复用同一份逻辑，多次调用靠 nonlocal 增量计数、不会重复累加。"""
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
            with open(marker, "w", encoding="utf-8") as f:
                f.write(path)
        except OSError:
            pass
        try:
            logfn(f"{pos}/{total} importing {path}")
            pages_total = None
            for attempt in range(1, MAX_INLINE_ATTEMPTS + 1):
                pages_total = await import_one(path, head, ds_params, logfn)
                if not store_state or not pages_total:
                    break
                # 必须先 flush 再数覆盖页：缓冲(batch=200)里未落库的段落不算
                # 覆盖，旧版在 flush 前盖章 → 段落数 < batch 的文件整本被误判
                # 0/N partial（2026-10-23 实测修复）。
                await flush_stats()
                if state_conn is None:
                    state_conn = await C.get_conn()
                try:
                    sid = await state_conn.fetchval(
                        "SELECT id FROM file_metadata WHERE path = $1", path)
                    if not sid:
                        break
                    covered = await legacy.count_covered_pages(state_conn, sid)
                    if covered >= pages_total:
                        st = await legacy.record_import_state(
                            state_conn, sid, pages_total, covered)
                        state[st] = state.get(st, 0) + 1
                        logfn(f"[state] complete {covered}/{pages_total} 页: {path}")
                        break
                    if attempt >= MAX_INLINE_ATTEMPTS:
                        st = await legacy.record_import_state(
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
    conn = await C.get_conn()
    last_ins: int | None = None
    last_t = time.time()
    last_files_left: int | None = None
    last_count_t = 0.0
    try:
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            # 单轮失败不许杀死整个 watcher（2026-10-24 实证：上一版异常后
            # 静默死亡，[progress] 断更 9 小时、连接 idle 9h；导入本身不受
            # 影响，但看板失去心跳）。失败记 traceback，下个 tick 重连再试。
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
                # asyncpg 把 sum(bigint) 返回为 Decimal，Decimal / float 会
                # TypeError 杀掉本 tick（2026-10-24 实证），统一转 int 再算速率。
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
                    conn = await C.get_conn()
                except Exception:
                    log.exception("[progress] watcher 重连失败, 下个 tick 再试")
    finally:
        await conn.close()


def acquire_lock() -> bool:
    if os.path.exists(LOCK_FILE):
        try:
            pid = int(open(LOCK_FILE).read().strip())
            os.kill(pid, 0)
            # 僵尸进程（state=Z）：/proc 仍在、os.kill 不抛异常，但它已死，
            # 不算「有人在跑」——先杀外层 watchdog 时 worker 常变僵尸（2026-10-20 实证），
            # 不识别会让锁永久卡死，watchdog 空转 60 轮后 rounds exhausted。
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


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress-secs", type=int, default=600)
    ap.add_argument("--stats-cache-secs", type=int, default=3600)
    ap.add_argument("--scan-cache-secs", type=int, default=3600)
    ap.add_argument("--progress-count-secs", type=int, default=1800)
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--selftest", action="store_true",
                    help="只构造 stage 链并打印，不连库不写数据")
    ap.add_argument("--report", default="batch_operation/reports/import_direct.json")
    args = ap.parse_args()

    if args.selftest:
        _cls, pipeline, saver = build_chain(args.batch)
        names = [f"{type(s).__name__}" for s in pipeline.stages]
        print("chain:", " -> ".join(names))
        print("saver:", type(saver).__name__, "batch=", saver.batch)
        print("ds_params:", json.dumps(PDF_PARAMS, ensure_ascii=False))
        return 0

    # 清单/统计复用旧脚本（模块级日志重定向，避免双写）
    import import_onedrive_pdfs as legacy
    legacy.log = log

    conn = await C.get_conn()
    disk = None
    try:
        pre = await legacy.stats(conn, cache_secs=args.stats_cache_secs)
        if not args.no_scan:
            disk = await legacy.scan_disk_upsert(
                conn, cache_secs=args.scan_cache_secs)
        if args.only:
            only_paths = [ln.strip() for ln in open(args.only, encoding="utf-8")
                          if ln.strip() and not ln.startswith("#")]
            q = ("SELECT path FROM file_metadata WHERE path = ANY($1::text[]) "
                 "AND lower(path) LIKE '%.pdf'")
            known = {r["path"] for r in await conn.fetch(q, only_paths)}
            keep = [p for p in only_paths if p in known]
            list_meta = {"only_list": args.only, "only_requested": len(only_paths),
                         "only_known": len(keep)}
            log.info("[only] %s", json.dumps(list_meta, ensure_ascii=False))
        else:
            keep, list_meta = await legacy.build_file_list(
                conn, disk=disk, exclude_complete=not args.no_state)
    finally:
        await conn.close()

    # 毒文件黑名单（段错误看门狗维护）
    skip_file = "batch_operation/reports/import_skip.txt"
    if os.path.exists(skip_file):
        skip = {ln.strip() for ln in open(skip_file, encoding="utf-8")
                if ln.strip() and not ln.startswith("#")}
        before = len(keep)
        keep = [p for p in keep if p not in skip]
        if before != len(keep):
            log.warning("skip-file: 从清单剔除 %d 个毒文件", before - len(keep))

    # 暂缓清单（人工维护，与毒文件语义分离）：整目录/整批 OCR-bound
    # 文件让路时用。每行一个前缀（以 / 结尾）或精确路径；'#' 注释。
    # 从暂缓恢复 = 删掉对应行，下轮自动回到清单（import_state 不变）。
    defer_file = "batch_operation/reports/import_defer.txt"
    if os.path.exists(defer_file):
        prefixes, exact = [], set()
        for ln in open(defer_file, encoding="utf-8"):
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
                        before - len(keep), defer_file)

    log.info("pre: files without paragraph=%s, paragraph total=%s, import list=%s",
             pre["onedrive_pdfs_without_paragraph"],
             f"{pre['paragraph_total']:,}", list_meta)

    if args.limit > 0:
        keep = keep[:args.limit]
        log.info("--limit %d: processing first %d files", args.limit, len(keep))

    if args.dry_run:
        print(json.dumps({"pre": pre, "list": list_meta, "limit": len(keep)},
                         ensure_ascii=False, indent=2))
        return 0

    if not acquire_lock():
        log.error("another importer is running (lock %s)", LOCK_FILE)
        return 3

    logfn = make_logfn(STAGE_LOG)
    files = deque(keep)
    counter = itertools.count(1)
    total = len(keep)
    state = {"files": 0, "failed": 0, "saved": 0, "failed_rows": 0,
             "complete": 0, "partial": 0}

    t0 = time.time()
    watcher = asyncio.create_task(
        progress_watcher(args.progress_secs, state, t0,
                         count_secs=args.progress_count_secs)
    )
    workers = [
        asyncio.create_task(worker(
            i, files, counter, total, args.batch, PDF_PARAMS, args, logfn,
            state, store_state=not args.no_state
        ))
        for i in range(max(1, args.workers))
    ]
    log.info("start: %d files, %d workers, batch=%d, engine=direct-stages",
             total, max(1, args.workers), args.batch)
    try:
        await asyncio.gather(*workers)
    finally:
        watcher.cancel()
        logfn.close()

    conn = await C.get_conn()
    try:
        post = await legacy.stats(conn)
    finally:
        await conn.close()

    summary = {
        "engine": "batch_import_direct (stages built from modules, no Task/TaskDBO)",
        "workers": args.workers,
        "batch": args.batch,
        "list": list_meta,
        "pre": pre,
        "post": post,
        "imported_files": (
            pre["onedrive_pdfs_without_paragraph"]
            - post["onedrive_pdfs_without_paragraph"]
        ),
        "files_processed": state["files"],
        "files_failed": state["failed"],
        "marked_complete": state["complete"],
        "marked_partial": state["partial"],
        "paragraphs_saved": state["saved"],
        "rows_dropped": state["failed_rows"],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    C.dump_json(args.report, summary)
    log.info("summary: %s", json.dumps(summary, ensure_ascii=False, default=str))
    try:
        os.unlink(LOCK_FILE)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
