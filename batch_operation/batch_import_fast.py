"""批量 PDF 导入（脱离 jindai.task，2026-10-12 用户指令）。

与 import_onedrive_pdfs.py 的区别：**不经过 Task/优先队列/worker 池**，
直接驱动 pipeline 代码：

1. 清单/磁盘登记/工具书与磁盘缺失过滤/毒文件跳过 —— 复用
   import_onedrive_pdfs.py 的同一套函数（LEGACY），口径完全一致。
2. 从存好的任务「导入 OneDrive PDF」取 stage 定义：PDFDataSource 的参数
   直接用来**逐文件**构造 PDFDataSource 并调 fetch()；其余 stage
   （LanguageDetect/Reparagraph/Condition/DatasetFromField/Condition）
   按 worker 各实例化一条链，段落用内联深度优先 driver（本文件 run_chain，
   等价 Task._run_chain，去掉队列/tqdm/并发调度）逐级 flow 下去。
3. SaveParagraph（逐段 commit + Terms.store 各一次事务）替换为
   BatchSaveParagraph：内容清洗/关键词 600B 截断照旧，先攒
   --batch（默认 200）段一次事务批量落库，段尾强制 flush；
   批量失败自动降级逐段插入以隔离毒段落；Terms 每次 flush 批量 upsert。
4. **文件级并发**：--workers（默认 4）个协程各持一条独立 stage 链、
   从共享队列取文件；OCR 等待与其它文件的抽取/写库互相重叠
   （旧脚本单文件串行：渲染→OCR→逐段提交全程首尾相接）。
5. 崩溃自愈：每个 worker 在处理中文件写
   reports/import_fast_inflight/w<id> 标记；看门狗
   import_watchdog_fast.sh 对**连续两轮 crash 的在飞集合取交集**才拉黑
   （单轮 crash 的在飞文件可能有无辜者，反复崩在同一文件才判毒）。
   段落续传仍靠 PDFDataSource 的页集续传（有非空段落的页跳过），
   所以任何中断都只损失未 flush 的缓冲（≤ --batch 段/文件）。

日志口径与旧脚本兼容：stdout → import_onedrive.log（看门狗重定向），
stage 日志 → reports/import_onedrive.json.log（监控 grep 的还是它），
summary 含 "imported_files": N（看门狗判进展用）。

  python batch_operation/batch_import_fast.py --dry-run --no-scan
  python batch_operation/batch_import_fast.py --limit 8 --no-scan -w 4
  python batch_operation/batch_import_fast.py            # 全量（含磁盘登记）
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import os
import re
import sys
import time
from collections import deque

import regex as re  # \p{Other} 等 Unicode 属性需要 regex 模块

sys.path.insert(0, "/app")

import common as C  # noqa: E402

log = C.setup_logging("import_fast", "batch_operation/reports/import_fast.log")

# stage 日志落点与旧脚本一致（监控/排障 grep 的是这个文件）
STAGE_LOG = "batch_operation/reports/import_onedrive.json.log"
INFLIGHT_DIR = "batch_operation/reports/import_fast_inflight"
LOCK_FILE = "batch_operation/reports/import_fast.lock"

# PDFDataSource 每个单文件实例自己会打 "1/1 importing <path>"；
# worker 已经打过带全局进度的 "N/M importing <path>"，把前者滤掉，
# 免得看门狗/人眼看到两套位置号。
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


# --------------------------------------------------------------------------
# 批量版 SaveParagraph
# --------------------------------------------------------------------------
def _make_saver_class(batch_default: int):
    """构建 BatchSaveParagraph（SaveParagraph 的批量落库替身）。

    行为对齐 plugins/pipelines/basics.py::SaveParagraph：
    - content 过滤 \\p{Other}
    - keywords 600 UTF-8 字节截断 + 空值过滤
    区别：不逐段 commit，先缓冲 batch 段再一次事务 add_all 落库，
    Terms 每次 flush 一次批量 upsert；段尾由 worker 调 flush()。
    必须在 jindai 可 import 之后调用（依赖 PipelineStage 基类），
    返回的类注册进 Pipeline.ctx["BatchSaveParagraph"]。
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
            # —— 与 SaveParagraph.resolve 相同的清洗 ——
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
                for i, p in enumerate(buf):
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


# --------------------------------------------------------------------------
# pipeline driver（替代 jindai.task.Task：内联深度优先，无队列无 worker 池）
# --------------------------------------------------------------------------
async def run_chain(paragraph, stage) -> None:
    """把一个段落沿 stage 链深度优先推到底（Task._run_chain 的直跑版）。

    - flow() 的每个 (result, next_stage) 都内联递归（Task 把兄弟结果放进
      有界队列交给并发 worker；这里串行内联，语义等价于 concurrent=1）。
    - next_stage 为 None = 链尾（SaveParagraph 已落库），返回。
    - 异常向上抛给 per-file try/except，只损失当前文件。
    """
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


def substitute_save(obj, batch: int):
    """把 stage 定义树里的 SaveParagraph 换成 BatchSaveParagraph。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "SaveParagraph":
                out["BatchSaveParagraph"] = {"batch": batch}
            else:
                out[k] = substitute_save(v, batch)
        return out
    if isinstance(obj, list):
        return [substitute_save(v, batch) for v in obj]
    return obj


async def import_one(path: str, head, ds_params: dict, logfn) -> None:
    """单文件：直接构造 PDFDataSource 调 fetch()，段落走内联链。"""
    from plugins.datasources.pdfdatasource import PDFDataSource

    params = dict(ds_params)
    params["content"] = path
    ds = PDFDataSource(**params)
    # log 必须在 apply_params 之前挂上：apply_params 第一行就打
    # "File Paths: ..."，晚挂会让它落到 root logger（缺 logfn 前缀/去向）。
    ds.log = logfn
    ds.gctx = {}
    ds.apply_params(**params)
    async for para in ds.fetch():
        await run_chain(para, head)


async def worker(
    wid: int,
    files: deque,
    counter,
    total: int,
    chain_dicts: list,
    ds_params: dict,
    args,
    logfn,
    state: dict,
) -> None:
    """一个文件级 worker：独立 stage 链 + 独立缓冲，共享文件队列。"""
    from jindai.pipeline import Pipeline

    pipeline = Pipeline(copy.deepcopy(chain_dicts), log=logfn, verbose=args.verbose)
    pipeline.gctx = {}
    head = pipeline.stages[0]
    saver = find_saver(pipeline.stages)
    assert saver is not None, "BatchSaveParagraph not found in stage chain"

    os.makedirs(INFLIGHT_DIR, exist_ok=True)
    marker = os.path.join(INFLIGHT_DIR, f"w{wid}")
    my_saved = 0
    my_failed_rows = 0

    while True:
        try:
            path = files.popleft()
        except IndexError:
            return
        pos = next(counter)
        try:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(path)
        except OSError:
            pass
        try:
            logfn(f"{pos}/{total} importing {path}")
            await import_one(path, head, ds_params, logfn)
            state["files"] += 1
        except Exception as ex:
            state["failed"] += 1
            logfn(f"[skip] file failed, next file: {path} :: "
                  f"{type(ex).__name__}: {ex}")
            import traceback
            logfn("".join(traceback.format_exception(type(ex), ex,
                                                     ex.__traceback__)))
        finally:
            try:
                await saver.flush()
            except Exception as ex:
                logfn(f"[error] flush failed, dropping buffer "
                      f"({saver.saved} saved so far): {type(ex).__name__}: {ex}")
                saver.clear()
            # saver 是每 worker 私有：把增量累加进共享 state
            state["saved"] += saver.saved - my_saved
            my_saved = saver.saved
            state["failed_rows"] += saver.failed - my_failed_rows
            my_failed_rows = saver.failed
            try:
                os.unlink(marker)
            except OSError:
                pass


async def progress_watcher(interval: int, state: dict, t0: float):
    """与旧脚本同口径的 [progress] 行；段落增速用 pg_stat 增量（免全表 count）。"""
    conn = await C.get_conn()
    last_ins: int | None = None
    last_t = time.time()
    try:
        while True:
            await asyncio.sleep(interval)
            files_left = await conn.fetchval(
                """
                SELECT count(*) FROM file_metadata f
                WHERE lower(f.path) LIKE '%.pdf'
                  AND f.path NOT LIKE 'Tool Books/%'
                  AND f.path NOT LIKE '.trash/%'
                  AND f.path NOT LIKE '../%'
                  AND NOT EXISTS (SELECT 1 FROM paragraph p
                                  WHERE p.source_id = f.id)
                """
            )
            ins = await conn.fetchval(
                "SELECT coalesce(sum(n_tup_ins), 0) FROM pg_stat_user_tables "
                "WHERE relname LIKE 'paragraph%'"
            )
            now = time.time()
            rate = ""
            if last_ins is not None and now > last_t:
                rate = f"  paras+{(ins - last_ins) / (now - last_t) * 60:,.0f}/min"
            last_ins, last_t = ins, now
            log.info(
                "[progress] files w/o paragraph=%s%s  files_done=%s failed=%s "
                "saved=%s  elapsed=%.1fmin",
                files_left, rate, state["files"], state["failed"],
                state["saved"], (time.time() - t0) / 60,
            )
    finally:
        await conn.close()


def acquire_lock() -> bool:
    """防双开：锁文件记 pid，活着就拒绝启动。"""
    if os.path.exists(LOCK_FILE):
        try:
            pid = int(open(LOCK_FILE).read().strip())
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # 陈旧锁
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="导入 OneDrive PDF")
    ap.add_argument("-w", "--workers", type=int, default=4,
                    help="并发文件 worker 数（默认 4）")
    ap.add_argument("--batch", type=int, default=200,
                    help="批量落库的段落数（默认 200）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-scan", action="store_true",
                    help="跳过磁盘→file_metadata 登记扫描")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress-secs", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理清单前 N 个文件（测试用）")
    ap.add_argument("--report", default="batch_operation/reports/import_fast.json")
    args = ap.parse_args()

    if not acquire_lock():
        log.error("another batch_import_fast is running (lock %s)", LOCK_FILE)
        return 3

    # 清单/统计复用旧脚本（模块级日志重定向到本进程 logger，避免双写）
    import import_onedrive_pdfs as legacy
    legacy.log = log

    conn = await C.get_conn()
    try:
        pre = await legacy.stats(conn)
        if not args.no_scan:
            await legacy.scan_disk_upsert(conn)
        keep, list_meta = await legacy.build_file_list(conn)
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

    log.info("pre: files without paragraph=%s, paragraph total=%s, import list=%s",
             pre["onedrive_pdfs_without_paragraph"],
             f"{pre['paragraph_total']:,}", list_meta)

    if args.limit > 0:
        keep = keep[:args.limit]
        log.info("--limit %d: processing first %d files", args.limit, len(keep))

    if args.dry_run:
        print(json.dumps({"pre": pre, "list": list_meta, "limit": len(keep)},
                         ensure_ascii=False, indent=2))
        os.unlink(LOCK_FILE)
        return 0

    # --- 载入任务 stage 定义 + 插件注册（同 `python -m jindai task`） ---
    from jindai.pipeline import Pipeline
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins

    dbo = await TaskDBO.get(args.task)
    if not dbo:
        log.error("task %r not found", args.task)
        os.unlink(LOCK_FILE)
        return 2
    _init_plugins()

    BatchSaveParagraph = _make_saver_class(args.batch)
    Pipeline.ctx["BatchSaveParagraph"] = BatchSaveParagraph

    ds_params = None
    chain_dicts = []
    for stage in dbo.pipeline:
        name = next(iter(stage))
        if "PDFDataSource" in name:
            ds_params = dict(stage[name])
            continue
        if "CalibreDataSource" in name:
            log.info("[clone] dropping CalibreDataSource stage (Calibre 路由已废止)")
            continue
        chain_dicts.append(substitute_save(stage, args.batch))
    if ds_params is None:
        log.error("no PDFDataSource stage in task %r", args.task)
        os.unlink(LOCK_FILE)
        return 2
    ds_params.pop("content", None)  # 逐文件注入

    # 自检：每个 worker 的链里必须能找到 BatchSaveParagraph
    from jindai.pipeline import Pipeline as _Pipe
    _probe = _Pipe(copy.deepcopy(chain_dicts), log=lambda *x: None)
    assert find_saver(_probe.stages) is not None, \
        "BatchSaveParagraph substitution failed"

    logfn = make_logfn(STAGE_LOG)

    files = deque(keep)
    counter = itertools.count(1)
    total = len(keep)
    state = {"files": 0, "failed": 0, "saved": 0, "failed_rows": 0}

    t0 = time.time()
    watcher = asyncio.create_task(
        progress_watcher(args.progress_secs, state, t0)
    )
    workers = [
        asyncio.create_task(worker(
            i, files, counter, total, chain_dicts, ds_params, args, logfn, state
        ))
        for i in range(max(1, args.workers))
    ]
    log.info("start: %d files, %d workers, batch=%d, task=%s",
             total, max(1, args.workers), args.batch, args.task)
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
        "task": args.task,
        "engine": "batch_import_fast (no jindai.task)",
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
