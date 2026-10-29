"""常驻 PDF 导入 worker（built-in plugin: ``plugins/importer``，原
``jindai/import_worker.py``）：与 embedding worker 同构的后台循环。

职责（吸收原 import_watchdog_fast.sh + auto_ingest_scoped 的核心逻辑）：
1. 监测文件夹变动：每轮引擎自带磁盘 upsert（TTL 缓存），新 PDF 自动登记；
2. 有待导入文件时，以**子进程**方式拉起导入引擎（batch_operation/
   batch_import_direct.py 进程入口 → plugins.importer.import_cli，进程名保持
   兼容监控 grep）；
3. 崩溃归因：子进程非 0 退出 → 读在飞标记集合，与上一崩溃轮取交集 →
   交集写入毒文件黑名单 import_skip.txt → 重启。段错误隔离必须在子进程：
   pymupdf 崩溃会杀死整个进程，绝不能放进 web 主进程里跑；
4. 对外暴露 /import/ 状态 API（tasks 页面显示当前导入状态）。

开关：config.yaml `import_worker_enabled`（app 启动时读一次，改后需重启容器）。
与手工导入共存：引擎自身持 import_fast.lock，worker 拉起前检查锁，
锁被外部占用时本轮让路（不会双跑）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

from jindai.config import config
from jindai import diskutils
from jindai.plugin import WorkerPlugin

from .import_lists import DEFER_FILE, REPORTS, SKIP_FILE, pending_counts

log = logging.getLogger("jindai.import_worker")   # 名字保留：运维日志 grep 依赖

INFLIGHT_DIR = REPORTS / "import_direct_inflight"
LOCK_FILE = REPORTS / "import_fast.lock"
STATUS_FILE = REPORTS / "import_status.json"      # 引擎轮内快照（import_engine 写）
WORKER_STATUS_FILE = REPORTS / "import_worker_status.json"  # 本模块写的 supervisor 状态
ENGINE_LOG = REPORTS / "import_onedrive.log"      # 子进程 stdout 追加落点
LAST_SUMMARY = REPORTS / "import_direct.json"     # 引擎每轮 summary

# 引擎 CLI：进程入口只做 `from plugins.importer.import_cli import main`，
# 保留它是因为进程名含 batch_import_direct——gpu_cpu_monitor.py /
# auto_ingest_scoped.py 的 ps grep 依赖这个进程名，改路径前先改监控。
ENGINE_CMD = [sys.executable, "/app/batch_operation/batch_import_direct.py"]

# 连续空轮（无进展干净退出）指数退避上限；有进展则下一轮立即开始。
IDLE_BASE_SLEEP = 30.0
IDLE_MAX_SLEEP = 900.0
# 崩溃轮重启前的固定等待（段错误后立刻重启大概率再崩）
CRASH_SLEEP = 5.0
# 连续崩溃 N 轮后拉长退避（防毒文件交集没命中时的无限快循环）
CRASH_BACKOFF_AFTER = 6
CRASH_MAX_SLEEP = 600.0


def _read_json(path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _lock_held_by_other() -> bool:
    """import_fast.lock 被活着的外部进程持有（手工导入/旧看门狗）。

    除 os.kill(pid,0) 存活检测外必须核对 cmdline 确是导入引擎：容器重启后
    PID 会被复用（2026-10-26 实证：锁里残留旧引擎 PID 144 被新的
    multiprocessing 子进程复用，worker 误判「外部导入器在跑」而永远让路）。
    """
    try:
        pid = int(open(LOCK_FILE).read().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    # 僵尸不算占用（与引擎 acquire_lock 同口径）
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        if state == "Z":
            return False
    except (OSError, IndexError):
        return False
    # PID 复用防线：cmdline 必须还是导入引擎本体
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        return "batch_import_direct" in cmd or "import_cli" in cmd
    except OSError:
        return False


def _collect_inflight() -> set[str]:
    """读引擎在飞标记（每行一个路径）。崩溃时 worker finally 没跑，标记留存。"""
    out: set[str] = set()
    try:
        for m in INFLIGHT_DIR.glob("w*"):
            try:
                p = m.read_text(encoding="utf-8").strip()
                if p:
                    out.add(p)
            except OSError:
                pass
    except OSError:
        pass
    return out


def _blacklist(paths: set[str]) -> int:
    """把确认的毒文件追加进 skip 名单（幂等，带换行）。返回新增数。"""
    if not paths:
        return 0
    existing = set()
    if SKIP_FILE.exists():
        existing = {ln.strip() for ln in open(SKIP_FILE, encoding="utf-8")
                    if ln.strip() and not ln.startswith("#")}
    new = sorted(paths - existing)
    if new:
        with open(SKIP_FILE, "a", encoding="utf-8") as f:
            for p in new:
                f.write(p + "\n")
    return len(new)


class ImportWorkerPlugin(WorkerPlugin):
    """常驻导入 supervisor：拉起/监督引擎子进程，维护状态供 /import/ API。"""

    worker_name = "import worker"
    enabled_key = "import_worker_enabled"
    enabled_default = False

    def __init__(self, pmanager=None, **conf):
        super().__init__(pmanager, **conf)
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.round_no = 0
        self.started_at: Optional[float] = None
        self.round_started_at: Optional[float] = None
        self.last_exit: Optional[dict] = None
        self.idle_streak = 0
        self.crash_streak = 0
        self.prev_crash_inflight: set[str] = set()
        self.blacklisted_total = 0
        self.last_round_summary: Optional[dict] = None
        self.paused_external = False   # 锁被外部占用，worker 让路

    def worker_details(self) -> dict:
        """GET /plugins/workers 额外状态（supervisor 视角 + 引擎轮内快照）。"""
        try:
            st = self.status()
        except Exception as ex:  # noqa: BLE001 - status must not break the API
            return {"error": f"{type(ex).__name__}: {ex}"}
        return {"supervisor": st.get("supervisor"), "engine": st.get("engine")}

    # ------------------------------------------------------------------ 状态
    def status(self) -> dict:
        """supervisor 视角状态 + 引擎轮内快照 + 清单口径 pending/partial。"""
        snap = _read_json(STATUS_FILE)
        worker = {
            "enabled": bool(getattr(config, "import_worker_enabled", False)),
            "running": self.proc is not None and self.proc.returncode is None,
            "round": self.round_no,
            "idle_streak": self.idle_streak,
            "crash_streak": self.crash_streak,
            "blacklisted_total": self.blacklisted_total,
            "last_exit": self.last_exit,
            "last_round_summary": self.last_round_summary,
            "external_importer_holds_lock": _lock_held_by_other(),
            "defer_file_exists": DEFER_FILE.exists(),
            "skip_file": str(SKIP_FILE),
        }
        if self.round_started_at:
            worker["round_elapsed_min"] = round(
                (time.time() - self.round_started_at) / 60, 1)
        out = {"supervisor": worker, "engine": snap}
        # pending/partial 秒回（import_state 部分索引）；API 里带上但不阻塞太久
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                out["pending"] = self._pending_cache
        except Exception:
            pass
        return out

    _pending_cache: Optional[dict] = None

    _embed_queued_cache: Optional[tuple[float, int]] = None  # (ts, count)

    async def _refresh_pending(self) -> None:
        from .import_lists import get_conn
        conn = await get_conn()
        try:
            self._pending_cache = await pending_counts(conn)
        except Exception:
            log.exception("[worker] pending_counts failed")
        finally:
            await conn.close()
        # 嵌入队列深度（87M 行 count(*) 秒级但不便宜，60s TTL 缓存）
        try:
            now = time.time()
            c = self._embed_queued_cache
            if c is None or now - c[0] > 60:
                conn = await get_conn()
                try:
                    n = int(await conn.fetchval(
                        "SELECT count(*) FROM embedding_pending_queue") or 0)
                finally:
                    await conn.close()
                self._embed_queued_cache = (now, n)
        except Exception:
            log.exception("[worker] embed queue count failed")

    # ------------------------------------------------------------------ 主循环
    async def worker_loop(self):
        """后台循环：空闲探测 → 有待导入则拉起引擎子进程 → 监督其退出。"""
        if not self.worker_enabled:
            log.info("Import Worker disabled (config.import_worker_enabled = false).")
            return
        log.info("Import Worker started.")
        # 磁盘守卫（2026-10-26）：只在「轮与轮之间」拦，不在轮中杀引擎
        # （半截页靠 page-set resume 能补，但没必要付这个代价）。阈值比
        # embedding worker 的（85%）更高——段落增速远小于向量，这里是
        # 防撞墙的最后闸门。
        guard_path = str(getattr(config, "disk_guard_path", "/app"))
        guard_pause = float(getattr(config, "import_disk_guard_pause_pct", 92))
        guard_resume = min(float(getattr(config, "import_disk_guard_resume_pct", 88)),
                           guard_pause)
        disk_paused = False
        disk_cycles = 0
        while True:
            pct = diskutils.used_pct(guard_path)
            if pct is not None:
                if disk_paused:
                    if pct <= guard_resume:
                        log.info("[worker] disk guard: usage %.1f%% <= %.1f%% "
                                 "— resuming imports", pct, guard_resume)
                        disk_paused = False
                        disk_cycles = 0
                    else:
                        disk_cycles += 1
                        if disk_cycles == 1 or disk_cycles % 10 == 0:
                            log.warning("[worker] disk guard: usage %.1f%% > "
                                        "%.1f%% — imports still paused "
                                        "(cycle %d)", pct, guard_resume,
                                        disk_cycles)
                        await self._write_worker_status()
                        await asyncio.sleep(120)
                        continue
                elif pct >= guard_pause:
                    log.warning("[worker] disk guard: usage %.1f%% >= %.1f%% "
                                "— pausing new import rounds (auto-resume at "
                                "<= %.1f%%)", pct, guard_pause, guard_resume)
                    disk_paused = True
                    disk_cycles = 0
                    await self._write_worker_status()
                    await asyncio.sleep(120)
                    continue
            try:
                await self._refresh_pending()
                pending = (self._pending_cache or {}).get("total", 0) or 0

                if self.proc is not None and self.proc.returncode is None:
                    # 引擎在跑：等它退出（不轮询数据库，引擎自己管进度）
                    await self._wait_round()
                    continue

                if self.paused_external or _lock_held_by_other():
                    if not self.paused_external:
                        log.info("[worker] lock held by external importer — standing by")
                    self.paused_external = True
                    await asyncio.sleep(30)
                    # 外部导入器退出后恢复
                    if not _lock_held_by_other():
                        self.paused_external = False
                        log.info("[worker] external importer gone — resuming")
                    continue

                if pending <= 0:
                    self.idle_streak += 1
                    sleep_s = min(IDLE_BASE_SLEEP * (2 ** self.idle_streak),
                                  IDLE_MAX_SLEEP)
                    if self.idle_streak <= 3 or self.idle_streak % 6 == 0:
                        log.info("[worker] nothing pending (streak %d), "
                                 "sleeping %.0fs", self.idle_streak, sleep_s)
                    await self._write_worker_status()
                    await asyncio.sleep(sleep_s)
                    continue

                await self._start_round()
            except asyncio.CancelledError:
                log.info("Import Worker is being cancelled.")
                await self._terminate_proc()
                return
            except Exception as ex:  # noqa: BLE001 - keep the worker alive
                log.exception("Import worker iteration failed: %s: %s",
                              type(ex).__name__, ex)
                await asyncio.sleep(60)

    async def _start_round(self) -> None:
        """拉起一轮引擎子进程。"""
        # 崩溃退避：连续崩溃不命中黑名单时拉长间隔
        if self.crash_streak >= CRASH_BACKOFF_AFTER:
            sleep_s = min(CRASH_SLEEP * (2 ** (self.crash_streak -
                                               CRASH_BACKOFF_AFTER + 1)),
                          CRASH_MAX_SLEEP)
            log.warning("[worker] crash streak %d — backoff %.0fs before restart",
                        self.crash_streak, sleep_s)
            await asyncio.sleep(sleep_s)

        self.round_no += 1
        self.round_started_at = time.time()
        # 无进展轮退避（设计口径=指数退避上限 900s）。此前只有 pending==0 分支
        # 会睡：pending>0 但轮轮无进展（如死路径撑大队列/ImportError 全文件秒败）
        # 会 ~2.5 分钟空转一轮（2026-10-26 实证 28 连击）。
        if self.idle_streak:
            sleep_s = min(IDLE_BASE_SLEEP * (2 ** min(self.idle_streak, 5)),
                          IDLE_MAX_SLEEP)
            log.info("[worker] no-progress backoff %.0fs (streak %d)",
                     sleep_s, self.idle_streak)
            await asyncio.sleep(sleep_s)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        logf = open(ENGINE_LOG, "a", encoding="utf-8")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *ENGINE_CMD,
                cwd="/app",
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,   # 段错误不牵连 web 进程
            )
        finally:
            logf.close()
        log.info("[worker] round %d started (pid %s, pending %s)",
                 self.round_no, self.proc.pid,
                 (self._pending_cache or {}).get("total"))
        await self._write_worker_status()

    async def _wait_round(self) -> None:
        """等待引擎子进程退出并做崩溃归因/退避决策。"""
        assert self.proc is not None
        rc = await self.proc.wait()
        elapsed = round((time.time() - (self.round_started_at or time.time())) / 60, 1)
        summary = _read_json(LAST_SUMMARY)
        progressed = bool(summary and summary.get("files_processed", 0) > 0)

        if rc == 0:
            self.last_exit = {"rc": 0, "ts": time.time(),
                              "elapsed_min": elapsed,
                              "files_processed": (summary or {}).get("files_processed"),
                              "paragraphs_saved": (summary or {}).get("paragraphs_saved")}
            self.last_round_summary = summary
            if progressed:
                if self.idle_streak:
                    log.info("[worker] round %d done with progress after %d idle "
                             "cycle(s) — next round immediately",
                             self.round_no, self.idle_streak)
                self.idle_streak = 0
                self.crash_streak = 0
            else:
                # 干净退出但无进展：清单已尽（defer/毒文件挡着或全部完成）
                self.idle_streak += 1
                log.info("[worker] round %d finished, no progress (streak %d)",
                         self.round_no, self.idle_streak)
        else:
            # 崩溃归因：在飞集合与上一崩溃轮取交集 → 拉黑（与旧看门狗同法，
            # 但标记文件已带换行，交集是真实文件集合而非拼接垃圾行）
            inflight = _collect_inflight()
            added = 0
            if self.prev_crash_inflight:
                inter = inflight & self.prev_crash_inflight
                added = _blacklist(inter)
                self.blacklisted_total += added
                if added:
                    log.warning("[worker] round %d crashed (rc=%s); blacklisted "
                                "%d files seen in flight in consecutive crashes",
                                self.round_no, rc, added)
            self.prev_crash_inflight = inflight
            self.crash_streak += 1
            self.last_exit = {"rc": rc, "ts": time.time(), "elapsed_min": elapsed,
                              "inflight": sorted(inflight),
                              "blacklisted_now": added}
            log.warning("[worker] round %d crashed rc=%s (streak %d, in-flight %d)",
                        self.round_no, rc, self.crash_streak, len(inflight))
            await asyncio.sleep(CRASH_SLEEP)

        self.proc = None
        self.round_started_at = None
        await self._write_worker_status()

    async def _terminate_proc(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=15)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass

    async def _write_worker_status(self) -> None:
        try:
            tmp = str(WORKER_STATUS_FILE) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.status(), f, ensure_ascii=False, default=str)
            os.replace(tmp, WORKER_STATUS_FILE)
        except OSError:
            log.debug("[worker] status write failed", exc_info=True)

    def register_routes(self, router):
        """注册 /import/ 状态路由（tasks 页面读它显示导入状态）。"""

        from fastapi import Depends
        from jindai.models import get_db

        @router.get("/import/", tags=["Import"])
        async def import_status(session=Depends(get_db)):
            """当前导入状态：supervisor（常驻 worker）+ engine（轮内快照）。"""
            await self._refresh_pending()
            out = self.status()
            out["pending"] = self._pending_cache
            if self._embed_queued_cache:
                out["embed_queued"] = self._embed_queued_cache[1]
            return out

        @router.post("/import/pause", tags=["Import"])
        async def import_pause():
            """暂停/恢复常驻导入（只影响 worker 拉新轮，不杀在跑的轮）。"""
            self.paused_external = not self.paused_external
            return {"paused": self.paused_external}

        return router
