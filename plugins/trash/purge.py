"""常驻回收站清空 worker（built-in plugin: ``plugins/trash``，原
``jindai/trash_purge.py``）：与 import/embedding worker 同构的后台循环。

背景：删除文献记录时关联文件被搬进 ``<storage>/.trash/<批次>/``（见
``jindai/trash.py``），需要定期清空。原方案是 Hermes 侧 cronjob（每周一 06:00
宿主脚本 docker exec 进本容器跑 ``batch_operation/purge_trash.py``）；
2026-10-25 用户指令改为**容器内常驻**：定时任务做在容器里、随 app 启动自动拉起
（lifespan），Hermes cronjob 已删除、勿重建。

工作方式：每 60s 醒一次看是否到达 next due（**每周一 06:00**，容器 TZ=
Asia/Shanghai，启动时计算）；到期执行 ``purge_once()``（逐批整目录 rmtree，
安全阀 = 批次内仍被 ``file_metadata`` 引用的文件整批跳过），写
``reports/trash_purge.json`` 汇总并记日志，然后排下一周。文件系统清扫在
``asyncio.to_thread`` 里跑，不阻塞 web 事件循环；单轮异常捕获后退避 10min 重试。

开关：config.yaml ``trash_purge_enabled``（app 启动读一次，改后 ``docker
restart jindai``）。``purge_once()`` 同时被 ``batch_operation/purge_trash.py``
CLI 复用（dry-run / --apply / --min-age-hours 手动演练不变）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import Optional

from jindai.import_lists import REPORTS, get_conn
from jindai.plugin import WorkerPlugin

from .core import TRASH_DIRNAME, trash_root

log = logging.getLogger("plugins.trash.purge")

STATUS_FILE = REPORTS / "trash_purge.json"

# 清空时刻：每周一 06:00（容器本地时区 = Asia/Shanghai）
WEEKDAY = 0   # Monday
HOUR = 6
POLL_S = 60.0         # 到期检查间隔
RETRY_S = 600.0       # 单轮失败后的退避


def _batch_files(batch_dir: str) -> tuple[int, int]:
    """(文件数, 字节数) —— 不含目录本身。"""
    n = b = 0
    for dirpath, _d, fns in os.walk(batch_dir):
        for fn in fns:
            n += 1
            try:
                b += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return n, b


def _next_due(now: Optional[datetime] = None) -> datetime:
    """下一个「周一 06:00」（本地时区）。恰在周一 06:00 之前 → 排今天；已过 → 下周。"""
    now = now or datetime.now().astimezone()
    due = now.replace(hour=HOUR, minute=0, second=0, microsecond=0)
    due += timedelta(days=(WEEKDAY - now.weekday()) % 7)
    if due <= now:
        due += timedelta(days=7)
    return due


def _purge_sync(root: str, referenced: set[str], apply: bool,
                min_age_hours: float) -> dict:
    """同步清扫一轮（在 thread 里跑）。referenced = 仍被 file_metadata 引用的回收站路径。"""
    now = time.time()
    batches = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    stat = {"mode": "apply" if apply else "dry-run",
            "batches_seen": len(batches), "batches_purged": 0,
            "batches_skipped": 0, "files_purged": 0, "bytes_purged": 0,
            "skipped": []}

    for name in batches:
        bdir = os.path.join(root, name)
        try:
            age_h = (now - os.path.getmtime(bdir)) / 3600.0
        except OSError:
            continue
        if age_h < min_age_hours:
            stat["batches_skipped"] += 1
            continue
        # 安全阀：批次内任何文件仍被引用 → 整批保留
        hit = None
        for dirpath, _d, fns in os.walk(bdir):
            for fn in fns:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                if rel in referenced:
                    hit = rel
                    break
            if hit:
                break
        if hit:
            stat["batches_skipped"] += 1
            stat["skipped"].append({"batch": name, "reason": f"still referenced: {hit}"})
            log.warning("[guard] 批次 %s 仍被引用（%s），跳过", name, hit)
            continue

        n, b = _batch_files(bdir)
        if apply:
            try:
                shutil.rmtree(bdir)
            except Exception as ex:  # noqa: BLE001 - 一个批次失败不中断
                stat["batches_skipped"] += 1
                stat["skipped"].append({"batch": name,
                                        "reason": f"{type(ex).__name__}: {ex}"})
                log.warning("[purge] %s 删除失败: %s", name, ex)
                continue
        stat["batches_purged"] += 1
        stat["files_purged"] += n
        stat["bytes_purged"] += b
        log.info("[purge] %s  %d files  %.1f MB  age=%.1fh  %s",
                 name, n, b / 1048576, age_h, "removed" if apply else "would remove")

    # 清空后把空的回收站根留着（导入器跳过隐藏目录，无副作用）
    if apply and stat["batches_purged"] and not os.listdir(root):
        log.info("[purge] 回收站已清空: %s", root)
    return stat


async def purge_once(apply: bool = True, min_age_hours: float = 0.0) -> dict:
    """清空回收站一轮（async 入口：先查引用集合，文件系统清扫走线程）。

    安全阀：批次里**仍被 file_metadata 引用**的文件不删（正常不会发生——
    ``.trash/`` 路径不会被登记——但只要发生了就绝不制造孤儿记录）。
    汇总写 ``reports/trash_purge.json`` 并返回。
    """
    root = trash_root()
    if not os.path.isdir(root):
        stat = {"mode": "apply" if apply else "dry-run",
                "batches_seen": 0, "batches_purged": 0, "batches_skipped": 0,
                "files_purged": 0, "bytes_purged": 0, "skipped": [],
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "trash_root": root, "note": "trash root does not exist"}
        _write_status(stat)
        return stat

    # 安全阀：`.trash/` 路径绝不该被 file_metadata 引用；有就跳过对应批次。
    conn = await get_conn()
    try:
        referenced = {r["path"] for r in await conn.fetch(
            "SELECT path FROM file_metadata WHERE path LIKE $1",
            f"{TRASH_DIRNAME}/%")}
    finally:
        await conn.close()
    if referenced:
        log.warning("[guard] %d 条记录仍指向回收站，相关批次跳过: %s",
                    len(referenced), sorted(referenced)[:10])

    stat = await asyncio.to_thread(_purge_sync, root, referenced,
                                   apply, min_age_hours)
    stat["ts"] = datetime.now().astimezone().isoformat(timespec="seconds")
    stat["trash_root"] = root
    _write_status(stat)
    log.info("[done] %s", json.dumps(stat, ensure_ascii=False, default=str))
    return stat


def _write_status(stat: dict) -> None:
    try:
        tmp = str(STATUS_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stat, f, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, STATUS_FILE)
    except OSError:
        log.debug("[trash-purge] status write failed", exc_info=True)


class TrashPurgeWorkerPlugin(WorkerPlugin):
    """常驻清空 supervisor：每周一 06:00 清一次，随 app lifespan 启停。"""

    worker_name = "trash purge worker"
    enabled_key = "trash_purge_enabled"
    enabled_default = False

    def __init__(self, pmanager=None, **conf):
        super().__init__(pmanager, **conf)
        self.next_due: Optional[datetime] = None
        self.last_run: Optional[dict] = None

    def status(self) -> dict:
        return {
            "enabled": self.worker_enabled,
            "running": any(not t.done() for t in self.worker_tasks),
            "next_due": self.next_due.isoformat(timespec="seconds")
                        if self.next_due else None,
            "last_run": self.last_run,
        }

    def worker_details(self) -> dict:
        """GET /plugins/workers 额外状态（下次到期时间 + 上一轮汇总）。"""
        return {
            "next_due": self.next_due.isoformat(timespec="seconds")
                        if self.next_due else None,
            "last_run": self.last_run,
        }

    async def worker_loop(self) -> None:
        if not self.worker_enabled:
            log.info("Trash Purge Worker disabled "
                     "(config.trash_purge_enabled = false).")
            return
        log.info("Trash Purge Worker started.")
        self.next_due = _next_due()
        log.info("[trash-purge] next run at %s",
                 self.next_due.isoformat(timespec="seconds"))
        while True:
            try:
                await asyncio.sleep(POLL_S)
                if datetime.now().astimezone() < self.next_due:
                    continue
                stat = await purge_once(apply=True)
                self.last_run = stat
                self.next_due = _next_due()
                log.info("[trash-purge] next run at %s",
                         self.next_due.isoformat(timespec="seconds"))
            except asyncio.CancelledError:
                log.info("Trash Purge Worker is being cancelled.")
                return
            except Exception as ex:  # noqa: BLE001 - keep the worker alive
                log.exception("[trash-purge] iteration failed: %s: %s",
                              type(ex).__name__, ex)
                await asyncio.sleep(RETRY_S)

    def register_routes(self, router) -> None:
        """注册 /trash/ 状态路由（下次清空时间 + 上一轮汇总）。"""

        @router.get("/trash/", tags=["Trash"])
        async def trash_status():
            """回收站清空 worker 的当前状态。"""
            out = self.status()
            out["trash_root"] = trash_root()
            return out
