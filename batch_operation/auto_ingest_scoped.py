#!/usr/bin/env python3
"""OneDrive/Zotero 增量自动入库（2026-10-14 用户指令：jindai 导入监控 cron 每小时拉起）。

范围＝容器内 /storage/OneDrive 与 /storage/Zotero 两个子树（隐藏目录剪枝；
Tool Books 不在范围内）。流程：
  1. 走盘收集范围内 *.pdf（CIFS 全 walk 实测 ~4 min，故由 gpu_cpu_monitor.py
     后台拉起，不阻塞整点采样）；
  2. 与 file_metadata（同前缀）比对，缺行 upsert（INSERT … ON CONFLICT (path)
     DO NOTHING；item_type：Zotero→journalArticle，其余→book，与全库
     scan_disk_upsert 同口径）；
  3. 有新登记 → fill_nontitle_rows.py --apply 回填 title/authors/item_type；
  4. 有新登记且当前无导入进程在跑（batch_import_direct / batch_import_fast /
     import_onedrive_pdfs / import_watchdog_fast）→ 后台拉起
     batch_import_direct.py --no-scan --only batch_operation/reports/auto_ingest_new.txt；
     导入进程在跑时不重复拉起——其下一轮清单自会带上新登记的行。
  5. 结果 JSON 写 reports/auto_ingest_state.json（含 ts，监控脚本读它出报告行）。

锁由宿主机侧 gpu_cpu_monitor.py 持有（auto_ingest_state.json.lock，>40min 视为陈旧）。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/app")
import common as C  # noqa: E402

log = C.setup_logging("auto_ingest", "batch_operation/reports/auto_ingest.log")

SCOPE = ("OneDrive", "Zotero")
SCAN_ITEM_TYPE = {"Zotero": "journalArticle"}
STATE = "batch_operation/reports/auto_ingest_state.json"
NEWLIST = "batch_operation/reports/auto_ingest_new.txt"
IMPORT_LOG = "batch_operation/reports/auto_ingest_import.log"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=60).stdout
    except Exception:
        return ""


async def main() -> int:
    t0 = time.time()
    from jindai.storage import storage
    root = storage.safe_join("OneDrive")[: -len("OneDrive")]

    # 1. 走盘（隐藏目录剪枝＝.trash 等不被登记）
    disk: dict[str, int] = {}
    for top in SCOPE:
        base = os.path.join(root, top)
        if not os.path.isdir(base):
            log.warning("[scan] %s missing on disk, skipped", base)
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if not d.startswith(".")]
            for fn in fns:
                if not fn.lower().endswith(".pdf"):
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                try:
                    disk[rel] = os.path.getsize(full)
                except OSError:
                    disk[rel] = 0

    # 2. 比对 + upsert
    conn = await C.get_conn()
    try:
        existing = {r["path"] for r in await conn.fetch(
            "SELECT path FROM file_metadata "
            "WHERE (path LIKE 'OneDrive/%' OR path LIKE 'Zotero/%') "
            "AND lower(path) LIKE '%.pdf'")}
        new = sorted(p for p in disk if p not in existing)
        dedup_skipped = 0
        if new:
            # 查重护栏（与引擎全盘登记同口径，2026-10-26）：与库里已有 PDF 行
            # 同文件名（NFC 小写）= 同一文件的另一份拷贝 → 不登记，只留档。
            from jindai.import_lists import dedup_new_paths
            keep, skipped = await dedup_new_paths(conn, new)
            dedup_skipped = len(skipped)
            new = sorted(keep)
        if new:
            await conn.executemany(
                "INSERT INTO file_metadata (path, extension, size_bytes, item_type, import_state) "
                "VALUES ($1, 'pdf', $2, $3, 'pending') ON CONFLICT (path) DO NOTHING",
                [(p, disk[p], SCAN_ITEM_TYPE.get(p.split("/")[0], "book"))
                 for p in new])
    finally:
        await conn.close()
    log.info("[scan] scoped pdfs=%d  existing rows=%d  upserted=%d  dedup-skipped=%d",
             len(disk), len(existing), len(new), dedup_skipped)

    # 3. 新登记行回填 title（站规：任何批量登记后必跑；幂等、只碰空 title 行）
    if new:
        try:
            r = subprocess.run(
                ["/app/.venv/bin/python", "/app/batch_operation/fill_nontitle_rows.py",
                 "--apply"],
                cwd="/app", capture_output=True, text=True, timeout=1800)
            log.info("[title] fill_nontitle rc=%s %s", r.returncode,
                     (r.stdout or "")[-300:])
        except Exception as ex:
            log.warning("[title] fill_nontitle failed: %s", ex)

    # 4. 触发导入（仅在空闲时；在跑则等它下一轮带上）
    # ps auxww：容器里 ps aux 会把命令截断到 80 列（".venv/bin/py"），
    # grep 脚本名必落空（2026-10-14 实测误报空闲）——必须 ww 不截断。
    procs = sh("ps auxww | grep -E '[b]atch_import_direct|[b]atch_import_fast|"
               "[i]mport_onedrive_pdfs|[i]mport_watchdog_fast'").strip()
    importer_running = bool(procs)

    # 重试上一轮已登记但还没出段落的（全量导入当轮清单是轮初建的，
    # 当轮中新登记的行要等下一轮；若全量导入恰好收尾退出，这里兜底）。
    prev_pending: list[str] = []
    if os.path.exists(NEWLIST):
        prev = [ln.strip() for ln in open(NEWLIST, encoding="utf-8")
                if ln.strip() and not ln.startswith("#")]
        if prev:
            conn2 = await C.get_conn()
            try:
                rows = await conn2.fetch(
                    "SELECT f.path FROM file_metadata f "
                    "WHERE f.path = ANY($1::text[]) "
                    "AND NOT EXISTS (SELECT 1 FROM paragraph p "
                                    "WHERE p.source_id = f.id)", prev)
            finally:
                await conn2.close()
            prev_pending = [r["path"] for r in rows]

    todo = sorted(set(new) | set(prev_pending))
    launched = False
    if todo and not importer_running:
        with open(NEWLIST, "w", encoding="utf-8") as f:
            f.write("\n".join(todo) + "\n")
        subprocess.Popen(
            "/app/.venv/bin/python batch_operation/batch_import_direct.py --no-scan "
            "--only batch_operation/reports/auto_ingest_new.txt --progress-secs 600 "
            ">> batch_operation/reports/auto_ingest_import.log 2>&1",
            shell=True, cwd="/app", start_new_session=True)
        launched = True
        log.info("[import] launched scoped import for %d files (+%d retry)",
                 len(new), len(todo) - len(new))
    elif todo:
        log.info("[import] importer busy, next full round will pick up %d files",
                 len(todo))
    elif os.path.exists(NEWLIST):
        # 全部导入完毕，清掉清单，避免下轮空转重试
        os.remove(NEWLIST)

    out = {
        "ts": time.strftime("%m-%d %H:%M"),
        "scope": "OneDrive+Zotero",
        "disk_pdfs_scoped": len(disk),
        "rows_scoped_before": len(existing),
        "new_registered": len(new),
        "new_sample": new[:5],
        "pending_retry": len(prev_pending),
        "todo": len(todo),
        "importer_running": importer_running,
        "import_launched": launched,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
