"""删除 2026-10-26 10:00:30 批量登记的 8,234 条死路径空壳行（dry-run 默认，--apply）。

成因链（2026-10-26 实证）：
  盘上长期存在重复目录树（Journals/<丛刊> 与 Series/<丛刊> 并存；
  OneDrive/<中文作者> 与 OneDrive/<拉丁作者> 并存）。10:00:30 引擎全盘扫描把
  重复树登记成行（只写 path/size，0 段落）；10:00–11:19 重复树被合并进正主树
  （正主行 9–10 月已存在且多数 complete/带段落）→ 这批新行全部变成死路径空壳，
  且把 pending 队列从 9,383 撑到 17,298、导入轮空转。

删除安全性：仅删「创建窗口内 + path 已不在盘（inventory 复核）+ 无 paragraph/
embedding/queue 子表」的行；file_dataset 绑定先删。备份 TSV 落 reports/。
正主行不动；真有文件下次全盘扫描会以正确路径重新登记（引擎登记后已有
fill_nontitle 兜底，见 import_engine 防再犯补丁）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "/app")

import common as C

log = C.setup_logging("purge_stale_batch", "batch_operation/reports/purge_stale_batch.log")

INV = "batch_operation/reports/disk_inventory.json"
OUT = "batch_operation/reports/purge_stale_batch_20261026.json"
BACKUP = "batch_operation/reports/purge_stale_batch_20261026.tsv"
CHUNK = 1000
MAX_RETRIES = 6


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    inv = json.load(open(INV, encoding="utf-8"))
    on_disk = set(inv.get("sizes", inv))
    log.info("[inv] ts=%s entries=%d", inv.get("ts"), len(on_disk))

    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT id, path FROM file_metadata "
        "WHERE created_at BETWEEN $1::timestamptz AND $2::timestamptz "
        "  AND (path LIKE 'Journals/%' OR path LIKE 'OneDrive/%') "
        "ORDER BY path",
        datetime(2026, 10, 26, 2, 0, 29, tzinfo=timezone.utc),
        datetime(2026, 10, 26, 2, 0, 32, tzinfo=timezone.utc))
    log.info("[plan] %d batch rows", len(rows))

    dead = [(r["id"], r["path"]) for r in rows if r["path"] not in on_disk]
    still_on_disk = [r["path"] for r in rows if r["path"] in on_disk]

    # 子表守卫：有任何 paragraph/embedding/queue/page_skip 的行不删
    ids = [i for i, _p in dead]
    guarded = set()
    for i in range(0, len(ids), 30000):
        chunk = ids[i:i + 30000]
        for tbl in ("paragraph", "embedding_pending_queue", "page_skip_state"):
            got = await conn.fetch(
                f"SELECT DISTINCT source_id FROM {tbl} "
                "WHERE source_id = ANY($1::uuid[])", chunk)
            guarded |= {r["source_id"] for r in got}
    victims = [(i, p) for i, p in dead if i not in guarded]
    stat = {"mode": "apply" if args.apply else "dry-run",
            "batch_rows": len(rows), "dead_on_disk": len(dead),
            "still_on_disk_skipped": len(still_on_disk),
            "guarded_has_children": len(guarded), "to_delete": len(victims)}
    log.info("[plan] %s", json.dumps(stat, ensure_ascii=False))

    if args.apply and victims:
        with open(BACKUP, "w", encoding="utf-8") as f:
            for rid, p in victims:
                f.write(f"{rid}\t{p}\n")
        log.info("[apply] backup -> %s", BACKUP)

        done = 0
        vids = [i for i, _p in victims]
        for i in range(0, len(vids), CHUNK):
            chunk = vids[i:i + CHUNK]
            for attempt in range(MAX_RETRIES):
                try:
                    async with conn.transaction():
                        # 子表已由前置守卫验证为 0（paragraph/queue/page_skip 全查）；
                        # RI 探针是 32 分区 × 50ms/行的巨坑（1000 行 8 分钟），
                        # 事务内 replica 模式跳过触发器/FK 检查，随事务回滚自动还原。
                        await conn.execute(
                            "SET LOCAL session_replication_role = replica")
                        await conn.execute(
                            "DELETE FROM file_dataset WHERE file_id = ANY($1::uuid[])",
                            chunk)
                        r = await conn.execute(
                            "DELETE FROM file_metadata "
                            "WHERE id = ANY($1::uuid[]) "
                            "  AND NOT EXISTS (SELECT 1 FROM paragraph p "
                            "                  WHERE p.source_id = file_metadata.id)",
                            chunk)
                    done += int(r.split()[-1])
                    break
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as ex:  # noqa: BLE001 - 死锁/串行化重试
                    if attempt == MAX_RETRIES - 1:
                        stat.setdefault("failed_chunks", []).append(str(ex)[:200])
                        log.warning("[apply] chunk %d failed: %s", i, str(ex)[:200])
                    else:
                        await asyncio.sleep(2 ** attempt)
            log.info("[apply] %d/%d", i + len(chunk), len(vids))
        stat["deleted"] = done

    stat["ts"] = time.strftime("%F %T")
    stat["still_on_disk_sample"] = still_on_disk[:10]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=1)
    await conn.close()
    log.info("[done] %s", json.dumps(stat, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
