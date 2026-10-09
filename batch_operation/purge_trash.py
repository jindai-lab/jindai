"""每周清空 jindai 回收站 ``/storage/.trash``。

背景：`DELETE /bibliography/{id}`（删文献记录）会把该记录关联的文件
（path / file_attachments / cover）搬进 ``<storage>/.trash/<批次>/<原相对路径>``，
而不是留在原地（见 `jindai/trash.py`）。``.trash`` 是隐藏目录，导入器的磁盘
upsert、`register_untracked_pdfs.py`、`pdf_disk_inventory.py`、`fix_moved_paths.py`、
`purge_sync_conflict_copies.py` 的 walk 全部跳过隐藏目录，所以里面的文件既不会被
重新登记、也不会进导入队列、更不会被"路径修复"脚本当成别的记录的落脚点。

本脚本就是那个"每周清空"：

  docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/purge_trash.py
  docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/purge_trash.py --apply
  ... --apply --min-age-hours 168      # 只删满 7 天的批次
  ... --apply --quiet-empty            # 没有东西可删时 stdout 保持空（cron 用）

安全阀：批次里**仍被 file_metadata 引用**的文件不删（正常不会发生——
`.trash/` 路径不会被登记——但只要发生了就绝不制造孤儿记录），
其余批次整目录删除。汇总写 `reports/trash_purge.json` 并追加进
`reports/trash_purge.log`。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

sys.path.insert(0, "/app")

import common as C

log = C.setup_logging("purge_trash", "batch_operation/reports/trash_purge.log")
OUT = "batch_operation/reports/trash_purge.json"

# setup_logging 把 StreamHandler 挂在 stdout 上；改到 stderr，
# 好让 stdout 只剩最后一行汇总（cron no_agent 任务投递的就是 stdout）。
for _h in log.handlers:
    if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
        _h.setStream(sys.stderr)


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


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="真删（默认 dry-run）")
    ap.add_argument("--min-age-hours", type=float, default=0.0,
                    help="只删龄期满 N 小时的批次（默认 0 = 全部）")
    ap.add_argument("--quiet-empty", action="store_true",
                    help="没有删除任何文件时 stdout 不输出（cron 用）")
    args = ap.parse_args()

    from jindai.trash import TRASH_DIRNAME, trash_root
    root = trash_root()
    if not os.path.isdir(root):
        msg = f"[回收站] {root} 不存在，无需清空"
        log.info(msg)
        if not args.quiet_empty:
            print(msg)
        return 0

    # 安全阀：`.trash/` 路径绝不该被 file_metadata 引用；有就跳过对应批次。
    conn = await C.get_conn()
    referenced = {r["path"] for r in await conn.fetch(
        "SELECT path FROM file_metadata WHERE path LIKE $1", f"{TRASH_DIRNAME}/%")}
    await conn.close()
    if referenced:
        log.warning("[guard] %d 条记录仍指向回收站，相关批次跳过: %s",
                    len(referenced), sorted(referenced)[:10])

    now = time.time()
    batches = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    stat = {"mode": "apply" if args.apply else "dry-run",
            "batches_seen": len(batches), "batches_purged": 0,
            "batches_skipped": 0, "files_purged": 0, "bytes_purged": 0,
            "skipped": []}

    for name in batches:
        bdir = os.path.join(root, name)
        try:
            age_h = (now - os.path.getmtime(bdir)) / 3600.0
        except OSError:
            continue
        if age_h < args.min_age_hours:
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
            continue

        n, b = _batch_files(bdir)
        if args.apply:
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
                 name, n, b / 1048576, age_h, "removed" if args.apply else "would remove")

    # 清空后把空的回收站根留着（导入器跳过隐藏目录，无副作用）
    if args.apply and stat["batches_purged"] and not os.listdir(root):
        log.info("[purge] 回收站已清空: %s", root)

    stat["ts"] = datetime.now().astimezone().isoformat(timespec="seconds")
    stat["trash_root"] = root
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=1)

    line = (f"[回收站{('清空' if args.apply else '演练')}] "
            f"批次 {stat['batches_purged']}/{stat['batches_seen']}，"
            f"文件 {stat['files_purged']} 个，{stat['bytes_purged'] / 1048576:.1f} MB，"
            f"跳过 {stat['batches_skipped']} 批"
            + (f"（{stat['skipped'][0]['reason']}）" if stat["skipped"] else ""))
    log.info("[done] %s", json.dumps(stat, ensure_ascii=False))
    if stat["files_purged"] or not args.quiet_empty:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
