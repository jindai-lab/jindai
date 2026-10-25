"""手动清空/演练 jindai 回收站 ``/storage/.trash``（薄壳 → jindai.trash_purge）。

背景：`DELETE /bibliography/{id}`（删文献记录）会把该记录关联的文件
（path / file_attachments / cover）搬进 ``<storage>/.trash/<批次>/<原相对路径>``
（见 `jindai/trash.py`）。核心清空逻辑已迁入主库 `jindai/trash_purge.py`：
**常驻每周清空**（每周一 06:00，app lifespan 随容器启动拉起，
config.trash_purge_enabled 开关，2026-10-25 起取代原 Hermes cronjob）就在那里。
本 CLI 只用于手动演练/补跑：

  docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/purge_trash.py
  docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/purge_trash.py --apply
  ... --apply --min-age-hours 168      # 只删满 7 天的批次
  ... --apply --quiet-empty            # 没有删除任何文件时 stdout 保持空

安全阀：批次里**仍被 file_metadata 引用**的文件不删（正常不会发生——
``.trash/`` 路径不会被登记——但只要发生了就绝不制造孤儿记录），其余批次整目录
删除。汇总写 `reports/trash_purge.json`（常驻 worker 与 CLI 同写此文件）并落
`reports/trash_purge.log`。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

sys.path.insert(0, "/app")

import common as C

from jindai.trash_purge import purge_once

log = C.setup_logging("purge_trash", "batch_operation/reports/trash_purge.log")

# setup_logging 把 StreamHandler 挂在 stdout 上；改到 stderr，
# 好让 stdout 只剩最后一行汇总（外层脚本/cron 投递的就是 stdout）。
for _h in log.handlers:
    if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
        _h.setStream(sys.stderr)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="真删（默认 dry-run）")
    ap.add_argument("--min-age-hours", type=float, default=0.0,
                    help="只删龄期满 N 小时的批次（默认 0 = 全部）")
    ap.add_argument("--quiet-empty", action="store_true",
                    help="没有删除任何文件时 stdout 不输出")
    args = ap.parse_args()

    stat = await purge_once(apply=args.apply, min_age_hours=args.min_age_hours)

    if stat.get("note"):  # 回收站根不存在
        msg = f"[回收站] {stat['trash_root']} 不存在，无需清空"
        log.info(msg)
        if not args.quiet_empty:
            print(msg)
        return 0

    line = (f"[回收站{('清空' if args.apply else '演练')}] "
            f"批次 {stat['batches_purged']}/{stat['batches_seen']}，"
            f"文件 {stat['files_purged']} 个，{stat['bytes_purged'] / 1048576:.1f} MB，"
            f"跳过 {stat['batches_skipped']} 批"
            + (f"（{stat['skipped'][0]['reason']}）" if stat["skipped"] else ""))
    if stat["files_purged"] or not args.quiet_empty:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
