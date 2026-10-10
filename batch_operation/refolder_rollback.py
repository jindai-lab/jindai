"""回滚 authors_refolder 第一轮造成的两处副作用（2026-10-10）。

事实（已核实）：
  * Authors/ 磁盘在本轮之前就几乎没文件 —— DB 674 行里仅 251 行的路径在盘上存在；
    计划里的 350 条 old 路径 **0/350 在盘**（历史改名/删除没同步 DB）。
  * 第一轮 per-file 分支的 os.makedirs 给 131 个**目标目录**建了空壳（11:22），
    随后 shutil.move 因源不存在而失败 → 空目录 + DB 被改写到空目录上。

本脚本：
  1. 删除本轮新建且**为空**的目录（仅限 mtime 晚于 11:15、且一层内无文件）
  2. 对计划中的行：**新路径不在盘** → 把 file_metadata.path 还原为 old
     （新路径在盘的保留 —— 那 8 条是真正修好的）
  3. 读回：还原条数、仍指向缺失文件的条数（历史遗留，交用户决定）

--dry-run 默认。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as C

log = C.setup_logging(
    "refolder_rollback", "batch_operation/reports/refolder_rollback.log"
)
STORAGE = "/storage"
PLAN = "batch_operation/reports/authors_refolder_plan.json"


async def run(apply: bool = False) -> dict:
    plan = json.load(open(PLAN))["moves"]
    conn = await C.get_conn()
    try:
        # --- 1) 空目录清理（只删本轮 11:15 之后创建、且真的空）---
        junk_dirs = []
        for top in ("Authors", "OneDrive", "Supplementary Books"):
            root = f"{STORAGE}/{top}"
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                d = os.path.join(root, name)
                if not os.path.isdir(d):
                    continue
                st = os.stat(d)
                if st.st_mtime < time.time() - 7200:   # 只认 2 小时内新建
                    continue
                if not os.listdir(d):
                    junk_dirs.append(d)
        log.info("[plan] 空目录候选 %d", len(junk_dirs))
        removed_dirs = 0
        if apply:
            for d in junk_dirs:
                try:
                    os.rmdir(d)
                    removed_dirs += 1
                except OSError as ex:
                    log.warning("[rmdir] skip %s (%s)", d, ex)

        # --- 2) 还原指向不存在文件的路径 ---
        revert, keep, already = [], [], 0
        for m in plan:
            if m["old"] == m["new"]:
                continue
            if os.path.exists(f"{STORAGE}/{m['new']}"):
                keep.append(m)
                continue
            if os.path.exists(f"{STORAGE}/{m['old']}"):
                already += 1
                continue
            revert.append(m)
        log.info("[plan] 待还原 %d，保留(新路径在盘) %d，老路径在盘 %d",
                 len(revert), len(keep), already)

        reverted = 0
        if apply and revert:
            reverted = 0
            for i in range(0, len(revert), 500):
                chunk = revert[i:i + 500]
                res = await conn.executemany(
                    "UPDATE file_metadata SET path = $2 WHERE id = $1 AND path = $3",
                    [(m["id"], m["old"], m["new"]) for m in chunk])
                reverted += len(chunk) if res else 0
            log.info("[apply] 还原 %d 条路径", reverted)

        # --- 3) 读回 ---
        still = 0
        if apply:
            for i in range(0, len(plan), 500):
                chunk = plan[i:i + 500]
                rows = await conn.fetch(
                    "SELECT path FROM file_metadata WHERE id = ANY($1)",
                    [m["id"] for m in chunk])
                still += sum(1 for r in rows
                             if not os.path.exists(f"{STORAGE}/{r['path']}"))
        out = {"applied": apply, "junk_dirs": len(junk_dirs),
               "dirs_removed": removed_dirs, "revert_planned": len(revert),
               "reverted": reverted, "kept_new": len(keep),
               "still_missing_paths": still}
        json.dump(out, open("batch_operation/reports/refolder_rollback.json", "w"),
                  ensure_ascii=False, indent=1)
        log.info("[summary] %s", json.dumps(out, ensure_ascii=False))
        return out
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    import asyncio
    asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    main()
