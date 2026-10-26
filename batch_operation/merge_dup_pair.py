"""同书两行合并：删 drop 行（连段落/嵌入/队列/绑定），保留 keep 行（用法：

  merge_dup_pair.py --keep '<路径>' --drop '<路径>' [--apply]

站规（2026-10-26 用户裁定沿用）：同书两行留**文本总字符更多**的一侧；平手留
作者归属正确的那侧。drop 侧的段落/嵌入/队列/page_skip/数据集绑定一并删，
文件与空目录随后删除（CIFS 删除有 NAS @Recycle 兜底）。备份 TSV 落 reports/。
守卫：keep 行必须在盘（或有段落）、drop 与 keep 的文件名必须一致（同名书），
否则中止。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, "/app")

import common as C

log = C.setup_logging("merge_dup_pair", "batch_operation/reports/merge_dup_pair.log")
ROOT = "/storage"


async def side_stats(conn, path: str) -> dict:
    row = await conn.fetchrow(
        "SELECT id, path FROM file_metadata WHERE path = $1", path)
    if not row:
        return {"exists": False}
    npara = await conn.fetchval(
        "SELECT count(*) FROM paragraph WHERE source_id = $1", row["id"])
    chars = await conn.fetchval(
        "SELECT coalesce(sum(length(content)),0) FROM paragraph "
        "WHERE source_id = $1", row["id"])
    return {"exists": True, "id": row["id"], "path": row["path"],
            "npara": int(npara or 0), "chars": int(chars or 0),
            "on_disk": os.path.isfile(os.path.join(ROOT, path))}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--drop", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = await C.get_conn()
    keep = await side_stats(conn, args.keep)
    drop = await side_stats(conn, args.drop)
    log.info("[plan] keep=%s", keep)
    log.info("[plan] drop=%s", drop)
    if not (keep["exists"] and drop["exists"]):
        log.error("[guard] one side missing — abort")
        return 1
    if keep["path"].split("/")[-1] != drop["path"].split("/")[-1]:
        log.error("[guard] basenames differ — not the same book — abort")
        return 1
    if keep["chars"] < drop["chars"]:
        log.warning("[guard] keep side has FEWER chars (%d < %d) — 按站规应留多的一侧，"
                    "如确认平手/作者归属理由请 --force 忽略", keep["chars"],
                    drop["chars"])
        return 1
    if not keep["on_disk"] and keep["npara"] == 0:
        log.error("[guard] keep side has neither file nor text — abort")
        return 1

    if not args.apply:
        log.info("[dry-run] nothing changed (rerun with --apply)")
        await conn.close()
        return 0

    bak = "batch_operation/reports/merge_dup_pair_%s.tsv" % time.strftime(
        "%Y%m%d-%H%M%S")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(f"KEEP\t{keep['id']}\t{keep['path']}\t{keep['chars']}\n")
        f.write(f"DROP\t{drop['id']}\t{drop['path']}\t{drop['chars']}\n")
    log.info("[apply] backup -> %s", bak)

    did = drop["id"]
    async with conn.transaction():
        await conn.execute(
            "SET LOCAL session_replication_role = replica")
        await conn.execute(
            "DELETE FROM file_dataset WHERE file_id = $1", did)
        await conn.execute(
            "DELETE FROM text_embeddings WHERE source_id = $1", did)
        await conn.execute(
            "DELETE FROM embedding_pending_queue WHERE source_id = $1", did)
        await conn.execute(
            "DELETE FROM page_skip_state WHERE source_id = $1", did)
        r = await conn.execute(
            "DELETE FROM paragraph WHERE source_id = $1", did)
        log.info("[apply] drop paragraphs: %s", r)
        r2 = await conn.execute(
            "DELETE FROM file_metadata WHERE id = $1 "
            "  AND NOT EXISTS (SELECT 1 FROM paragraph p "
            "                  WHERE p.source_id = file_metadata.id)", did)
        log.info("[apply] drop row: %s", r2)
    await conn.close()

    full = os.path.join(ROOT, drop["path"])
    try:
        if os.path.isfile(full):
            os.unlink(full)
            log.info("[rm] %s", drop["path"])
        d = os.path.dirname(full)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            log.info("[rm] removed empty dir %s", drop["path"].split("/")[1:])
    except OSError as ex:
        log.warning("[rm] failed: %s", ex)
    log.info("[done] keep=%s drop=%s", keep["path"][:80], drop["path"][:80])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
