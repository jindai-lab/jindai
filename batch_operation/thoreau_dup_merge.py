#!/usr/bin/env python3
"""thoreau_dup_merge.py — HENRY DAVID THOREAU/Henry David Thoreau 同书两行合并（dry-run 默认）。

9 对重复：两侧盘文件 size/mtime 相同、DB 两行分别指向两目录。
规则：保留段落字符更多的一侧；file_dataset 迁移；删另一行（守卫 0 段落即可删行）；
盘上删冗余副本（只在两侧文件 size 相同时删 drop 侧，保留 keep 侧）。旧目录其余成员
（Complete Works 副文件、Walden 等）不在本批范围。
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys
sys.path.insert(0, "/app")
import common as C

log = C.setup_logging("thoreau_dup", "batch_operation/reports/thoreau_dup.log")
ST = "/storage"
OLD_D = "OneDrive/HENRY DAVID THOREAU"
NEW_D = "OneDrive/Henry David Thoreau"
BOOKS = ["瓦尔登湖 (20040)", "瓦尔登湖 (20041)", "瓦尔登湖 (20042)", "瓦尔登湖 (20043)",
         "瓦尔登湖 (20044)", "瓦尔登湖 (20045)", "梭罗日记 (15787)", "梭罗集 (15788)",
         "HENRY DAVID THOREAU (12764)"]


async def run(apply: bool) -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, coalesce(authors,'{}') AS authors FROM file_metadata "
            "WHERE path LIKE $1 || '/%/%' OR path LIKE $2 || '/%/%'", OLD_D, NEW_D)
    finally:
        await conn.close()
    by = {}
    for r in rows:
        seg = r["path"].split("/")
        by[(seg[1], seg[2])] = r
    plan = []
    for b in BOOKS:
        o = by.get(("HENRY DAVID THOREAU", b))
        n = by.get(("Henry David Thoreau", b))
        if not o or not n:
            log.info("[skip] %s old=%s new=%s", b, bool(o), bool(n))
            continue
        conn = await C.get_conn()
        try:
            chars = {}
            for tag, r in (("old", o), ("new", n)):
                chars[tag] = await conn.fetchval(
                    "SELECT coalesce(sum(length(content)),0) FROM paragraph "
                    "WHERE source_id=$1::uuid", r["id"])
        finally:
            await conn.close()
        if chars["old"] > chars["new"]:
            keep, drop = ("old", o), ("new", n)
        else:
            keep, drop = ("new", n), ("old", o)
        plan.append({"keep_id": str(keep[1]["id"]), "keep_path": keep[1]["path"],
                     "drop_id": str(drop[1]["id"]), "drop_path": drop[1]["path"],
                     "chars_old": chars["old"], "chars_new": chars["new"]})
        log.info("[pair] %s keep=%s(%d) drop=%s(%d)", b, keep[0], chars[keep[0]],
                 drop[0], chars[drop[0]])
    if not apply:
        return {"applied": False, "pairs": len(plan), "detail": plan}
    done = 0
    for x in plan:
        kp, dp = x["keep_path"], x["drop_path"]
        # 盘上：仅当两侧文件都在且 size 相同才删 drop 侧文件（同实体副本）
        kf, df = os.path.join(ST, kp), os.path.join(ST, dp)
        same = (os.path.exists(kf) and os.path.exists(df)
                and os.path.getsize(kf) == os.path.getsize(df))
        conn = await C.get_conn()
        try:
            await conn.execute(
                "INSERT INTO file_dataset (file_id, dataset_id) "
                "SELECT $1::uuid, dataset_id FROM file_dataset WHERE file_id=$2::uuid "
                "ON CONFLICT DO NOTHING", x["keep_id"], x["drop_id"])
            await conn.execute("DELETE FROM file_dataset WHERE file_id=$1::uuid", x["drop_id"])
            await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id=$1::uuid", x["drop_id"])
            await conn.execute("DELETE FROM page_skip_state WHERE source_id=$1::uuid", x["drop_id"])
            await conn.execute("DELETE FROM text_embeddings WHERE source_id=$1::uuid", x["drop_id"])
            tag = await conn.execute("DELETE FROM paragraph WHERE source_id=$1::uuid", x["drop_id"])
            n_para = int(tag.split()[-1])
            await conn.execute("DELETE FROM file_metadata WHERE id=$1::uuid", x["drop_id"])
        finally:
            await conn.close()
        if same:
            os.remove(df)
            book_dir = os.path.dirname(df)
            try:
                if not os.listdir(book_dir):
                    os.rmdir(book_dir)
            except OSError:
                pass
        done += 1
        log.info("[merge] drop=%s paras=%d dup_file_removed=%s", dp[:60], n_para, same)
    # 旧目录若只剩副文件/空目录，保留并报告
    return {"applied": True, "pairs": len(plan), "merged": done}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    print(json.dumps(asyncio.run(run(ap.parse_args().apply)), ensure_ascii=False, indent=1))
