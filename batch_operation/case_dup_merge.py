#!/usr/bin/env python3
"""case_dup_merge.py — 大小写变体重复行合并（3 对定点，dry-run 默认 --apply）。

背景：共享盘大小写不敏感，同一实体文件在 DB 有大小写变体两行。
规则：保留段落字符多的一侧，file_dataset 迁移，删另一行；保留行 path 统一为盘上实际大小写。
"""
from __future__ import annotations
import argparse, asyncio, csv, datetime as dt, json, sys
sys.path.insert(0, "/app/batch_operation")
import common as C

log = C.setup_logging("case_dup", "batch_operation/reports/case_dup.log")

# (大写行id, 小写行id, 盘上实际路径)
PAIRS = [
    ("b2783956-f004-467a-94fe-5b4c13140318", "f7dea947-e5f4-4c2a-a724-31de780f1e96",
     "OneDrive/Gyorgy Kepes/THE NATURE AND ART OF MOTION (8092)/THE NATURE AND ART OF MOTION - Gyorgy Kepes.pdf"),
    ("6d94c85e-66c9-4325-9ee0-b752f49b2baa", "5fa23d8c-bbfc-4707-82da-0ac086ea808d",
     "OneDrive/Luce Irigaray/Dialogues_ Luce Irigaray (9807)/Dialogues_ Luce Irigaray - luce irigaray.pdf"),
    ("a06fd2b9-9f22-4033-a4a0-61d446ee962b", "bfd133b5-b2ee-4990-9806-ab6b36285a0b",
     "OneDrive/Luce Irigaray/To Speak is Never Neutral (9806)/To Speak is Never Neutral - luce irigaray.pdf"),
]


async def run(apply: bool) -> dict:
    conn = await C.get_conn()
    plan = []
    try:
        for cap_id, low_id, disk_path in PAIRS:
            chars = {}
            for tag, fid in (("A", cap_id), ("B", low_id)):
                chars[tag] = await conn.fetchval(
                    "SELECT coalesce(sum(length(content)),0) FROM paragraph WHERE source_id=$1::uuid", fid)
            keep, drop = (cap_id, low_id) if chars["A"] > chars["B"] else (low_id, cap_id)
            plan.append({"keep": keep, "drop": drop, "disk_path": disk_path,
                         "chars_A": chars["A"], "chars_B": chars["B"]})
            log.info("[pair] keep=%s(%d) drop=%s(%d) -> %s", keep[:8],
                     chars["A"] if keep == cap_id else chars["B"], drop[:8],
                     chars["A"] if drop == cap_id else chars["B"], disk_path[-50:])
    finally:
        await conn.close()
    if not apply:
        return {"applied": False, "pairs": plan}
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/case_dup_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["keep_id", "drop_id", "final_path"])
        for x in plan:
            w.writerow([x["keep"], x["drop"], x["disk_path"]])
    merged = 0
    for x in plan:
        conn = await C.get_conn()
        try:
            # 顺序：先改 keep 行 path（若与目标冲突则该行已是目标路径），再删 drop
            # 1) 若 keep 行当前 path 是目标的大小写变体 → 直接改名（先查唯一性：目标路径不能已被 drop 行占用）
            kp = await conn.fetchval("SELECT path FROM file_metadata WHERE id=$1::uuid", x["keep"])
            if kp != x["disk_path"]:
                # drop 行正占着目标路径 → 先把 drop 行挪开（临时路径）
                await conn.execute(
                    "UPDATE file_metadata SET path='case_dup_tmp/'||id::text WHERE id=$1::uuid AND path=$2",
                    x["drop"], x["disk_path"])
                tag = await conn.execute(
                    "UPDATE file_metadata SET path=$2 WHERE id=$1::uuid AND path=$3",
                    x["keep"], x["disk_path"], kp)
                log.info("[repath] %s -> %s (%s)", kp[-45:], x["disk_path"][-45:], tag)
            # 2) 数据集迁移
            await conn.execute(
                "INSERT INTO file_dataset (file_id, dataset_id) "
                "SELECT $1::uuid, dataset_id FROM file_dataset WHERE file_id=$2::uuid ON CONFLICT DO NOTHING",
                x["keep"], x["drop"])
            await conn.execute("DELETE FROM file_dataset WHERE file_id=$1::uuid", x["drop"])
            await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id=$1::uuid", x["drop"])
            await conn.execute("DELETE FROM page_skip_state WHERE source_id=$1::uuid", x["drop"])
            await conn.execute("DELETE FROM text_embeddings WHERE source_id=$1::uuid", x["drop"])
            n = await conn.execute("DELETE FROM paragraph WHERE source_id=$1::uuid", x["drop"])
            await conn.execute("DELETE FROM file_metadata WHERE id=$1::uuid", x["drop"])
            merged += 1
            log.info("[merge] drop_paras=%s", n)
        finally:
            await conn.close()
    # epub 行路径顺手统一（To Speak 那本的 epub 盘上也是小写文件名）
    conn = await C.get_conn()
    try:
        await conn.execute(
            "UPDATE file_metadata SET path='OneDrive/Luce Irigaray/To Speak is Never Neutral (9806)/To Speak is Never Neutral - luce irigaray.epub' "
            "WHERE path='OneDrive/Luce Irigaray/To Speak is Never Neutral (9806)/To Speak is Never Neutral - Luce Irigaray.epub'")
    finally:
        await conn.close()
    return {"applied": True, "merged": merged, "backup": bak}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    print(json.dumps(asyncio.run(run(ap.parse_args().apply)), ensure_ascii=False, indent=1))
