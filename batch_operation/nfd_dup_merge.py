#!/usr/bin/env python3
"""nfd_dup_merge.py — NFD/NFC 重复行合并（dry-run 默认，--apply）。

背景：refolder 目录改名（Marc Augé NFD→NFC）后，库里历史分裂出的 NFD 行与 NFC 行
指向同一实体文件，UPDATE 撞 uniq_path。其余 NFD 行（盘上仍以 NFD 存在、无同 NFC 行）
不动。
处置：每对 NFD/NFC 重复行保留段落字符更多的一侧，file_dataset 迁移、删另一侧（守卫：
drop 侧行数为 0 时不删行只报）。
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, unicodedata
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

log = C.setup_logging("nfd_dup_merge", "batch_operation/reports/nfd_dup_merge.log")
STORAGE = "/storage"


def has_nfd(s: str) -> bool:
    return any(unicodedata.combining(c) for c in s)


async def find_pairs(conn):
    rows = await conn.fetch(
        "SELECT id, path FROM file_metadata "
        "WHERE path LIKE 'OneDrive/%' OR path LIKE 'Authors/%' "
        "OR path LIKE 'Supplementary Books/%'")
    nfd = [(str(r["id"]), r["path"]) for r in rows if has_nfd(r["path"])]
    nfc_paths = {r["path"]: str(r["id"]) for r in rows}
    pairs = []
    others = []
    for i, p in nfd:
        nfc = unicodedata.normalize("NFC", p)
        j = nfc_paths.get(nfc)
        if j and j != i:
            pairs.append((i, j, p, nfc))
        else:
            others.append((i, p))
    return pairs, others


async def run(apply: bool) -> dict:
    conn = await C.get_conn()
    try:
        pairs, others = await find_pairs(conn)
        log.info("冲突对=%d 无冲突NFD行=%d", len(pairs), len(others))
        plan = []
        for i, j, p_nfd, p_nfc in pairs:
            # 盘上实体在哪
            disk_nfd = os.path.exists(os.path.join(STORAGE, p_nfd))
            disk_nfc = os.path.exists(os.path.join(STORAGE, p_nfc))
            chars = {}
            for pid in (i, j):
                chars[pid] = await conn.fetchval(
                    "SELECT coalesce(sum(length(content)),0) FROM paragraph "
                    "WHERE source_id=$1::uuid", pid)
            # 保留侧：字符多者；平手保留 NFC 行（其路径与盘一致）
            if chars[i] > chars[j]:
                keep, drop = i, j
            else:
                keep, drop = j, i
            plan.append({"keep": keep, "drop": drop, "nfd": p_nfd, "nfc": p_nfc,
                         "disk_nfd": disk_nfd, "disk_nfc": disk_nfc,
                         "chars_nfd": chars[i], "chars_nfc": chars[j]})
        for x in plan:
            log.info("[pair] keep=%s(drop chars nfd=%d nfc=%d) disk(nfd=%s nfc=%s) %s",
                     x["keep"][:8], x["chars_nfd"], x["chars_nfc"],
                     x["disk_nfd"], x["disk_nfc"], x["nfd"][:70])
        if not apply:
            return {"applied": False, "pairs": len(plan),
                    "detail": [{k: v for k, v in x.items()} for x in plan]}
        done = 0
        for x in plan:
            keep, drop = x["keep"], x["drop"]
            # 若 keep 行路径不在盘且 drop 行在盘 → 说明保留侧应是 drop（极少见），跳过报人
            if not os.path.exists(os.path.join(STORAGE, x["nfc"])) \
               and not os.path.exists(os.path.join(STORAGE, x["nfd"])):
                log.warning("[skip] 双侧盘上都缺: %s", x["nfd"][:70])
                continue
            try:
                # file_dataset 迁移
                await conn.execute(
                    "INSERT INTO file_dataset (file_id, dataset_id) "
                    "SELECT $1::uuid, dataset_id FROM file_dataset WHERE file_id=$2::uuid "
                    "ON CONFLICT DO NOTHING", keep, drop)
                await conn.execute("DELETE FROM file_dataset WHERE file_id=$1::uuid", drop)
                await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id=$1::uuid", drop)
                await conn.execute("DELETE FROM page_skip_state WHERE source_id=$1::uuid", drop)
                await conn.execute("DELETE FROM text_embeddings WHERE source_id=$1::uuid", drop)
                tag = await conn.execute("DELETE FROM paragraph WHERE source_id=$1::uuid", drop)
                n_para = int(tag.split()[-1])
                await conn.execute("DELETE FROM file_metadata WHERE id=$1::uuid", drop)
                done += 1
                log.info("[merge] dropped=%s paras=%d kept=%s", drop[:8], n_para, keep[:8])
            except Exception as e:  # noqa: BLE001
                log.warning("[fail] %s :: %s", x["nfd"][:60], str(e)[:150])
        return {"applied": True, "pairs": len(plan), "merged": done}
    finally:
        await conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    print(json.dumps(asyncio.run(run(ap.parse_args().apply)),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
