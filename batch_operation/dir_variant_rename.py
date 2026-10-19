#!/usr/bin/env python3
"""dir_variant_rename.py — 目录异写组定点改名（dry-run 默认，--apply）。

范围：refolder 因 token_same（点号/换序异写）不动的 17 组目录。
动作：整目录共识 rename（目标已存在则逐文件并入）→ file_metadata.path 同步。
数据集绑定不在此处动——随后 backfill_folder_datasets 按新目录名重建。
计划：reports/dir_variant_plan.json（宿主侧生成）。
"""
from __future__ import annotations
import argparse, asyncio, csv, datetime as dt, json, os, shutil, sys
sys.path.insert(0, "/app")
import common as C

log = C.setup_logging("dir_variant", "batch_operation/reports/dir_variant.log")
ST = "/storage"
PLAN = "/app/batch_operation/reports/dir_variant_plan.json"


def main_plan_dir(p):  # 取目录路径
    return "/".join(p.split("/")[:2])


async def run(apply: bool) -> dict:
    plan = json.load(open(PLAN, encoding="utf-8"))   # [{top, old, new, files}]
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path FROM file_metadata "
            "WHERE split_part(path,'/',1)||'/'||split_part(path,'/',2) = ANY($1::text[])",
            [f"{g['top']}/{g['old']}" for g in plan])
    finally:
        await conn.close()
    by_dir = {}
    for r in rows:
        by_dir.setdefault(main_plan_dir(r["path"]), []).append((str(r["id"]), r["path"]))
    log.info("组=%d 命中行=%d", len(plan), len(rows))
    moves = []
    for g in plan:
        d = f"{g['top']}/{g['old']}"
        tgt = f"{g['top']}/{g['new']}"
        src_disk, dst_disk = os.path.join(ST, d), os.path.join(ST, tgt)
        whole = os.path.isdir(src_disk) and not os.path.exists(dst_disk)
        for pid, p in by_dir.get(d, []):
            seg = p.split("/")
            seg[1] = g["new"]
            moves.append({"id": pid, "old": p, "new": "/".join(seg),
                          "dir_whole": whole, "src": src_disk, "tgt": dst_disk})
    if not apply:
        return {"applied": False, "moves": len(moves),
                "whole_dirs": sum(1 for g in plan
                                  if os.path.isdir(os.path.join(ST, g["top"], g["old"]))
                                  and not os.path.exists(os.path.join(ST, g["top"], g["new"]))),
                "detail": moves[:20]}
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/dir_variant_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "old_path", "new_path"])
        for m in moves:
            w.writerow([m["id"], m["old"], m["new"]])
    done_dirs = set()
    moved = upd = 0
    for m in moves:
        try:
            if m["dir_whole"] and m["src"] not in done_dirs:
                os.rename(m["src"], m["tgt"])
                done_dirs.add(m["src"])
                moved += 1
            else:
                # 逐文件（目标目录已存在）
                os.makedirs(os.path.dirname(os.path.join(ST, m["new"])), exist_ok=True)
                s2 = os.path.join(ST, m["old"])
                if os.path.exists(s2):
                    shutil.move(s2, os.path.join(ST, m["new"]))
                    moved += 1
            conn = await C.get_conn()
            try:
                tag = await conn.execute(
                    "UPDATE file_metadata SET path=$2 WHERE id=$1::uuid "
                    "AND path=$3 AND NOT EXISTS "
                    "(SELECT 1 FROM file_metadata f2 WHERE f2.path=$2 AND f2.id<>$1::uuid)",
                    m["id"], m["new"], m["old"])
            finally:
                await conn.close()
            if tag.endswith("1"):
                upd += 1
            else:
                log.warning("[skip] path 冲突或不匹配: %s", m["old"][:70])
        except Exception as e:  # noqa: BLE001
            log.warning("[fail] %s :: %s", m["old"][:60], str(e)[:150])
    summary = {"applied": True, "moves": len(moves), "disk_moved": moved,
               "db_updated": upd, "dirs_renamed": len(done_dirs), "backup": bak}
    log.info("[apply] %s", json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    print(json.dumps(asyncio.run(run(ap.parse_args().apply)), ensure_ascii=False, indent=1))
