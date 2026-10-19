#!/usr/bin/env python3
"""name_map_apply.py — 按判读映射表统一作者名（dry-run 默认，--apply 写库）。

2026-10-19 用户指令：外国作者无论是否有中文名都用原名（华裔及日韩除外）。
映射来源：scratch/name_final_map.json —— foreign 676 条（CJK 译名→拉丁原名）
+ jp/kr 26 条（日文汉字→简体中文）。work/org/unknown/diaspora/zh 不在表内 → 不动。

行级动作：authors 数组里命中键 → 逐元素替换（保序、去重）。护栏：UPDATE 带
authors 旧值数组比对；先落备份 TSV；读回逐行核对终态。
"""
from __future__ import annotations
import argparse, asyncio, csv, datetime as dt, json, os, sys
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

log = C.setup_logging("name_map_apply", "batch_operation/reports/name_map_apply.log")
MAP = "/app/batch_operation/reports/name_final_map.json"


async def run(apply: bool) -> dict:
    m = json.load(open(MAP, encoding="utf-8"))
    keys = list(m)
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, authors FROM file_metadata "
            "WHERE authors && $1::text[]", keys)
    finally:
        await conn.close()
    log.info("命中行=%d 映射键=%d", len(rows), len(keys))
    plan = []
    for r in rows:
        old = list(r["authors"] or [])
        new = list(dict.fromkeys(m.get(a, a) for a in old))
        if new != old:
            plan.append({"id": str(r["id"]), "path": r["path"],
                         "old": old, "new": new})
    by_old = sum(1 for p in plan if any(a in m for a in p["old"]))
    log.info("to_update=%d", len(plan))
    if not apply:
        return {"applied": False, "to_update": len(plan),
                "sample": [{"path": p["path"][:70], "old": p["old"], "new": p["new"]}
                           for p in plan[:15]]}
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/name_map_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "authors"])
        for p in plan:
            w.writerow([p["id"], p["path"], "{" + ",".join(p["old"]) + "}"])
    conn = await C.get_conn()
    try:
        # executemany 的 tag 只反映最后一条语句，不作计数依据——信读回
        await conn.executemany(
            "UPDATE file_metadata SET authors=$2::text[] WHERE id=$1 AND authors=$3::text[]",
            [(p["id"], p["new"], p["old"]) for p in plan])
    finally:
        await conn.close()
    # 读回
    conn = await C.get_conn()
    try:
        post = await conn.fetch(
            "SELECT id, authors FROM file_metadata WHERE id = ANY($1::uuid[])",
            [p["id"] for p in plan])
    finally:
        await conn.close()
    post_by = {str(r["id"]): list(r["authors"] or []) for r in post}
    verify_ok = sum(1 for p in plan if post_by.get(p["id"]) == p["new"])
    # 残留：仍在用映射键的行
    conn = await C.get_conn()
    try:
        leftover = await conn.fetchval(
            "SELECT count(*) FROM file_metadata WHERE authors && $1::text[]", keys)
    finally:
        await conn.close()
    summary = {"applied": True, "to_update": len(plan),
               "verify_ok": verify_ok, "leftover_rows": leftover, "backup": bak}
    log.info("[apply] %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args.apply)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
