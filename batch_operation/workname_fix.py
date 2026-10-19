#!/usr/bin/env python3
"""workname_fix.py — 「作品名混进作者位/题名」定点修复（dry-run 默认，--apply 写库）。

2026-10-19 用户指令：判读为 work 的条目修复标题与作者信息。
计划来源：scratch/work_fixes.json（逐条附证据；由 PDF 首页/OCR版权页/CIP/epub dc:creator/
联网核验得出）。行级护栏：UPDATE 带 old title + old authors 双守卫；先落备份 TSV。
"""
from __future__ import annotations
import argparse, asyncio, csv, datetime as dt, json, os, sys
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

log = C.setup_logging("workname_fix", "batch_operation/reports/workname_fix.log")
PLAN = "/app/batch_operation/reports/work_fixes.json"          # docker cp 进来
OLD = "/app/batch_operation/reports/work_rows_old.tsv"          # path -> old title/authors

def load_plan():
    fixes = json.load(open(PLAN, encoding="utf-8"))
    olds = {}
    with open(OLD, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3:
                olds[row[0]] = (row[1], row[2])
    return fixes, olds


def parse_pgarr(s: str) -> list[str]:
    s = (s or "").strip()
    if s.startswith("{"): s = s[1:-1]
    if not s: return []
    out, cur, inq = [], "", False
    for ch in s:
        if ch == '"': inq = not inq
        elif ch == "," and not inq:
            out.append(cur); cur = ""
        else: cur += ch
    out.append(cur)
    return [x.strip().strip('"') for x in out]

async def run(apply: bool) -> dict:
    fixes, olds = load_plan()
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, coalesce(title,'') AS title, authors "
            "FROM file_metadata WHERE path = ANY($1::text[])", list(fixes))
    finally:
        await conn.close()
    by_path = {r["path"]: r for r in rows}
    plan, skipped = [], []
    for p, (nt, na, ev) in fixes.items():
        r = by_path.get(p)
        if not r:
            skipped.append((p, "not_in_db")); continue
        old_t, old_a_raw = olds.get(p, ("", ""))
        old_a = parse_pgarr(old_a_raw)
        if r["title"] != old_t or list(r["authors"] or []) != old_a:
            skipped.append((p, f"drift title={r['title'][:30]!r} authors={list(r['authors'])}"))
            continue
        new_t = nt if nt is not None else r["title"]
        new_a = na if na is not None else list(r["authors"] or [])
        if new_t == r["title"] and new_a == list(r["authors"] or []):
            skipped.append((p, "no_change")); continue
        plan.append({"id": str(r["id"]), "path": p,
                     "old_title": r["title"], "new_title": new_t,
                     "old_authors": list(r["authors"] or []), "new_authors": new_a,
                     "ev": ev})
    log.info("plan=%d skipped=%d", len(plan), len(skipped))
    for s in skipped[:10]: log.info("skip %s :: %s", s[0][:60], s[1])
    if not apply:
        return {"applied": False, "planned": len(plan), "skipped": len(skipped),
                "detail": [{"path": x["path"], "t": [x["old_title"], x["new_title"]],
                            "a": [x["old_authors"], x["new_authors"]], "ev": x["ev"]}
                           for x in plan]}
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/workname_fix_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old_title", "old_authors"])
        for x in plan:
            w.writerow([x["id"], x["path"], x["old_title"],
                        "{" + ",".join(x["old_authors"]) + "}"])
    conn = await C.get_conn()
    try:
        upd_t = upd_a = 0
        for x in plan:
            tag_t = await conn.execute(
                "UPDATE file_metadata SET title=$2 WHERE id=$1 "
                "AND coalesce(title,'')=$3 AND authors=$4::text[]",
                x["id"], x["new_title"], x["old_title"], x["old_authors"])
            if tag_t.endswith("1"): upd_t += 1
            tag_a = await conn.execute(
                "UPDATE file_metadata SET authors=$2::text[] WHERE id=$1 "
                "AND coalesce(title,'')=$3 AND authors=$4::text[]",
                x["id"], x["new_authors"], x["new_title"], x["old_authors"])
            if tag_a.endswith("1"): upd_a += 1
    finally:
        await conn.close()
    # 读回：逐行核对落库终态
    conn = await C.get_conn()
    try:
        post = await conn.fetch(
            "SELECT path, coalesce(title,'') AS title, authors "
            "FROM file_metadata WHERE path = ANY($1::text[])", [x["path"] for x in plan])
    finally:
        await conn.close()
    post_by = {r["path"]: (r["title"], list(r["authors"] or [])) for r in post}
    verify_ok = sum(
        1 for x in plan
        if post_by.get(x["path"]) == (x["new_title"], x["new_authors"]))
    summary = {"applied": True, "planned": len(plan), "title_updated": upd_t,
               "authors_updated": upd_a, "verify_ok": verify_ok, "backup": bak}
    log.info("[apply] %s", json.dumps(summary, ensure_ascii=False))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args.apply)), ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
