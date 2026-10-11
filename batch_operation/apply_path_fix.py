"""路径修复落库（带旧值护栏 + 独立备份）
TSV 列：id / old_path / new_path / 规则 / ...
用法：apply_path_fix.py --plan reports/xxx.tsv [--apply]
"""
import argparse, asyncio, csv, datetime, sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = []
    for l in open(a.plan, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if len(f) < 3 or f[0] == "id" or not f[0]:
            continue
        rows.append((f[0], f[1], f[2], f[3] if len(f) > 3 else ""))
    seen, uniq = set(), []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        uniq.append(r)
    print(f"计划行 {len(rows)}，去重后 {len(uniq)}")
    if not a.apply:
        print("[dry-run] 未写库")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/path_fix_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path"])
        for rid, old, new, rule in uniq:
            w.writerow([rid, old])

    async def _w():
        conn = await C.get_conn()
        try:
            # 先确认旧值仍存在，避免误改
            cur = await conn.fetch("SELECT id, path FROM file_metadata WHERE id = ANY($1::uuid[])",
                                   [r[0] for r in uniq])
            now = {str(x["id"]): x["path"] for x in cur}
            todo = [(rid, new, old) for rid, old, new, _ in uniq if now.get(rid) == old]
            skipped = len(uniq) - len(todo)
            upd = dup = 0
            dups = []
            for rid, new, old in todo:
                r = await conn.execute(
                    "UPDATE file_metadata SET path = $2 WHERE id = $1 AND path = $3 "
                    "AND NOT EXISTS (SELECT 1 FROM file_metadata WHERE path = $2 AND id <> $1)",
                    rid, new, old)
                if r.endswith("1"):
                    upd += 1
                else:
                    dup += 1
                    dups.append((rid, old, new))
            print(f"[apply] updated={upd} 目标已被占用(重复记录)={dup} skipped(旧值不匹配)={skipped} backup={bak}")
            if dups:
                with open("batch_operation/reports/path_fix_duplicates.tsv", "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["id", "old_path", "target_path"])
                    for d in dups:
                        w.writerow(d)
        finally:
            await conn.close()

    asyncio.run(_w())


main()
