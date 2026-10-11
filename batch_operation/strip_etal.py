"""剥掉 authors 元素结尾的「 et al.」/「et al」/「等」，并在数组内去重（保序）。

产物：reports/etal_strip_backup_<ts>.tsv（旧值）+ 报表
用法：python strip_etal.py [--apply]
"""
import argparse, asyncio, csv, datetime, re, sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

PAT = re.compile(r"(?:[,，、]?\s*(?:et\.?\s*al\.?|等))\s*$", re.I)
# 整个元素就是「etal」这类占位（无空格无点）→ 整条删除
WHOLE = re.compile(r"^\s*(?:et\.?\s*al\.?|等)\s*$", re.I)


def strip(a: str):
    new = PAT.sub("", a).strip(" ,，、")
    return new or a


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, authors FROM file_metadata "
            "WHERE EXISTS (SELECT 1 FROM unnest(authors) x "
            "   WHERE x ~* '(^|[[:space:],，、])(et[[:space:]]*\\.?[[:space:]]*al\\.?)$' OR x ~ '等$')")
        print(f"含 et al/等 的元素：{len(rows)} 文件")
        plan = []
        for r in rows:
            old = list(r["authors"])
            new, seen = [], set()
            for x in old:
                if WHOLE.match(x):
                    continue          # 「etal」不是人名，整条去掉
                y = strip(x)
                if y not in seen:
                    seen.add(y)
                    new.append(y)
            if new != old:
                plan.append((r["id"], r["path"], old, new))
        print(f"需改动：{len(plan)} 文件")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"batch_operation/reports/etal_strip_backup_{stamp}.tsv"
        with open(bak, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "path", "old", "new"])
            for i, p, o, n in plan:
                w.writerow([i, p, "|".join(o), "|".join(n)])
        print(f"备份 → {bak}")
        for _, p, o, n in plan[:12]:
            print(f"   {o} -> {n}")
        if not a.apply:
            print("[dry-run] 未写库")
            return
        async with conn.transaction():
            for i, p, o, n in plan:
                await conn.execute(
                    "UPDATE file_metadata SET authors=$2::text[] WHERE id=$1 AND authors=$3::text[]",
                    i, n, o)
        left = await conn.fetchval(
            "SELECT count(*) FROM file_metadata WHERE EXISTS (SELECT 1 FROM unnest(authors) x "
            "   WHERE x ~* '(^|[[:space:],，、])(et[[:space:]]*\\.?[[:space:]]*al\\.?)$' OR x ~ '等$')")
        print(f"[apply] 已写 {len(plan)} 行；残留含 et al/等 = {left}")
    finally:
        await conn.close()

asyncio.run(main())
