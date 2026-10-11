"""合并 onechar_plan2/3 → 落库（带旧值护栏 + 独立备份）

用法：apply_onechar.py --plan reports/xxx.tsv [--apply]
TSV 列：id / path / old(竖线分隔) / new(竖线分隔) / src / 证据
"""
import argparse, asyncio, csv, datetime, os, re, sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

CJK = r"\u4e00-\u9fff"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    BAD = re.compile(r"[的了著译编说话是们等]|老师|教授|先生|女士|同学")

    def namelike(old1, new1):
        """新值必须像人名：以原姓开头、2–6 字、不含虚字/称呼词"""
        if not old1 or not new1 or len(new1) < 2:
            return False
        m = re.match(rf"^([{CJK}])", old1)
        if m and not new1.startswith(m.group(1)):
            return False
        if len(new1) > 12 or BAD.search(new1):
            return False
        return True

    rows = []
    for l in open(a.plan, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if len(f) < 4 or f[0] == "id" or not f[3]:
            continue
        old, new = f[2].split("|"), f[3].split("|")
        if len(old) != len(new):
            continue
        # 只接受「全部单字作者都被补全」的行（`X et al.` 也算单字作者）
        isone = lambda x: bool(re.fullmatch(rf"[{CJK}](?: et al\.?)?", x))
        if any(isone(x) for x in new):
            continue
        if not any(isone(x) for x in old):
            continue
        if not all(namelike(o, n) for o, n in zip(old, new)):
            print(f"  跳过（不像人名）: {f[2]} -> {f[3]}  {f[1].split('/')[-1][:40]}")
            continue
        rows.append((f[0], f[1], old, new))
    # 去重（同 id 取第一条）
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
    bak = f"batch_operation/reports/onechar_backup2_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "authors"])
        for rid, path, old, _ in uniq:
            w.writerow([rid, path, "{" + ",".join(old) + "}"])

    async def _w():
        conn = await C.get_conn()
        try:
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2::text[] WHERE id = $1 "
                "AND authors = $3::text[]",
                [(rid, new, old) for rid, _, old, new in uniq])
        finally:
            await conn.close()

    asyncio.run(_w())
    print(f"[apply] updated={len(uniq)} backup={bak}")

main()
