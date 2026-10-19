#!/usr/bin/env python3
"""ds_variant_merge.py — 「书库--X」数据集异写组合并（dry-run 默认，--apply）。

2026-10-19 用户指令：数据集名称同步合并更正。
异写组 = 归一化键（NFKD 去变音、去标点空白小写）相同的数据集名集合。规范名选取：
  ① 与作者目录集（file_metadata path 段2）同名者优先（backfill 规则会重建它）
  ② 其余选带重音/缩写点/连字符的完整形，再看成员数
成员迁移（INSERT SELECT … ON CONFLICT DO NOTHING → 删旧行绑定 → 删零成员数据集）；
删行前查三处引用：task_dbo.pipeline、dataset.alias_of（被指）、file_metadata.dataset_id。
零成员的非规范异写数据集一并删。所有删除先备份 TSV。
"""
from __future__ import annotations
import argparse, asyncio, csv, datetime as dt, json, os, re, sys, unicodedata
from collections import defaultdict
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

log = C.setup_logging("ds_variant_merge", "batch_operation/reports/ds_variant_merge.log")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", "", s.lower())


def pick_canon(names: list[str], members: dict[str, int], dirs: set[str]) -> str:
    indir = [n for n in names if n in dirs]
    pool = indir if indir else names
    if len(pool) == 1:
        return pool[0]

    def pref(x: str):
        return (-int(any(ord(c) > 127 for c in x)),
                -(x.count(".") + x.count("-") + x.count(" ")),
                -members.get(x, 0), x)
    return sorted(pool, key=pref)[0]


async def run(apply: bool) -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT d.id, d.name, count(fd.file_id) AS n FROM dataset d "
            "LEFT JOIN file_dataset fd ON fd.dataset_id=d.id "
            "WHERE d.name LIKE '书库--%' GROUP BY d.id, d.name")
        dirs = {p for p in (
            r["path"] for r in await conn.fetch(
                "SELECT DISTINCT split_part(path,'/',2) AS path FROM file_metadata "
                "WHERE (path LIKE 'OneDrive/%' OR path LIKE 'Supplementary Books/%') "
                "AND array_length(string_to_array(path,'/'),1) >= 3")) if p}
    finally:
        await conn.close()

    groups = defaultdict(list)
    mem = {}
    ids = {}
    for r in rows:
        suf = r["name"].split("书库--", 1)[1]
        groups[norm(suf)].append(suf)
        mem[suf] = r["n"]
        ids[suf] = str(r["id"])
    variants = {k: v for k, v in groups.items() if len(v) > 1}
    log.info("书库数据集=%d 异写组=%d 涉及数据集=%d", len(rows), len(variants),
             sum(len(v) for v in variants.values()))

    moves, drops = [], []     # moves: 成员迁移 (from,to)；drops: 删数据集
    for k, names in variants.items():
        canon = pick_canon(names, mem, dirs)
        for n in names:
            if n != canon:
                moves.append((ids[n], n, ids[canon], canon, mem.get(n, 0)))
        for n in names:
            if n != canon and mem.get(n, 0) == 0:
                drops.append((ids[n], n))

    # 零成员且后缀不在目录集的孤名数据集（非异写成员）——单独列出，谨慎删
    lone_zero = [s for s in mem if mem[s] == 0 and s not in dirs
                 and not any(s in v for v in variants.values())]
    log.info("迁移=%d 零成员异写删除=%d 零成员孤名(列而不删)=%d",
             len(moves), len(drops), len(lone_zero))
    if not apply:
        return {"applied": False, "moves": len(moves), "drops": len(drops),
                "lone_zero": len(lone_zero),
                "sample": [{"from": m[1], "to": m[3], "members": m[4]} for m in moves[:20]]}

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/ds_variant_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset_id", "name", "members", "action"])
        for m in moves:
            w.writerow([m[0], m[1], m[4], f"merge_into:{m[3]}"])
        for d in drops:
            w.writerow([d[0], d[1], 0, "drop"])

    conn = await C.get_conn()
    moved = dropped = 0
    try:
        for fid, fname, tid, tname, n in moves:
            await conn.execute(
                "INSERT INTO file_dataset (file_id, dataset_id) "
                "SELECT file_id, $1::uuid FROM file_dataset WHERE dataset_id=$2::uuid "
                "ON CONFLICT DO NOTHING", tid, fid)
            await conn.execute("DELETE FROM file_dataset WHERE dataset_id=$1::uuid", fid)
            moved += 1
        # 删数据集行（三处引用检查）
        for did, name in drops:
            refs = await conn.fetchval(
                "SELECT count(*) FROM task_dbo WHERE pipeline::text LIKE '%'||$1||'%' ", name)
            alias_ref = await conn.fetchval(
                "SELECT count(*) FROM dataset WHERE alias_of=$1::uuid", did)
            fmeta_ref = await conn.fetchval(
                "SELECT count(*) FROM file_metadata WHERE dataset_id=$1::uuid", did)
            if refs or alias_ref or fmeta_ref:
                log.warning("[keep] %s 仍被引用(task=%d alias=%d fmeta=%d)",
                            name, refs, alias_ref, fmeta_ref)
                continue
            tag = await conn.execute("DELETE FROM dataset WHERE id=$1::uuid", did)
            if tag.endswith("1"):
                dropped += 1
    finally:
        await conn.close()
    summary = {"applied": True, "moves": len(moves), "moved": moved,
               "drops": len(drops), "dropped": dropped, "backup": bak,
               "lone_zero_listed": len(lone_zero)}
    with open("batch_operation/reports/ds_variant_lone_zero.json", "w",
              encoding="utf-8") as f:
        json.dump(lone_zero, f, ensure_ascii=False, indent=1)
    log.info("[apply] %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    print(json.dumps(asyncio.run(run(ap.parse_args().apply)),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
