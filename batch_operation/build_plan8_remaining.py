#!/usr/bin/env python3
"""plan8：对剩余行放宽匹配——标题/文件名「归一化包含」+ 作者重叠（作者缺失时只看标题包含）。
命中存活记录（文件在盘）→ (1) 对方有正文→删本行 (2) 对方无正文→删对方、本行改指。
输出 plan8_pairs.tsv（id/old_path/target_path）
"""
import csv, os, json, re, subprocess, collections
import sys

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
HOST = "/misc/Public/jindai-storage"
rem = list(csv.DictReader(open(P + "remaining_final.tsv", encoding="utf-8"), delimiter="\t"))

dbrows = [l.split("\t") for l in subprocess.run(
    ["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t", "-F", "\t", "-c",
     "SELECT id, path, coalesce(title,''), coalesce(array_to_string(authors,'|'),''), "
     "(SELECT count(*) FROM paragraph p WHERE p.source_id=f.id) FROM file_metadata f"],
    capture_output=True, text=True).stdout.splitlines() if l.count("\t") == 4]
idmap = {r[1]: r[0] for r in dbrows}


def norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())


def toks(s):
    return {t for t in re.split(r"[|,、;（）()\[\]]+", s) if len(t.strip()) >= 2}


def stem(s):
    return norm(os.path.splitext(s)[0])


disk = set()
for _l in open(P + "disk_index.jsonl", encoding="utf-8"):
    disk.add(json.loads(_l)["p"])
live = [r for r in dbrows if (r[1][7:] if r[1].startswith("../app/") else r[1]) in disk]
print(f"剩余 {len(rem)} | 存活记录 {len(live)}")

# n-gram 倒排索引（8-gram, step 3）预筛
inv = collections.defaultdict(set)
def grams(x, n=8, step=3):
    return {x[i:i+n] for i in range(0, max(1, len(x)-n+1), step)} if len(x) >= n else set()
for i, t in enumerate(live):
    for g in grams(norm(t[2])) | grams(stem(t[1].split("/")[-1])):
        inv[g].add(i)
print("倒排索引 n-gram", len(inv))

pairs, info = [], []
for r in rem:
    myt, mya, mystem = norm(r["title"]), toks(r["authors"]), stem(r["file"])
    myg = grams(myt) | grams(mystem)
    cnt = collections.Counter()
    for g in myg:
        for i in inv.get(g, ()):
            cnt[i] += 1
    cand_idx = [i for i, c in cnt.items() if c >= 2]
    best = None
    for i in cand_idx:
        t = live[i]
        if t[1] == r["path"]:
            continue
        tt, ts = norm(t[2]), stem(t[1].split("/")[-1])
        au = toks(t[3])
        a_ok = (not mya) or (not au) or bool(mya & au)
        if not a_ok:
            continue
        s = 0
        if myt and tt and len(min(myt, tt, key=len)) >= 6 and (myt in tt or tt in myt):
            s = 3   # 标题包含
        if mystem and ts and len(min(mystem, ts, key=len)) >= 8 and (mystem in ts or ts in mystem):
            s = max(s, 2)   # 文件名包含
        if s and (best is None or s > best[0] or (s == best[0] and len(t[2]) > len(best[1][2]))):
            best = (s, t)
    if best:
        pairs.append({"id": idmap[r["path"]], "old_path": r["path"], "target_path": best[1][1],
                      "标题": r["title"], "作者": r["authors"],
                      "理由": f"plan8 {'标题包含' if best[0]==3 else '文件名包含'}；目标正文 {best[1][4]} 段"})
        info.append((r, best[1], best[0]))
print(f"命中 {len(pairs)} 行")
for r, t, s in info[:8]:
    print(f"   [{'标题' if s==3 else '文件名'}] {r['file'][:46]}\n        → {t[1][:70]} ({t[4]} 段)")
with open(P + "plan8_pairs.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "old_path", "target_path", "标题", "作者", "理由"], delimiter="\t")
    w.writeheader()
    for x in pairs:
        w.writerow(x)
print("已写 plan8_pairs.tsv")
