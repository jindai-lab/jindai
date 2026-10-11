#!/usr/bin/env python3
"""收尾：① 定点修复；② 对剩余行做 ≥0.85 且唯一的相似匹配（跨目录），生成 plan5。"""
import csv, json, os, re, subprocess, collections, difflib

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
S = "/misc/Public/jindai-storage"
rest = json.load(open(P + "remaining_missing.json", encoding="utf-8"))
audit = {r["path"]: r for r in csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t")}

idx = collections.defaultdict(list)
allp = []
lenbucket = collections.defaultdict(list)
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    idx[d["b"].lower()].append(d["p"])
    allp.append((d["b"].lower(), d["p"]))
    bl = len(d["b"])
    for L in range(max(1, int(bl / 1.6) - 1), int(bl / 0.62) + 3):
        lenbucket[L].append((d["b"].lower(), d["p"]))
db = set(subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t",
                         "-c", "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout.splitlines())

plan, amb = [], 0
for p0 in rest:
    r = audit.get(p0)
    if not r:
        continue
    base = r["file"].lower()
    if not base:
        continue
    parts = p0.split("/")
    author = parts[1].lower() if len(parts) >= 3 else ""
    best = []
    lo, hi = int(len(base) * 0.62), int(len(base) * 1.6) + 2
    for k, cand in lenbucket.get(len(base), []):
        if author and author.split(",")[0].strip() and author.split(",")[0].strip() not in cand.lower():
            # 作者姓氏必须出现在候选路径里（跨目录迁移的强约束）
            if not any(t in cand.lower() for t in re.findall(r"[a-z\u4e00-\u9fff]{2,}", author)[:2]):
                continue
        s = difflib.SequenceMatcher(None, base, k).ratio()
        if s >= 0.85:
            best.append((s, cand))
    if not best:
        continue
    best.sort(reverse=True)
    if len(best) > 1 and abs(best[0][0] - best[1][0]) < 0.02:
        amb += 1
        continue
    s, cand = best[0]
    if cand in db:
        continue
    plan.append([p0, cand, f"相似 {s:.2f} 唯一候选", r["title"], r["authors"], r["file"], os.path.basename(cand)])
    db.add(cand)
print(f"收尾可修 {len(plan)} 行 | 分数接近放弃 {amb} 行")
idmap = {}
pp = [x[0] for x in plan]
for i in range(0, len(pp), 400):
    ch = pp[i:i + 400]
    inlist = ",".join("'" + x.replace("'", "''") + "'" for x in ch)
    if not inlist:
        continue
    out = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t",
                          "-F", "\t", "-c", f"SELECT id, path FROM file_metadata WHERE path IN ({inlist})"],
                         capture_output=True, text=True).stdout
    for l in out.splitlines():
        if "\t" in l:
            a, b = l.split("\t", 1)
            idmap.setdefault(b, a)
with open(P + "path_fix_plan5.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for x in plan:
        w.writerow([idmap.get(x[0], "")] + x)
print("已写 path_fix_plan5.tsv", f"(取到 id {len(idmap)})")
for x in plan[:10]:
    print(f"   {x[5][:52]}\n     → {x[1][:80]}")
