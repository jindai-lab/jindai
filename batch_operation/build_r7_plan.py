#!/usr/bin/env python3
"""R7：作者改名导致的整体迁移匹配（跨目录「前缀+后缀同时吻合且唯一」）
输出 path_fix_plan3.tsv
"""
import csv, json, os, re, subprocess, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))
done = set()
for f in ["path_fix_plan.tsv", "path_fix_plan2.tsv", "path_fix_dups.tsv"]:
    for r in csv.DictReader(open(P + f, encoding="utf-8"), delimiter="\t"):
        done.add(r.get("old_path") or r.get("id") or "")
rows = [r for r in rows if r["path"] not in done]
print("本轮待匹配", len(rows))

sufidx = collections.defaultdict(list)   # 末 12 字符(小写) -> [rel]
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    b = d["b"].lower()
    if len(b) >= 12:
        sufidx[b[-12:]].append(d["p"])

out = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t", "-c",
                      "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout
taken = set(out.splitlines())

plan, amb = [], 0
for r in rows:
    base = r["file"].lower()
    if len(base) < 28:
        continue
    pre, suf = base[:15], base[-12:]
    cands = [p for p in sufidx.get(suf, []) if os.path.basename(p).lower().startswith(pre)]
    if len(cands) == 1:
        newp = cands[0]
        if newp in taken:
            continue
        plan.append([r["path"], newp, "R7 作者改名整体迁移(前后缀吻合)", r["title"], r["authors"],
                     r["file"], os.path.basename(newp)])
        taken.add(newp)
    elif len(cands) > 1:
        amb += 1
print(f"R7 可修 {len(plan)} 行 | 多候选 {amb} 行")

idmap = {}
paths = [p[0] for p in plan]
for i in range(0, len(paths), 400):
    chunk = paths[i:i + 400]
    inlist = ",".join("'" + x.replace("'", "''") + "'" for x in chunk)
    if not inlist:
        continue
    q = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai",
                        "-A", "-t", "-F", "\t", "-c",
                        f"SELECT id, path FROM file_metadata WHERE path IN ({inlist})"],
                       capture_output=True, text=True).stdout
    for l in q.splitlines():
        if "\t" in l:
            a, b = l.split("\t", 1)
            idmap.setdefault(b, a)
with open(P + "path_fix_plan3.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for p in plan:
        w.writerow([idmap.get(p[0], "")] + p)
print("已写", P + "path_fix_plan3.tsv", f"（取到 id {len(idmap)}）")
for p in plan[:8]:
    print(f"   {p[0][:58]}\n     → {p[1][:74]}")
