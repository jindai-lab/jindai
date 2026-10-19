#!/usr/bin/env python3
"""用盘上文件名索引，为缺失行做跨目录定位（唯一命中才采纳，并避开已被占用的 path）。"""
import csv, json, os, re, subprocess, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
S = "/misc/Public/jindai-storage"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))

# 索引：basename -> [rel...]；归一化 -> [rel...]
idx = collections.defaultdict(list)
nidx = collections.defaultdict(list)
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    idx[d["b"]].append(d["p"])
    nidx[re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", d["b"].lower())].append(d["p"])
print(f"索引载入 {sum(len(v) for v in idx.values())} 个文件")

# 已被占用的 path（避免唯一约束冲突）
out = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t", "-c",
                      "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout
taken = set(out.splitlines())
print(f"库内已占用 path {len(taken)} 条")

plan, dup, ambiguous = [], [], 0
for r in rows:
    base = r["file"]
    cands = []
    # ① 同名；② 去 #UUID；③ 换扩展名后的同名；④ 归一化同名；⑤ 去重复扩展名
    keys = [base, re.sub(r"#[0-9a-fA-F-]{8,}$", "", base)]
    for k in list(keys):
        b, e = os.path.splitext(k)
        b2, e2 = os.path.splitext(b)
        if e2 and e2.lower() == e.lower():
            keys.append(b)
    for k in keys:
        if idx.get(k):
            cands = idx[k]
            rule = "跨目录同名"
            break
    if not cands:
        nk = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", base.lower())
        if nidx.get(nk):
            cands = nidx[nk]
            rule = "跨目录归一化同名"
    if not cands:
        continue
    if len(cands) != 1:
        ambiguous += 1
        continue
    newp = cands[0]
    if newp in taken:
        dup.append([r["path"], newp, r["title"], r["authors"]])
        continue
    plan.append([r["path"], newp, rule, r["title"], r["authors"], base, os.path.basename(newp)])
    taken.add(newp)

print(f"跨目录可修 {len(plan)} 行 | 目标已被别的行占用（重复）{len(dup)} 行 | 多候选不唯一 {ambiguous} 行")
with open(P + "path_fix_plan_index.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    old_paths = [p[0] for p in plan]
    idmap = {}
    for i in range(0, len(old_paths), 400):
        chunk = old_paths[i:i + 400]
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
    for p in plan:
        w.writerow([idmap.get(p[0], "")] + p)
print("已写", P + "path_fix_plan_index.tsv")
with open(P + "path_fix_dups.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["old_path", "existing_path", "title", "authors"])
    for d in dup:
        w.writerow(d)
