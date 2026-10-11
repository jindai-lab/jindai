#!/usr/bin/env python3
"""最终分类：对仍未处理的缺失行，用全部规则找目标路径，分成
A 可改路径（目标未占用） B 重复记录（目标已在库 → 删本行） C 无目标（人工）
输出 path_fix_final.tsv / path_delete_final.tsv / path_unmatched.tsv
"""
import csv, json, os, re, subprocess, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))

done = set()
for f in ["path_fix_plan.tsv", "path_fix_plan2.tsv", "path_fix_plan3.tsv", "path_fix_plan4.tsv", "path_fix_dups.tsv"]:
    try:
        for r in csv.DictReader(open(P + f, encoding="utf-8"), delimiter="\t"):
            done.add(r.get("old_path") or "")
    except FileNotFoundError:
        pass
rows = [r for r in rows if r["path"] not in done]
print("待分类", len(rows))

STOPW = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "de", "la", "le", "des", "der"}


def toks(s):
    return [t for t in re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", s).lower().split() if t]


def resid(name, author):
    name = os.path.splitext(name)[0]
    a = collections.Counter(toks(author))
    r = []
    for t in toks(name):
        if a[t] > 0:
            a[t] -= 1
        else:
            r.append(t)
    return tuple(sorted(r))


idx = collections.defaultdict(list)
nidx = collections.defaultdict(list)
stem = collections.defaultdict(list)
residx = collections.defaultdict(list)
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    b, p = d["b"], d["p"]
    idx[b].append(p)
    nidx[re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", b.lower())].append(p)
    stem[os.path.splitext(b)[0].lower()].append(p)
    parts = p.split("/")
    if len(parts) >= 3 and parts[0] in ("OneDrive", "Authors", "Series", "Zotero"):
        k = resid(b, parts[1])
        if len(k) >= 2:
            residx[k].append(p)

dbpaths = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai",
                          "-A", "-t", "-c", "SELECT path FROM file_metadata"],
                         capture_output=True, text=True).stdout.splitlines()
taken = set(dbpaths)
print("库内 path", len(taken))

B = collections.defaultdict(list)
for r in rows:
    base = r["file"]
    cands = idx.get(base) or []
    rule = "R1 跨目录同名"
    if len(cands) != 1:
        c2 = nidx.get(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", base.lower())) or []
        if len(c2) == 1:
            cands, rule = c2, "R1' 归一化同名"
        else:
            c3 = stem.get(os.path.splitext(base)[0].lower()) or []
            if len(c3) == 1:
                cands, rule = c3, "R2 换扩展名"
            else:
                parts = r["path"].split("/")
                if len(parts) >= 3 and parts[0] in ("OneDrive", "Authors", "Series", "Zotero"):
                    k = resid(base, parts[1])
                    c4 = residx.get(k) or []
                    if len(c4) == 1:
                        cands, rule = c4, "R8 作者改名/换位迁移"
    B[rule if len(cands) == 1 else "C 无目标（人工）"].append((r, cands[0] if len(cands) == 1 else ""))

fix, dele, un = [], [], []
for rule, lst in B.items():
    for r, tgt in lst:
        if not tgt:
            un.append([r["path"], r["file"], r["title"], r["authors"], "无匹配"])
            continue
        if tgt in taken:
            dele.append([r["path"], tgt, r["title"], r["authors"], rule + "（目标已在库→重复记录）"])
        else:
            fix.append([r["path"], tgt, rule, r["title"], r["authors"], r["file"], os.path.basename(tgt)])
            taken.add(tgt)
print(f"A 可改路径 {len(fix)} | B 重复记录(待删) {len(dele)} | C 人工 {len(un)}")
print("规则分布:", dict(collections.Counter([f[2] for f in fix] + [d[4] for d in dele])))


def idmap(paths):
    m = {}
    for i in range(0, len(paths), 400):
        chunk = paths[i:i + 400]
        inlist = ",".join("'" + x.replace("'", "''") + "'" for x in chunk)
        if not inlist:
            continue
        q = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai",
                            "-A", "-t", "-F", "\t", "-c", f"SELECT id, path FROM file_metadata WHERE path IN ({inlist})"],
                           capture_output=True, text=True).stdout
        for l in q.splitlines():
            if "\t" in l:
                a, b = l.split("\t", 1)
                m.setdefault(b, a)
    return m


mp = idmap([f[0] for f in fix] + [d[0] for d in dele] + [u[0] for u in un])
with open(P + "path_fix_final.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for x in fix:
        w.writerow([mp.get(x[0], "")] + x)
with open(P + "path_delete_final.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "target_path", "标题", "作者", "理由"])
    for x in dele:
        w.writerow([mp.get(x[0], "")] + x)
with open(P + "path_unmatched.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "file", "标题", "作者", "原因"])
    for x in un:
        w.writerow([mp.get(x[0], "")] + x)
print("已写 path_fix_final.tsv / path_delete_final.tsv / path_unmatched.tsv")
