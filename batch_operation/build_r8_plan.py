#!/usr/bin/env python3
"""R8：作者改名/换位迁移匹配——去掉作者名后「剩余词集」完全相同（处理 A-Title ↔ Title-A）"""
import csv, json, os, re, subprocess, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))
done = set()
for f in ["path_fix_plan.tsv", "path_fix_plan2.tsv", "path_fix_plan3.tsv", "path_fix_dups.tsv"]:
    for r in csv.DictReader(open(P + f, encoding="utf-8"), delimiter="\t"):
        done.add(r.get("old_path") or "")
rows = [r for r in rows if r["path"] not in done]
print("本轮待匹配", len(rows))

STOP = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "de", "la", "le", "des", "der", "die", "das", "und", "et"}


def toks(s):
    return [t for t in re.sub(r"[^0-9A-Za-zÀ-ÿ\u4e00-\u9fff]+", " ", s).lower().split() if t]


def resid(name, author):
    name = os.path.splitext(name)[0]          # 扩展名不参与分词
    a = collections.Counter(toks(author))
    r = []
    for t in toks(name):
        if a[t] > 0:
            a[t] -= 1
        else:
            r.append(t)
    return tuple(sorted(t for t in r))


residx = collections.defaultdict(list)
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    parts = d["p"].split("/")
    if len(parts) >= 3 and parts[0] in ("OneDrive", "Authors", "Series", "Zotero"):
        author = parts[1]
        key = resid(d["b"], author)
        if len(key) >= 2:
            residx[key].append(d["p"])
print("剩余词集索引键", len(residx))

out = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t", "-c",
                      "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout
taken = set(out.splitlines())

plan, multi = [], 0
for r in rows:
    parts = r["path"].split("/")
    if len(parts) < 3 or parts[0] not in ("OneDrive", "Authors", "Series", "Zotero"):
        continue
    author = parts[1]
    key = resid(r["file"], author)
    if len(key) < 2:
        continue
    cands = residx.get(key) or []
    if len(cands) == 1:
        newp = cands[0]
        if newp in taken:
            continue
        plan.append([r["path"], newp, "R8 作者改名/换位迁移(余词集相同)", r["title"], r["authors"],
                     r["file"], os.path.basename(newp)])
        taken.add(newp)
    elif len(cands) > 1:
        multi += 1
print(f"R8 可修 {len(plan)} 行 | 多候选(放弃) {multi} 行")

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
with open(P + "path_fix_plan4.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for p in plan:
        w.writerow([idmap.get(p[0], "")] + p)
print("已写", P + "path_fix_plan4.tsv", f"（取到 id {len(idmap)}）")
for p in plan[:8]:
    print(f"   {p[5][:56]}\n     → {p[1][:78]}")
