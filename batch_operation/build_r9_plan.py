#!/usr/bin/env python3
"""R9：Authors/ → OneDrive/<作者>/<书名 (编号)>/<文件> 迁移匹配。
同作者（词集任意序相同）目录下，书目录名与库内文件名高相似且唯一 → 命中最贴近的那个书目录里的文件。
输出 path_fix_plan6.tsv
"""
import csv, json, os, re, subprocess, collections, difflib

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rem = list(csv.DictReader(open(P + "remaining_final.tsv", encoding="utf-8"), delimiter="\t"))
rem = [r for r in rem if r["path"].startswith("Authors/")]
print("Authors/ 待匹配", len(rem))


def toks(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s.lower()).split()


def nk(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())


# 盘上结构：OneDrive/<作者>/<书目录>/<文件>
books = collections.defaultdict(list)   # 作者词集 -> [(书目录名, [文件...])]
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    parts = d["p"].split("/")
    if len(parts) >= 4 and parts[0] == "OneDrive":
        author = frozenset(toks(parts[1]))
        books[author].append(d["p"])
print("盘上作者词集组", len(books))


def author_key(path):
    a = path.split("/")[1]
    return frozenset(toks(a))


plan, amb, nomatch = [], 0, 0
for r in rem:
    p0 = r["path"]
    base = r["file"]
    ak = author_key(p0)
    cands = books.get(ak) or []
    if not cands:
        # 允许子集：库内作者词 ⊂ 盘上作者词（盘上常多中间名缩写），且必须共享姓氏
        sur = sorted(ak, key=len)[-1] if ak else ""
        cands = [p for k, v in books.items() if ak and ak <= k and sur in k for p in v]
    if not cands:
        nomatch += 1
        continue
    best = []
    for cand in cands:
        bookdir = cand.split("/")[2] if len(cand.split("/")) > 2 else ""
        s = difflib.SequenceMatcher(None, nk(base), nk(bookdir)).ratio()
        best.append((s, cand))
    best.sort(reverse=True)
    if not best:
        nomatch += 1
        continue
    s, cand = best[0]
    if s < 0.72:
        nomatch += 1
        continue
    if len(best) > 1 and abs(best[0][0] - best[1][0]) < 0.04:
        amb += 1
        continue
    plan.append([p0, cand, f"R9 作者目录+书名吻合({s:.2f})", r["title"], r["authors"], base, os.path.basename(cand)])

db = set(subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t",
                         "-c", "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout.splitlines())
dup = sum(1 for x in plan if x[1] in db)
print(f"R9 命中 {len(plan)} 行（其中目标已在库 {dup} → 属重复记录）| 多候选 {amb} | 无匹配 {nomatch}")
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
with open(P + "path_fix_plan6.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for x in plan:
        w.writerow([idmap.get(x[0], "")] + x)
print("已写 path_fix_plan6.tsv", f"(取到 id {len(idmap)})")
for x in plan[:8]:
    print(f"   {x[5][:56]}\n     → {x[1][:84]}")
