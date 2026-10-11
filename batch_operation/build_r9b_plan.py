#!/usr/bin/env python3
"""R9b：Authors/ → OneDrive/<作者>/<书目录>/<文件>：归一化后「前缀包含」且唯一。
正文文件取书目录内非 cover/metadata 的那个（若多个，取与库内作者名最贴近的）。
输出 path_fix_plan6.tsv
"""
import csv, json, os, re, subprocess, collections, difflib

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rem = [r for r in csv.DictReader(open(P + "remaining_final.tsv", encoding="utf-8"), delimiter="\t")
       if r["path"].startswith("Authors/")]
print("Authors/ 待匹配", len(rem))
ACC = {"cover.jpg", "cover.jpeg", "cover.png", "metadata.opf", "metadata 1.opf", "urls.txt"}


def toks(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s.lower()).split()


def nk(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())


bookfiles = collections.defaultdict(list)   # 作者词集 -> [(书目录, 文件)]
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    parts = d["p"].split("/")
    if len(parts) >= 4 and parts[0] == "OneDrive":
        bookfiles[frozenset(toks(parts[1]))].append((parts[2], d["b"], d["p"]))

plan, amb, nomatch = [], 0, 0
for r in rem:
    p0, base = r["path"], r["file"]
    ak = frozenset(toks(p0.split("/")[1]))
    sur = sorted(ak, key=len)[-1] if ak else ""
    pool = [x for k, v in bookfiles.items() if ak and ak <= k and sur in k for x in v]
    if not pool:
        nomatch += 1
        continue
    stem = nk(os.path.splitext(base)[0])
    stem = re.sub(r"(theodor|adorno|gregory|bateson)$", "", stem)   # 去掉尾部作者名干扰
    hits = collections.defaultdict(float)
    for bookdir, fn, full in pool:
        nb = nk(bookdir)
        if len(stem) >= 12 and (nb.startswith(stem[:12]) or stem.startswith(nb[:12])):
            score = difflib.SequenceMatcher(None, stem, nb).ratio()
            hits[bookdir] = max(hits[bookdir], score)
    if not hits:
        nomatch += 1
        continue
    top = sorted(hits.items(), key=lambda x: -x[1])
    if len(top) > 1 and top[0][1] - top[1][1] < 0.05:
        amb += 1
        continue
    bookdir, score = top[0]
    files = [x for x in pool if x[0] == bookdir and x[1] not in ACC]
    if not files:
        nomatch += 1
        continue
    # 多个正文文件时取与库内文件名最像的
    files.sort(key=lambda x: -difflib.SequenceMatcher(None, nk(base), nk(x[1])).ratio())
    _, fn, full = files[0]
    plan.append([p0, full, f"R9b 作者目录+书名前缀吻合({score:.2f})", r["title"], r["authors"], base, fn])

db = set(subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t",
                         "-c", "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout.splitlines())
dup = sum(1 for x in plan if x[1] in db)
print(f"R9b 命中 {len(plan)} 行（目标已在库→重复 {dup}）| 多候选 {amb} | 无匹配 {nomatch}")
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
    print(f"   {x[5][:56]}\n     → {x[1][:88]}")
