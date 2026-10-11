#!/usr/bin/env python3
"""按用户给出的迁移规则做路径修复计划：
R1 跨目录同名（含 Albums/Vasulka Archive → Series/Vasulka Archive）
R2 换扩展名（东方杂志 .djvu → 盘上 .pdf）
R3 作者改名导致的整体迁移（Tree/<旧作者>/<余下路径> → Tree/<新作者>/<同样余下路径>）
R4 时间戳冗余文件名（形如 x_2026-08-23T154007.594292Z.pdf）→ 若存在无时间戳的同名文件，标记删除整条记录
输出：path_fix_plan2.tsv（改路径）、path_delete_plan.tsv（待删记录）
"""
import csv, json, os, re, subprocess, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
S = "/misc/Public/jindai-storage"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))

idx = collections.defaultdict(list)          # basename -> [rel]
nidx = collections.defaultdict(list)         # 归一化 basename -> [rel]
stem = collections.defaultdict(list)         # 去扩展名 stem(小写) -> [rel]
suf2 = collections.defaultdict(list)         # 末两段(小写) -> [rel]
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    b, p = d["b"], d["p"]
    idx[b].append(p)
    nidx[re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", b.lower())].append(p)
    stem[os.path.splitext(b)[0].lower()].append(p)
    parts = p.split("/")
    suf2["/".join(parts[-2:]).lower()].append(p)
print(f"索引 {sum(len(v) for v in idx.values())} 文件 | 末两段键 {len(suf2)}")

out = subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t", "-c",
                      "SELECT path FROM file_metadata"], capture_output=True, text=True).stdout
taken = set(out.splitlines())

TS = re.compile(r"[_\-\s]?\d{4}-\d{2}-\d{2}T\d{6}(?:\.\d+)?Z")
plan, dele, dup, amb = [], [], [], 0


def pick(cands, rule):
    if not cands:
        return None, ""
    if len(cands) == 1:
        return cands[0], rule
    return None, ""


for r in rows:
    base, p0 = r["file"], r["path"]
    parts = p0.split("/")
    # R4 时间戳冗余
    if TS.search(base):
        clean = TS.sub("", base)
        raw = base.replace(TS.search(base).group(), "")
        cands = idx.get(clean) or idx.get(raw) or []
        if len(cands) == 1:
            dele.append([p0, cands[0], r["title"], r["authors"], "时间戳冗余，已有无时间戳版本"])
            continue
    cands = idx.get(base) or []
    rule = "R1 跨目录同名"
    if not cands:
        cands = nidx.get(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", base.lower())) or []
        rule = "R1' 跨目录归一化同名"
    if not cands:
        cands = stem.get(os.path.splitext(base)[0].lower()) or []
        rule = "R2 换扩展名（djvu→pdf 等）"
    if not cands:
        # R3 作者改名：末两段相同
        if len(parts) >= 3:
            key = "/".join(parts[-2:]).lower()
            cands = suf2.get(key) or []
            rule = "R3 作者改名迁移(末两段吻合)"
    if not cands:
        continue
    if len(cands) > 1:
        amb += 1
        continue
    newp = cands[0]
    if newp in taken:
        dup.append([p0, newp, r["title"], r["authors"], rule])
        continue
    plan.append([p0, newp, rule, r["title"], r["authors"], base, os.path.basename(newp)])
    taken.add(newp)

print(f"可改路径 {len(plan)} 行 | 待删(时间戳冗余) {len(dele)} 行 | 目标已占用(重复) {len(dup)} 行 | 多候选 {amb} 行")
print("规则分布:", dict(collections.Counter(p[2] for p in plan)))


def with_id(paths, header_extra):
    idmap = {}
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
    return idmap


idmap = with_id([p[0] for p in plan] + [d[0] for d in dele], [])
with open(P + "path_fix_plan2.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for p in plan:
        w.writerow([idmap.get(p[0], "")] + p)
with open(P + "path_delete_plan.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "对照(盘上无时间戳版本)", "标题", "作者", "理由"])
    for d in dele:
        w.writerow([idmap.get(d[0], "")] + d)
with open(P + "path_fix_dups.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["old_path", "existing_path", "标题", "作者", "规则"])
    for d in dup:
        w.writerow(d)
print("已写 path_fix_plan2.tsv / path_delete_plan.tsv / path_fix_dups.tsv")
