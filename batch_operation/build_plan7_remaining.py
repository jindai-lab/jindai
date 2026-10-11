#!/usr/bin/env python3
"""剩余「文件不在盘」统一清理：
1) 盘上候选（同名/#截断/重复后缀双向/CompArt/Zotero 镜像）
   - 候选未注册 → 改路径
   - 候选已注册 → 重复记录：目标有正文→删本行；目标无正文→删目标、本行改指目标路径
2) 找不到盘上文件时，按 标题+作者 找库内「同一部作品」的存活记录 → 同上 (1)(2) 逻辑
输出 plan7_repath.tsv / plan7_delete_self.tsv / plan7_delete_target.tsv
"""
import csv, os, json, re, subprocess, collections, difflib

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
STORE = "/storage"          # 容器内 storage 根
HOST = "/misc/Public/jindai-storage"

rem = list(csv.DictReader(open(P + "remaining_final.tsv", encoding="utf-8"), delimiter="\t"))
idx = collections.defaultdict(list)
for l in open(P + "disk_index.jsonl", encoding="utf-8"):
    d = json.loads(l)
    idx[d["b"].lower()].append(d["p"])
# 补充索引：CompArt + OneDrive/zotero 镜像（不在 jindai-storage 下，单独扫）
extra = {}
for tree in ["/misc/Public/OneDrive/zotero", "/misc/Public/OneDrive/Zotero"]:
    if os.path.isdir(tree):
        for f in os.listdir(tree):
            extra.setdefault(f.lower(), []).append("OneDrive/zotero/" + f)


def norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())


def variants(fn):
    stem, ext = os.path.splitext(fn)
    vs = {fn}
    if "#" in fn:
        vs.add(fn.split("#")[0])
    if stem.lower().endswith(ext.lower()):
        vs.add(stem)
    vs.add(fn + ext)
    if re.search(r"（中文版）|\(中文版\)|中文版", stem):
        vs.add(re.sub(r"[（(]?中文版[)）]?", "", fn))
    return {v for v in vs if v}


# 库内存活记录（文件在盘）+ 全库标题索引
dbrows = [l.split("\t") for l in subprocess.run(
    ["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai", "-A", "-t", "-F", "\t", "-c",
     "SELECT id, path, coalesce(title,''), coalesce(array_to_string(authors,'|'),''), "
     "(SELECT count(*) FROM paragraph p WHERE p.source_id=f.id) FROM file_metadata f"],
    capture_output=True, text=True).stdout.splitlines() if l.count("\t") == 4]
bypath = {r[1]: r for r in dbrows}
bytitle = collections.defaultdict(list)
for r in dbrows:
    if r[2]:
        bytitle[norm(r[2])].append(r)
print(f"库内 {len(dbrows)} 行 | 待处理 {len(rem)}")

repath, del_self, del_target, unresolved = [], [], [], []
for r in rem:
    p0, fn = r["path"], r["file"]
    cands = []
    for v in variants(fn):
        cands += idx.get(v.lower(), [])
        cands += extra.get(v.lower(), [])
    cands = [c for c in dict.fromkeys(cands) if not os.path.exists(os.path.join(HOST, p0))]
    # CompArt 明确规则
    if not cands and "COMPUTER_GRAPHICS_AND_ART" in fn:
        cands = idx.get((fn + os.path.splitext(fn)[1]).lower(), [])
    if cands:
        # 同顶层优先
        top = p0.split("/")[0]
        same = [c for c in cands if c.split("/")[0] == top]
        c = (same or cands)[0] if len(same or cands) == 1 else (sorted(same or cands)[0] if same else None)
        if c:
            if c not in bypath:
                repath.append((r, c, "盘上同名/变体命中"))
            else:
                t = bypath[c]
                if int(t[4]) > 0:
                    del_self.append((r, c, t, "目标已有正文→删本行"))
                else:
                    del_target.append((r, c, t, "目标无正文→删目标、本行改指"))
            continue
    # 无盘上文件 → 找同作品存活记录
    cand = None
    for t in bytitle.get(norm(r["title"]), []):
        if t[1] == p0:
            continue
        au = set(filter(None, re.split(r"[|,、]", r["authors"])))
        tu = set(filter(None, re.split(r"[|,、]", t[3])))
        if not au or (au & tu) or norm(t[2]) != norm(r["title"]):
            cand = t
            break
    if cand and os.path.exists(os.path.join(HOST, cand[1])):
        if int(cand[4]) > 0:
            del_self.append((r, cand[1], cand, "同作品已在库且有正文→删本行"))
        else:
            del_target.append((r, cand[1], cand, "同作品在库但无正文→删目标、本行改指"))
    else:
        unresolved.append(r)
print(f"改路径 {len(repath)} | 删本行 {len(del_self)} | 删目标+改指 {len(del_target)} | 仍无解 {len(unresolved)}")


def w(f, head, rows):
    with open(P + f, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(head)
        wr.writerows(rows)


w("plan7_repath.tsv", ["old_path", "new_path", "file", "title", "authors", "规则"],
  [[r["path"], c, r["file"], r["title"], r["authors"], why] for r, c, why in repath])
w("plan7_delete_self.tsv", ["old_path", "target_path", "target_has_text", "file", "title", "rules"],
  [[r["path"], c, t[4], r["file"], r["title"], why] for r, c, t, why in del_self])
w("plan7_delete_target.tsv", ["old_path", "target_path", "target_id", "file", "title", "规则"],
  [[r["path"], c, t[0], r["file"], r["title"], why] for r, c, t, why in del_target])
w("plan7_unresolved.tsv", ["path", "file", "title", "authors"], [[r["path"], r["file"], r["title"], r["authors"]] for r in unresolved])
print("已写 plan7_*.tsv")
for r, c, why in repath[:6]:
    print(f"   [改路径] {r['file'][:44]}\n            → {c[:76]}")
for r, c, t, why in del_self[:4]:
    print(f"   [删本行] {r['file'][:44]}  (目标 {t[4]} 段)")
