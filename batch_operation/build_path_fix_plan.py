#!/usr/bin/env python3
"""严格匹配生成路径修复计划（宁缺毋滥：每条必须唯一命中且可解释）。"""
import csv, os, json, re, sys, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
S = "/misc/Public/jindai-storage"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))

EXT = [".pdf", ".epub", ".djvu", ".docx", ".md", ".txt", ".html", ".jpg", ".png", ".pdb"]


def norm(s):
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s.lower())


def dbl(s):
    b, e = os.path.splitext(s)
    b2, e2 = os.path.splitext(b)
    return e2.lower() == e.lower() and e != "" and b2 != ""


cache = {}


def listing(d):
    if d not in cache:
        try:
            cache[d] = os.listdir(d)
        except Exception:
            cache[d] = []
    return cache[d]


plan = []
for r in rows:
    if r["父目录存在"] != "是":
        continue
    d, base = r["父目录"], r["file"]
    # ⓪ 库内路径尾部带 #UUID 片段 → 先去掉再用后续规则
    stripped = re.sub(r"#[0-9a-fA-F-]{8,}$", "", base)
    # ⓪-b 去掉 #UUID 后盘上就有同名文件 → 直接修
    if stripped != base:
        files0 = listing(d)
        if stripped in files0:
            plan.append([r["path"], r["path"].rsplit("/", 1)[0] + "/" + stripped, "去 #UUID 尾缀",
                         r["title"], r["authors"], base, stripped])
            continue
    files = listing(d)
    cand, rule = "", ""
    # 1 完全同名
    if base in files:
        cand, rule = base, "同名"
    # 2 忽略大小写
    if not cand:
        m = [f for f in files if f.lower() == base.lower()]
        if len(m) == 1:
            cand, rule = m[0], "忽略大小写同名"
    # 3 去扩展名重复（x.epub.epub → x.epub）
    if not cand and dbl(base):
        stem = os.path.splitext(base)[0]
        m = [f for f in files if f == stem or norm(f) == norm(stem)]
        if len(m) == 1:
            cand, rule = m[0], "重复扩展名"
    # 4 补扩展名
    if not cand:
        m = [f for f in files for e in EXT if f == base + e]
        if len(m) == 1:
            cand, rule = m[0], "补扩展名"
    # 5 归一化（去空格/标点/大小写）一致
    if not cand:
        m = [f for f in files if norm(f) == norm(base)]
        if len(m) == 1:
            cand, rule = m[0], "归一化一致"
    # 6 库内被截断：盘上以库内名为前缀（≥20 字符，唯一）
    if not cand and len(base) >= 20:
        m = [f for f in files if f.startswith(base)]
        if len(m) == 1:
            cand, rule = m[0], "库内名被截断"
    # 7 中间被截断：前缀(≥15)+后缀(≥10) 同时吻合且唯一
    if not cand and len(base) >= 30:
        pre, suf = base[:15], base[-12:]
        m = [f for f in files if f.startswith(pre) and f.endswith(suf)]
        if len(m) == 1:
            cand, rule = m[0], "库内名中间截断"
    if cand:
        plan.append([r["path"], r["path"].rsplit("/", 1)[0] + "/" + cand, rule, r["title"], r["authors"],
                     r["file"], cand])
        r["_fixed"] = "1"

print(f"可严格修复 {len(plan)} 行 / 目录存在的 {sum(1 for r in rows if r['父目录存在']=='是')} 行")
c2 = collections.Counter(p[2] for p in plan)
print("规则分布:", dict(c2))
with open(P + "path_fix_plan.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
# 补 id（用 path 反查，供带护栏更新）
import subprocess
old_paths = [p[0] for p in plan]
idmap = {}
for i in range(0, len(old_paths), 500):
    chunk = old_paths[i:i+500]
    inlist = ",".join("'" + x.replace("'", "''") + "'" for x in chunk)
    out = subprocess.run(["docker","exec","jindai-pgvector","psql","-U","myuser","-d","jindai","-A","-t","-F","\t","-c",
                          f"SELECT id, path FROM file_metadata WHERE path IN ({inlist})"], capture_output=True, text=True).stdout
    for l in out.splitlines():
        if "\t" in l:
            a, b = l.split("\t", 1)
            idmap.setdefault(b, a)
with open(P + "path_fix_plan.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "old_path", "new_path", "规则", "标题", "作者", "库内文件名", "盘上文件名"])
    for p in plan:
        w.writerow([idmap.get(p[0], "")] + p)
print(f"已写 {P}path_fix_plan.tsv（{len(plan)} 行，取到 id {len(idmap)} 个）")
