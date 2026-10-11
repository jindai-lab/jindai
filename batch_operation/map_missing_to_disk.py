#!/usr/bin/env python3
"""对父目录存在的缺失行，在盘上同目录做相似度匹配，给出疑似对应文件。"""
import csv, os, difflib, collections

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))
cache = {}


def listing(d):
    if d not in cache:
        try:
            cache[d] = os.listdir(d)
        except Exception:
            cache[d] = []
    return cache[d]


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


hit = 0
for r in rows:
    r["疑似对应文件"] = ""
    r["相似度"] = ""
    if r["父目录存在"] != "是":
        continue
    d, base = r["父目录"], r["file"]
    best, score = "", 0.0
    for f in listing(d):
        s = difflib.SequenceMatcher(None, norm(base), norm(f)).ratio()
        if s > score:
            best, score = f, s
    if score >= 0.62 and best:
        r["疑似对应文件"] = best
        r["相似度"] = f"{score:.2f}"
        hit += 1

# 重写 TSV（带两列新字段）
cols = list(rows[0].keys())
with open(P + "disk_missing_audit.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"完成：{len(rows)} 行，{hit} 行给出疑似对应文件（父目录存在的行中）")
