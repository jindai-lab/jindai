#!/usr/bin/env python3
"""把 disk_missing_audit.tsv 汇总成给用户的可读报告（Markdown）。"""
import csv, os, collections, sys

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))
STORE = "/misc/Public/jindai-storage"


def top(p):
    p = p[7:] if p.startswith("../app/") else p
    return p.split("/")[0] or "(空)"


by = collections.defaultdict(list)
for r in rows:
    by[top(r["path"])].append(r)

# 顶层目录在盘上是否存在
exists = {t: os.path.isdir(os.path.join(STORE, t)) for t in by}
tot_dir = sum(1 for r in rows if r["父目录存在"] == "是")

out = []
out.append("# jindai 文库：记录在库但本地文件不在盘的清单\n")
out.append(f"- 盘点范围：`file_metadata` 全库 274,000 行（逐行 `os.path.exists` 实测）")
out.append(f"- 命中「本地路径但文件不在盘」：**{len(rows)}** 行")
out.append(f"- 其中父目录存在（文件被删/改名）：**{tot_dir}** 行；父目录也不存在（整棵目录树缺失）：**{len(rows)-tot_dir}** 行\n")
out.append("## 按顶层目录分布\n")
out.append("| 顶层目录 | 行数 | 该目录是否在盘 |")
out.append("|---|---:|---|")
for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    out.append(f"| `{t}/` | {len(rs)} | {'是' if exists.get(t) else '**否**'} |")
out.append("")
for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    out.append(f"\n## {t}/（{len(rs)} 行）\n")
    out.append("| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |")
    out.append("|---:|---|---|---|---|---|")
    for i, r in enumerate(rs, 1):
        f = lambda s: (s or "").replace("|", "\\|")[:120]
        out.append(f"| {i} | `{f(r['path'])}` | {f(r['file'])} | {f(r['title'])} | {f(r['authors'])} | {r['父目录存在']} |")
open(P + "disk_missing_report.md", "w", encoding="utf-8").write("\n".join(out))
print(f"报告已写 {P}disk_missing_report.md（{len(rows)} 行）")
print("顶层分布:", {t: len(v) for t, v in sorted(by.items(), key=lambda x: -len(x[1]))})
