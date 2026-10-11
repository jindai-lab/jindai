#!/usr/bin/env python3
"""生成「文件不在盘」清单报告：HTML → PDF（weasyprint），中文用 Sarasa/Noto。"""
import csv, os, collections, html, subprocess

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "disk_missing_audit.tsv", encoding="utf-8"), delimiter="\t"))
S = "/misc/Public/jindai-storage"


def top(p):
    p = p[7:] if p.startswith("../app/") else p
    return p.split("/")[0] or "(空)"


by = collections.defaultdict(list)
for r in rows:
    by[top(r["path"])].append(r)
tot_dir = sum(1 for r in rows if r["父目录存在"] == "是")
sim = sum(1 for r in rows if r["相似度"])

e = html.escape
h = []
h.append("""<html><head><meta charset="utf-8"><style>
@page { size: A4 landscape; margin: 12mm 10mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; } }
body { font-family: "Sarasa Gothic SC","Noto Sans CJK SC","Sarasa Term SC",sans-serif; font-size: 8pt; line-height: 1.35; }
h1 { font-size: 15pt; margin: 0 0 4mm 0; }
h2 { font-size: 11pt; margin: 6mm 0 2mm 0; border-bottom: 1px solid #999; padding-bottom: 1mm; }
table { width: 100%; border-collapse: collapse; }
th, td { border: 0.4pt solid #bbb; padding: 1.2mm 1.5mm; vertical-align: top; word-break: break-all; }
th { background: #eee; font-size: 7.5pt; }
td.n { text-align: right; color: #666; width: 6mm; }
.path { font-family: "Sarasa Mono SC",monospace; font-size: 7pt; }
.miss { color: #b00; font-weight: bold; }
.sim { color: #070; }
.sum { margin: 0 0 3mm 0; }
</style></head><body>""")
h.append("<h1>jindai 文库：记录在库但本地文件不在盘 — 全量清单</h1>")
h.append(f"""<div class="sum"><b>盘点方法</b>：对 <code>file_metadata</code> 全库 <b>274,000</b> 行逐行实测 <code>os.path.exists</code>（存储根 <code>{e(S)}</code>）。<br>
<b>结果</b>：网址类记录 198,133 行（本无本地文件，已排除）；本地文件 75,867 行，其中 <b class="miss">1,305 行</b>文件不在盘。<br>
其中父目录仍存在（文件被删/改名）<b>{tot_dir}</b> 行；父目录也不存在（整棵目录树缺失）<b>{len(rows)-tot_dir}</b> 行；
已自动匹配到盘上疑似对应文件 <b class="sim">{sim}</b> 行。</div>""")
h.append("<h2>按顶层目录分布</h2><table><tr><th>顶层目录</th><th>行数</th><th>顶层目录在盘</th><th>父目录仍存在</th></tr>")
for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    h.append(f"<tr><td class='path'>{e(t)}/</td><td class='n'>{len(rs)}</td>"
             f"<td>{'是' if os.path.isdir(os.path.join(S,t)) else '<span class=miss>否</span>'}</td>"
             f"<td class='n'>{sum(1 for r in rs if r['父目录存在']=='是')}</td></tr>")
h.append("</table>")

for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    h.append(f"<h2>{e(t)}/ （{len(rs)} 行）</h2>")
    h.append("<table><tr><th>#</th><th>原路径（库内记录）</th><th>文件名</th><th>标题</th><th>作者</th>"
             "<th>父目录<br>在盘</th><th>盘上疑似对应文件（相似度）</th></tr>")
    for i, r in enumerate(rs, 1):
        cans = f"{e(r['疑似对应文件'])} <span class='sim'>({e(r['相似度'])})</span>" if r["相似度"] else ""
        h.append(f"<tr><td class='n'>{i}</td><td class='path'>{e(r['path'])}</td><td>{e(r['file'])}</td>"
                 f"<td>{e(r['title'])}</td><td>{e(r['authors'])}</td>"
                 f"<td>{'是' if r['父目录存在']=='是' else '<span class=miss>否</span>'}</td><td>{cans}</td></tr>")
    h.append("</table>")
h.append("</body></html>")
open(P + "disk_missing_report.html", "w", encoding="utf-8").write("".join(h))
subprocess.run(["weasyprint", P + "disk_missing_report.html", P + "jindai_文件不在盘清单.pdf"], check=False)
print("PDF:", P + "jindai_文件不在盘清单.pdf")
print(subprocess.run(["ls", "-lh", P + "jindai_文件不在盘清单.pdf"], capture_output=True, text=True).stdout)
