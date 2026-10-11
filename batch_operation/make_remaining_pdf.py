#!/usr/bin/env python3
"""剩余 508 行（盘上确无文件）的 PDF 清单"""
import csv, collections, html, subprocess

P = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
rows = list(csv.DictReader(open(P + "remaining_final.tsv", encoding="utf-8"), delimiter="\t"))
by = collections.defaultdict(list)
for r in rows:
    by[(r["path"].split("/")[0] if not r["path"].startswith("../app/") else r["path"].split("/")[2])].append(r)
e = html.escape
h = ["""<html><head><meta charset="utf-8"><style>
@page { size: A4 landscape; margin: 12mm 10mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; } }
body { font-family: "Sarasa Gothic SC","Noto Sans CJK SC",sans-serif; font-size: 8pt; }
h1 { font-size: 15pt; } h2 { font-size: 11pt; border-bottom: 1px solid #999; margin: 5mm 0 2mm; }
table { width: 100%; border-collapse: collapse; }
th,td { border: .4pt solid #bbb; padding: 1.2mm 1.5mm; vertical-align: top; word-break: break-all; }
th { background: #eee; font-size: 7.5pt; } td.n { text-align: right; color: #666; width: 6mm; }
.path { font-family: "Sarasa Mono SC",monospace; font-size: 7pt; }
</style></head><body>"""]
h.append("<h1>jindai 文库：仍未修复的「文件不在盘」记录（盘上确无文件）</h1>")
h.append(f"""<div>初始盘点命中 1,305 行；已批量修复/清理 <b>1,157</b> 行（改路径 / 删除重复·冗余·无文件记录 / 盘上双后缀重命名，全库 274,000 → 273,434 行）；
本清单为剩余 <b>{len(rows)}</b> 行——逐行复验过盘上确实没有对应文件，无法自动修复，需人工判断（补文件 或 删除记录）。<br>
分类：{ '、'.join(f'{k} {len(v)} 行' for k,v in sorted(by.items(), key=lambda x:-len(x[1]))) }</div>""")
for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    h.append(f"<h2>{e(t)}/ （{len(rs)} 行）</h2><table><tr><th>#</th><th>原路径（库内记录）</th><th>文件名</th><th>标题</th><th>作者</th><th>父目录<br>在盘</th></tr>")
    for i, r in enumerate(rs, 1):
        h.append(f"<tr><td class='n'>{i}</td><td class='path'>{e(r['path'])}</td><td>{e(r['file'])}</td>"
                 f"<td>{e(r['title'])}</td><td>{e(r['authors'])}</td><td>{e(r['父目录存在'])}</td></tr>")
    h.append("</table>")
h.append("</body></html>")
open(P + "remaining_report.html", "w", encoding="utf-8").write("".join(h))
subprocess.run(["weasyprint", P + "remaining_report.html", P + "jindai_仍缺失清单.pdf"], check=False)
print(subprocess.run(["ls", "-lh", P + "jindai_仍缺失清单.pdf"], capture_output=True, text=True).stdout)
subprocess.run(["cp", P + "jindai_仍缺失清单.pdf", "/mnt/repo/workspace/"], check=False)
