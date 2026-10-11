"""dump T2 行的 PDF 题名页文本，供人工判定著者（只读）。"""
import json, os, sys, re
sys.path.insert(0, "/app"); sys.path.insert(0, "/app/batch_operation")

PLAN = "/app/batch_operation/reports/authors_quality_plan.json"
OUT = "/app/batch_operation/reports/t2_titlepages.txt"
ROOT = "/storage"

import fitz  # pymupdf

plans = json.load(open(PLAN, encoding="utf-8"))["plans"]
t2 = [x for x in plans if x["tier"] == "T2"]

buf = []
for i, x in enumerate(t2, 1):
    rel = x["path"]
    abs_path = os.path.join(ROOT, rel)
    buf.append(f"\n{'='*100}\n[{i}] {rel}\n    title_in_db : {x['old_title']}\n    authors_in_db: {x['old_authors']}")
    if not os.path.exists(abs_path):
        buf.append("    !! 文件不在盘上")
        continue
    try:
        doc = fitz.open(abs_path)
        txt = ""
        for pno in range(min(3, doc.page_count)):
            txt += doc[pno].get_text() or ""
        doc.close()
        txt = re.sub(r"[ \t]+", " ", txt).strip()
        if len(txt) < 30:
            buf.append(f"    !! 无文本层（扫描版），提取 {len(txt)} 字符")
            buf.append("    " + txt[:200].replace("\n", " ⏎ "))
        else:
            buf.append("    ---- 前3页文本（前1200字）----")
            buf.append(txt[:1200])
    except Exception as e:
        buf.append(f"    !! 打开失败: {type(e).__name__}: {e}")

open(OUT, "w", encoding="utf-8").write("\n".join(buf))
print("rows:", len(t2), "->", OUT, "chars:", sum(len(b) for b in buf))
