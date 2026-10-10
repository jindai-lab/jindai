"""一次性探针：Supplementary Books 根级 42 个 PDF 的前两页文本（用于推断作者）。"""
import json
import sys

sys.path.insert(0, "/app")

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from jindai.storage import storage

conn_paths = sys.stdin.read().splitlines()
out = []
for rel in conn_paths:
    rel = rel.strip()
    if not rel:
        continue
    rec = {"path": rel, "text": "", "err": ""}
    try:
        doc = fitz.open(storage.safe_join(rel))
        chunks = []
        for p in range(min(2, doc.page_count)):
            chunks.append(doc[p].get_text()[:1800])
        rec["text"] = "\n---\n".join(chunks).strip()
        rec["pages"] = doc.page_count
        doc.close()
    except Exception as ex:
        rec["err"] = f"{type(ex).__name__}: {ex}"[:200]
    out.append(rec)

print(json.dumps(out, ensure_ascii=False, indent=1))
