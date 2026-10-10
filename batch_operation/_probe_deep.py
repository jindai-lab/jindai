"""一次性探针：对指定 PDF 取前若干页文本，用于推断作者（stdin 传路径列表）。"""
import json
import sys

sys.path.insert(0, "/app")

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from jindai.storage import storage

out = []
for rel in sys.stdin.read().splitlines():
    rel = rel.strip()
    if not rel:
        continue
    doc = fitz.open(storage.safe_join(rel))
    txt = []
    for p in list(range(0, 8)) + [10, 12, 15]:
        if p < doc.page_count:
            t = doc[p].get_text().strip()
            if t:
                txt.append(f"[p{p + 1}] " + t[:700])
    out.append({"path": rel, "n": doc.page_count, "t": "\n".join(txt)[:4000]})
    doc.close()

print(json.dumps(out, ensure_ascii=False, indent=1))
