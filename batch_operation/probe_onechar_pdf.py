"""探测：单汉字作者的文件，PDF 首页/题名页能否取到全名（容器内跑，PDF 挂在 /storage）"""
import re, pathlib, sys
import fitz

ROWS = "/app/batch_operation/reports/onechar_rows.tsv"
CJK = r"\u4e00-\u9fff"
NAME = re.compile(rf"[{CJK}]{{2,4}}")

def probe(paths, n=12):
    for p in paths[:n]:
        full = pathlib.Path("/storage") / p
        if not full.exists():
            print(f"[miss-file] {p}")
            continue
        try:
            doc = fitz.open(full)
            txt = ""
            for pg in doc[:2]:
                txt += pg.get_text()
            doc.close()
        except Exception as e:
            print(f"[err] {p} :: {e}")
            continue
        txt = re.sub(r"\s+", " ", txt)
        print(f"--- {p}")
        print("    " + txt[:300])

if __name__ == "__main__":
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    probe([r[1] for r in rows], 12)
