"""剩余单汉字作者：导出文本（有文本层读 8 页；无文本层 → 前 2 页渲染 PNG → 本地 OCR）

输出 reports/onechar_texts.tsv：path \t 姓(逗号分隔) \t 来源(text/ocr/none) \t 文本(≤3000 字)
"""
import csv, re, json, pathlib, sys
import fitz
import urllib.request

ROWS = "/app/batch_operation/reports/onechar_rows2.tsv"
OUT = "/app/batch_operation/reports/onechar_texts.tsv"
IMG = "/tmp/onechar_img"
OCR = "http://172.24.1.230:8080/ocr"
CJK = r"\u4e00-\u9fff"
ONE = re.compile(rf"^([{CJK}])(?: et al\.?)?$")

def ocr_png(png_bytes, name="p.png"):
    """PaddleOCR 容器：POST /ocr/ JSON {file: base64, fileType:1, visualize:false}"""
    import base64
    payload = json.dumps({"file": base64.b64encode(png_bytes).decode(),
                          "fileType": 1, "visualize": False}).encode()
    req = urllib.request.Request(OCR, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read())
    try:
        return "\n".join(d["result"]["ocrResults"][0]["prunedResult"]["rec_texts"])
    except Exception:
        return ""


def main():
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    stat = {}
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "姓氏", "来源", "文本"])
        for rid, path, title, au in rows:
            surs = [ONE.match(a).group(1) for a in au.split("|") if ONE.match(a)]
            if not surs:
                continue
            p = pathlib.Path("/storage") / path
            if not p.exists():
                p = pathlib.Path(path)
            txt = ""
            src = "none"
            if p.exists():
                try:
                    doc = fitz.open(p)
                    txt = "".join(pg.get_text() for pg in doc[:8])
                    doc.close()
                except Exception as e:
                    txt = ""
                if len(re.sub(r"\s", "", txt)) >= 80:
                    src = "text"
                else:
                    try:
                        doc = fitz.open(p)
                        chunks = []
                        for i in range(min(2, doc.page_count)):
                            pix = doc[i].get_pixmap(dpi=200)
                            chunks.append(ocr_png(pix.tobytes("png"), f"p{i}.png"))
                        txt = "\n".join(chunks)
                        doc.close()
                        src = "ocr" if len(re.sub(r"\s", "", txt)) >= 40 else "none"
                    except Exception as e:
                        txt, src = f"OCRERR {e}", "none"
            flat = re.sub(r"[ \t\u3000]+", " ", txt)
            flat = re.sub(r"\n{2,}", "\n", flat)[:3000]
            stat[src] = stat.get(src, 0) + 1
            w.writerow([path, ",".join(surs), src, flat])
            print(f"[{src}] {path[:60]}", flush=True)
    print(dict(stat))

main()
