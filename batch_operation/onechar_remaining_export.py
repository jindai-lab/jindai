"""导出剩余单汉字作者清单（含候选线索）→ reports/onechar_remaining.tsv"""
import csv, re, subprocess, collections

ROWS = "/app/batch_operation/reports/onechar_rows2.tsv"
M2 = "/app/batch_operation/reports/onechar_manual2.tsv"
OUT = "/app/batch_operation/reports/onechar_remaining.tsv"
CJK = r"\u4e00-\u9fff"
ONE = re.compile(rf"^([{CJK}])(?: et al\.?)?$")

def main():
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    m2 = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(M2, encoding="utf-8") if l.strip()}
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old_authors", "title", "候选线索"])
        n = 0
        for rid, path, title, au in rows:
            if not any(ONE.match(a) for a in au.split("|")):
                continue
            hint = ""
            e = m2.get(path)
            if e and len(e) > 3:
                hint = f"{e[3]}:{e[2]}"
            w.writerow([rid, path, au, title, hint[:120]])
            n += 1
    print("剩余清单行数:", n)

main()
