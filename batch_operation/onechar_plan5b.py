"""P5 候选消歧：用 Zotero 全库作者名（lastName+firstName）当词典

候选 3 字名不在词典、而其 2 字前缀在词典里 → 取 2 字形（王峰理→王峰、杨震艺→杨震）。
输出 reports/onechar_plan5b.tsv
"""
import csv, json, re, collections

PLAN5 = "/app/batch_operation/reports/onechar_plan5.tsv"
ZOT = "/app/batch_operation/reports/zotero_items.json"
OUT = "/app/batch_operation/reports/onechar_plan5b.tsv"
CJK = r"\u4e00-\u9fff"

def main():
    zo = json.load(open(ZOT, encoding="utf-8"))
    D = set()
    for i in zo:
        for c in i.get("creators", []):
            ln = (c.get("lastName") or "").strip()
            fn = (c.get("firstName") or "").strip()
            nm = ln + fn
            if 2 <= len(nm) <= 4 and re.fullmatch(rf"[{CJK}]+", nm):
                D.add(nm)
    print("Zotero 人名词典:", len(D))

    rows = [l.rstrip("\n").split("\t") for l in open(PLAN5, encoding="utf-8") if l.strip()][1:]
    stat = collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "证据"])
        for f0 in rows:
            rid, path, olds, news = f0[0], f0[1], f0[2], f0[3]
            out = []
            for o, n in zip(olds.split("|"), news.split("|")):
                if len(n) == 3 and n not in D:
                    p2 = n[:2]
                    if p2 in D:
                        out.append(p2)
                        stat["dict_fix"] += 1
                        continue
                out.append(n)
            w.writerow([rid, path, olds, "|".join(out), f0[4], f0[5]])
    print(dict(stat))

main()
