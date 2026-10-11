"""剩余单汉字作者：模糊匹配 Zotero（题名互为前后缀/包含）+ 文件名题名段

输出 reports/onechar_plan4.tsv
"""
import csv, re, json, collections, pathlib

ROWS = "/app/batch_operation/reports/onechar_rows2.tsv"
ZOT = "/app/batch_operation/reports/zotero_items.json"
OUT = "/app/batch_operation/reports/onechar_plan4.tsv"
CJK = r"\u4e00-\u9fff"
ONE = re.compile(rf"^([{CJK}])(?: et al\.?)?$")

def norm_t(s):
    return re.sub(r"[\s：:，,。．.、·—\-－()（）《》\"'“”‘’？！?!]", "", s or "")

def title_from(path):
    base = pathlib.Path(path).name.rsplit(".", 1)[0].split("_")
    yi = next((i for i, x in enumerate(base) if re.fullmatch(r"\d{4}", x)), None)
    if yi is not None and yi + 1 < len(base):
        return "_".join(base[yi + 1:])
    return ""

def main():
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    zo = json.load(open(ZOT, encoding="utf-8"))
    nt = [(norm_t(i.get("title", "")), i) for i in zo]
    stat = collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "证据"])
        for rid, path, title, au in rows:
            old = au.split("|")
            idx = [k for k, a in enumerate(old) if ONE.match(a)]
            if not idx:
                continue
            key = norm_t(title_from(path)) or norm_t(title)
            if len(key) < 5:
                stat["no_key"] += 1
                continue
            hits = [it for t2, it in nt if t2.startswith(key) or key.startswith(t2)
                    or (len(key) >= 8 and key in t2)]
            names = {}
            for it in hits[:8]:
                pool = []
                for c in it.get("creators", []):
                    if c.get("creatorType") != "author":
                        continue
                    ln = (c.get("lastName") or "").strip()
                    fn = (c.get("firstName") or "").strip()
                    if ln and fn and re.search(rf"[{CJK}]", fn):
                        pool.append((ln, fn))
                mm, used = {}, set()
                for k in idx:
                    sur = ONE.match(old[k]).group(1)
                    for j, (ln, fn) in enumerate(pool):
                        if j in used or not ln.startswith(sur):
                            continue
                        mm[k], _ = ln + fn, used.add(j)
                        break
                if len(mm) == len(idx):
                    names = mm
                    break
            if names and all(k in names for k in idx):
                new = list(old)
                for k, v in names.items():
                    new[k] = v
                w.writerow([rid, path, "|".join(old), "|".join(new), "Z4", "模糊题名"])
                stat["apply"] += 1
            else:
                stat["manual"] += 1
    print(dict(stat))

main()
