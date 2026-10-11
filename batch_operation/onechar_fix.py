"""单汉字作者补全：生成计划 reports/onechar_plan.tsv + .json

证据优先级：
  1) Zotero 条目按序的作者全名（lastName==单字且 firstName 非空）
  2) PDF 首页词块（多名单字作者时找 4 字连排，如「刘涛曹锐」）
  3) 无证据 → manual 档（不改）
dry-run 默认；--apply 写库（UPDATE ... AND authors = $old::text[]）
"""
import json, re, pathlib, argparse, collections
import fitz

ROWS = "/app/batch_operation/reports/onechar_rows.tsv"
ZOT = "/app/batch_operation/reports/zotero_items.json"
EVID = "/app/batch_operation/reports/onechar_evidence.tsv"
PLAN_TSV = "/app/batch_operation/reports/onechar_plan.tsv"
PLAN_JSON = "/app/batch_operation/reports/onechar_plan.json"
CJK = r"\u4e00-\u9fff"
ONE = re.compile(rf"^([{CJK}])( et al\.?)?$")

def norm_t(s):
    return re.sub(r"[\s：:，,。．.、·—\-－()（）《》\"'“”‘’？！?!]", "", s or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    zo = json.load(open(ZOT, encoding="utf-8"))
    t2i = collections.defaultdict(list)
    for i in zo:
        t2i[norm_t(i.get("title", ""))].append(i)
    ev = {}
    for l in open(EVID, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if f:
            ev[f[0]] = f

    plan, stat = [], collections.Counter()
    for rid, path, title, au in rows:
        auths = [x for x in au.split("|") if x]
        idx = [k for k, x in enumerate(auths) if ONE.match(x)]
        if not idx:
            continue
        names, src = {}, ""
        # ---- 1) Zotero 按序 ----
        for it in t2i.get(norm_t(title), []):
            cr = [c for c in it.get("creators", []) if c.get("creatorType") == "author"]
            pool = []
            for c in cr:
                ln = (c.get("lastName") or "").strip()
                fn = (c.get("firstName") or "").strip()
                if len(ln) == 1 and fn and re.fullmatch(rf"[{CJK}]+", ln + fn):
                    pool.append((ln, fn))
            # 必须逐个按「姓」匹配（条目的作者序列里可能夹着译者，
            # 否则会把《艺术终结后的艺术人类学》的译者 陈韵 配给单字姓 尤）
            m, used = {}, set()
            for k in idx:
                sur = ONE.match(auths[k]).group(1)
                for j, (ln, fn) in enumerate(pool):
                    if j in used or ln != sur:
                        continue
                    m[k], _ = ln + fn, used.add(j)
                    break
            if len(m) == len(idx):
                names, src = m, "Z"
                break
        # ---- 2) PDF 词块 ----
        if not names:
            e = ev.get(path)
            head = e[5] if e and len(e) > 5 else ""
            cl = [c for c in (e[4].split(",") if e and len(e) > 4 and e[4] else []) if c]
            if len(idx) == 1:
                sur = ONE.match(auths[idx[0]]).group(1)
                pick = next((c for c in cl if c.startswith(sur)), "")
                pick = re.sub(r"[摘译编著校辑注释]$", "", pick)
                if pick and len(pick) >= 2:
                    names[idx[0]] = pick
                    src = "P"
            else:
                # 找 4 字连排：a b c d，其中 a==姓1 且 c==姓2
                got = {}
                for m in re.finditer(rf"[{CJK}]{{4}}", head):
                    w = m.group()
                    for k in idx:
                        if w[0] == ONE.match(auths[k]).group(1) and w[2] == ONE.match(auths[idx[-1]]).group(1):
                            got[idx[0]] = w[:2]
                            got[idx[-1]] = w[2:]
                if len(got) == len(idx):
                    names.update(got)
                    src = "P"
        if names and all(k in names for k in idx):
            new = list(auths)
            for k, nm in names.items():
                new[k] = nm
            plan.append({"id": rid, "path": path, "old": auths, "new": new, "src": src})
            stat["apply_" + src] += 1
        else:
            plan.append({"id": rid, "path": path, "old": auths,
                         "new": None, "src": "-",
                         "cands": (ev.get(path, ["", "", "", "", ""])[4])})
            stat["manual"] += 1
    for kind, tsv in (("apply", PLAN_TSV), ("all", PLAN_TSV)):
        with open(tsv, "w", encoding="utf-8") as f:
            f.write("id\tpath\told_authors\tnew_authors\tsrc\tcands\n")
            for p in plan:
                f.write("\t".join([p["id"], p["path"], "|".join(p["old"]),
                                   "|".join(p["new"]) if p["new"] else "",
                                   p["src"], p.get("cands", "")]) + "\n")
        break
    json.dump({"plans": plan}, open(PLAN_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(dict(stat))

    todo = [p for p in plan if p["new"]]
    if not a.apply:
        print(f"[dry-run] 待改 {len(todo)} 行，未写库")
        return
    import asyncio
    import csv as _csv
    import datetime
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/batch_operation")
    import common as C

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"batch_operation/reports/onechar_backup_{stamp}.tsv"
    with open(bak, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "authors"])
        for p in todo:
            w.writerow([p["id"], p["path"], "{" + ",".join(p["old"]) + "}"])

    async def _write():
        conn = await C.get_conn()
        try:
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2::text[] WHERE id = $1 "
                "AND authors = $3::text[]",
                [(p["id"], p["new"], p["old"]) for p in todo])
        finally:
            await conn.close()

    asyncio.run(_write())
    print(f"[apply] updated={len(todo)} backup={bak}")

main()
