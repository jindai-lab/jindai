"""单汉字作者：多作者行（`Zotero/<姓1>_<姓2>_<年>_…`）双姓解析 + 单作者行的二三路补漏

输出 reports/onechar_plan3.tsv（id/path/old/new/src/证据）
"""
import csv, re, json, collections, pathlib
import fitz

PLAN = "/app/batch_operation/reports/onechar_plan.tsv"
ZOT = "/app/batch_operation/reports/zotero_items.json"
OUT = "/app/batch_operation/reports/onechar_plan3.tsv"
CJK = r"\u4e00-\u9fff"
RUN = re.compile(rf"[{CJK}]{{1,8}}")
CLEAN = re.compile(rf"^[{CJK}]{{2,3}}$")
STOP = ("内容提要|提要|摘要|关键词|作者简介|简介|责任编辑|收稿|通讯|邮箱|基金|项目|大学|"
        "学院|研究院|研究所|中心|学报|期刊|杂志|出版社|编辑部|教授|副教授|讲师|博士|硕士|"
        "研究生|导师|编译|著|译|卷|期|第|页|年|月|日|引用格式|文献|中图|文章编号|理论|"
        "研究|分析|问题|文化|艺术|媒介|城市|技术|数字|社会|中国|世界|历史|美学|电影")
ONE = re.compile(rf"^([{CJK}])(?: et al\.?)?$")
# 候选尾部常见虚字（安全集：人名几乎不以这些字收尾）
TRAIL = re.compile(r"[的体教说们等著译编摘]+$")
# 明显是称呼而非姓名
ROLEWORD = re.compile(r"老师|教授|先生|女士|博士|同学|记者|作者")

def tidy(c, sur):
    c = re.sub(rf"^{sur}[0-9]+", sur, c)
    if ROLEWORD.search(c):
        return ""
    c2 = TRAIL.sub("", c)
    return c2 if len(c2) >= 2 else c

def norm_t(s):
    return re.sub(r"[\s：:，,。．.、·—\-－()（）《》\"'“”‘’？！?!]", "", s or "")

def cands_for(flat, sur, limit=700):
    """该姓在正文里的干净候选（按出现顺序，去重）"""
    out = []
    for r0 in RUN.findall(flat[:limit]):
        if len(r0) < 2 or r0[0] != sur:
            continue
        c = r0[:3] if len(r0) >= 3 else r0
        if c in out or re.search(STOP, c):
            continue
        out.append(c)
    return out

def main():
    plan = [r for r in csv.DictReader(open(PLAN, encoding="utf-8"), delimiter="\t")]
    man = {r["path"]: r for r in plan if not r["new_authors"]}
    zo = json.load(open(ZOT, encoding="utf-8"))
    t2i = collections.defaultdict(list)
    for i in zo:
        t2i[norm_t(i.get("title", ""))].append(i)

    stat = collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "证据"])
        for path, m in man.items():
            old = m["old_authors"].split("|")
            idx = [k for k, a in enumerate(old) if ONE.match(a)]
            if not idx:
                continue
            title = ""
            for seg in pathlib.Path(path).name.split("_"):
                pass
            # 从文件名取题名段（Zotero/<…>_<年>_<题名>.pdf）
            # Zotero/<作者段>_<年>_<题名>.pdf：作者段可能 1–2 段，要跳过 4 位年份
            base = pathlib.Path(path).name.rsplit(".", 1)[0].split("_")
            yi = next((i for i, x in enumerate(base) if re.fullmatch(r"\d{4}", x)), None)
            if yi is not None and yi + 1 < len(base):
                title = "_".join(base[yi + 1:])
            names, src, why = {}, "", ""
            # --- 1) Zotero：允许译名里的「·」「-」 ---
            items = []
            if title:
                items = t2i.get(norm_t(title), [])
                if not items and len(norm_t(title)) >= 5:
                    nt = norm_t(title)
                    items = [it for it in zo
                             if norm_t(it.get("title", "")).startswith(nt)][:5]
            for it in items:
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
                    names, src, why = mm, "Z", "Zotero 作者序列"
                    break
            # --- 2) PDF 正文按姓独立解析（要求唯一干净候选） ---
            if not names:
                p = pathlib.Path("/storage") / path
                if not p.exists():
                    p = pathlib.Path(path)
                flat = ""
                if p.exists():
                    try:
                        doc = fitz.open(p)
                        flat = re.sub(r"\s+", "", "".join(pg.get_text()
                                                          for pg in doc[:2]))
                        doc.close()
                    except Exception:
                        flat = ""
                mm, ok = {}, True
                for k in idx:
                    sur = ONE.match(old[k]).group(1)
                    cs = [c for c in (tidy(x, sur) for x in cands_for(flat, sur)) if c]
                    if len(cs) == 1:
                        mm[k] = cs[0]
                    else:
                        ok = False
                        break
                # 若 Zotero 里有更长（更完整）的同名作者（罗素 → 罗素·沙曼），取长的
                if ok and len(mm) == len(idx):
                    for it in items:
                        for c in it.get("creators", []):
                            ln = (c.get("lastName") or "").strip()
                            fn = (c.get("firstName") or "").strip()
                            full = ln + fn
                            for k in idx:
                                if mm.get(k) and full.startswith(mm[k]) and len(full) > len(mm[k]):
                                    mm[k] = full
                    names, src, why = mm, "P2", "正文唯一候选(取 Zotero 更长名)"
            if names and all(k in names for k in idx):
                new = list(old)
                for k, nm in names.items():
                    new[k] = nm
                w.writerow([m["id"], path, "|".join(old), "|".join(new), src, why])
                stat["apply_" + src] += 1
            else:
                stat["manual"] += 1
    print(dict(stat))

main()
