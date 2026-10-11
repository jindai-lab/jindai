"""单汉字作者 → 补全名：三路取证 v2（Zotero / PDF 元数据 / PDF 首页词块）

按 CJK 词块切分（保留原文空格），过滤停用词，输出 reports/onechar_evidence.tsv：
  path \t 单字 \t zotero全名 \t pdf_meta \t cands(逗号分隔) \t head
"""
import json, re, pathlib, collections
import fitz

ROWS = "/app/batch_operation/reports/onechar_rows.tsv"
ZOT = "/app/batch_operation/reports/zotero_items.json"
OUT = "/app/batch_operation/reports/onechar_evidence.tsv"
CJK = r"\u4e00-\u9fff"
RUN = re.compile(rf"[{CJK}]{{1,8}}")
STOP = ("内容提要|提要|摘要|关键词|作者简介|简介|责任编辑|收稿|通讯|邮箱|基金|项目|"
        "大学|学院|研究院|研究所|中心|学报|期刊|杂志|出版社|编辑部|教授|副教授|讲师|"
        "博士|硕士|研究生|导师|通讯作者|第一作者|编译|著|译|卷|期|第|页|年|月|日|"
        "引用格式|文献|中图分类号|文章编号|文献标识码|DOI|http|www")

def norm_t(s):
    return re.sub(r"[\s：:，,。．.、·—\-－()（）《》\"'“”‘’？！?!]", "", s or "")

def main():
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    zo = json.load(open(ZOT, encoding="utf-8"))
    t2i = collections.defaultdict(list)
    for i in zo:
        t2i[norm_t(i.get("title", ""))].append(i)

    stat = collections.Counter()
    with open(OUT, "w", encoding="utf-8") as out:
        for rid, path, title, au in rows:
            ones = [a for a in au.split("|") if re.fullmatch(rf"[{CJK}]( et al\.?)?", a)]
            if not ones:
                continue
            sur = ones[0][0]
            zfull = ""
            for it in t2i.get(norm_t(title), []):
                for c in it.get("creators", []):
                    if c.get("creatorType") == "author" and (c.get("lastName") or "").strip() == sur:
                        fn = (c.get("firstName") or "").strip()
                        if fn:
                            zfull = sur + fn
                        break
                if zfull:
                    break
            meta, raw = "", ""
            f = pathlib.Path("/storage") / path
            if not f.exists():
                f = pathlib.Path(path)
            if f.exists():
                try:
                    doc = fitz.open(f)
                    meta = (doc.metadata or {}).get("author") or ""
                    raw = "".join(p.get_text() for p in (doc[:2] if doc.page_count > 1 else doc[:1]))
                    doc.close()
                except Exception as e:
                    meta = f"ERR:{e}"
            runs = RUN.findall(raw)
            cands = []
            for i, r0 in enumerate(runs[:80]):
                if r0[0] != sur:
                    continue
                if len(r0) >= 2:
                    nm = r0
                else:
                    # 姓被拆成单字时（CNKI 排版 `卢 春 红`），沿单字块向后拼到 3 字为止；
                    # 遇到长度 ≥2 的块就停（那是下一个词，`刘 泉 黄丁芳` → 刘泉）
                    nm, j = r0, i + 1
                    while j < len(runs) and len(nm) < 3 and len(runs[j]) == 1:
                        nm += runs[j]
                        j += 1
                cands.append(nm)
                if len(cands) >= 8:
                    break
            clean = []
            for c in cands:
                if c in clean or re.search(STOP, c):
                    continue
                clean.append(c[:4])
            meta_ok = norm_t(meta) if (meta and not meta.startswith("ERR") and re.match(rf"^{sur}", norm_t(meta))) else ""
            stat["rows"] += 1
            stat["zotero" if zfull else "-"] += 1
            stat["pdfmeta_ok" if meta_ok else "-"] += 1
            stat["cands" if clean else "-"] += 1
            stat["noev" if not (zfull or clean or meta_ok) else "-"] += 1
            out.write("\t".join([path, sur, zfull, meta_ok, ",".join(clean),
                                 re.sub(r"\s+", "", raw)[:200]]) + "\n")
    print(dict(stat))

main()
