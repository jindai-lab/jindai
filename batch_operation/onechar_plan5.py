"""剩余单汉字作者：全篇唯一候选规则（读最多 8 页，每个姓在全文里只能有唯一干净候选）

输出 reports/onechar_plan5.tsv
"""
import csv, re, collections, pathlib
import fitz

ROWS = "/app/batch_operation/reports/onechar_rows2.tsv"
OUT = "/app/batch_operation/reports/onechar_plan5.tsv"
CJK = r"\u4e00-\u9fff"
RUN = re.compile(rf"[{CJK}]{{1,8}}")
ONE = re.compile(rf"^([{CJK}])(?: et al\.?)?$")
STOP = ("内容提要|提要|摘要|关键词|作者简介|简介|责任编辑|收稿|通讯|邮箱|基金|项目|大学|"
        "学院|研究院|研究所|中心|学报|期刊|杂志|出版社|编辑部|教授|副教授|讲师|博士|硕士|"
        "研究生|导师|编译|著|译|卷|期|第|页|年|月|日|引用格式|文献|中图|文章编号|理论|"
        "研究|分析|问题|文化|艺术|媒介|城市|技术|数字|社会|中国|世界|历史|美学|电影|"
        "老师|先生|女士|同学|记者|作者")
TRAIL = re.compile(r"[的体教说们等著译编摘]+$")

def tidy(c, sur):
    c = re.sub(rf"^{sur}[0-9]+", sur, c)
    if re.search(r"老师|教授|先生|女士|同学|记者|作者", c):
        return ""
    c2 = TRAIL.sub("", c)
    return c2 if len(c2) >= 2 else c

def main():
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()]
    stat = collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "证据"])
        for rid, path, title, au in rows:
            old = au.split("|")
            idx = [k for k, a in enumerate(old) if ONE.match(a)]
            if not idx:
                continue
            p = pathlib.Path("/storage") / path
            if not p.exists():
                p = pathlib.Path(path)
            flat = ""
            if p.exists():
                try:
                    doc = fitz.open(p)
                    flat = re.sub(r"\s+", "", "".join(pg.get_text() for pg in doc[:8]))
                    doc.close()
                except Exception:
                    flat = ""
            names, why = {}, []
            for k in idx:
                sur = ONE.match(old[k]).group(1)
                seen = []
                for r0 in RUN.findall(flat):
                    if len(r0) < 2 or r0[0] != sur:
                        continue
                    c = tidy(r0[:3] if len(r0) >= 3 else r0, sur)
                    if not c or len(c) < 2 or re.search(STOP, c) or c in seen:
                        continue
                    seen.append(c)
                if len(seen) == 1:
                    names[k] = seen[0]
                    why.append(f"{sur}→{seen[0]}(全文唯一)")
                else:
                    stat["multi_or_none"] += 1
                    why.append(f"{sur}→{len(seen)}个候选")
            if len(names) == len(idx):
                new = list(old)
                for k, v in names.items():
                    new[k] = v
                w.writerow([rid, path, "|".join(old), "|".join(new), "P5", ";".join(why)])
                stat["apply"] += 1
            else:
                stat["manual"] += 1
    print(dict(stat))

main()
