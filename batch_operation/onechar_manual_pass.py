"""单汉字作者第二轮：对 manual 档，扫全篇（最多 8 页）找「作者简介 / 收稿 / 文/ / 记者」等署名模式

输出 reports/onechar_manual2.tsv：path \t 姓 \t 候选(逗号分隔) \t 命中模式 \t 片段
"""
import re, pathlib, collections
import fitz

MAN = "/app/batch_operation/reports/onechar_manual.tsv"
OUT = "/app/batch_operation/reports/onechar_manual2.tsv"
CJK = r"\u4e00-\u9fff"
RUN = re.compile(rf"[{CJK}]{{1,8}}")
PATTERNS = [
    (r"作者简介[:：]?\s*", "作者简介"),
    (r"姓名[:：]\s*", "姓名"),
    (r"(?:文|作者)[/／]\s*", "文/"),
    (r"记者[:：]?\s*", "记者"),
    (r"通讯作者[:：]?\s*", "通讯作者"),
    (r"撰写[:：]?\s*", "撰写"),
    (r"责编[:：]?\s*", "责编"),
]

def main():
    rows = [l.rstrip("\n").split("\t") for l in open(MAN, encoding="utf-8") if l.strip()]
    hits = collections.Counter()
    with open(OUT, "w", encoding="utf-8") as out:
        for path, sur in rows:
            f = pathlib.Path("/storage") / path
            if not f.exists():
                f = pathlib.Path(path)
            if not f.exists():
                hits["nofile"] += 1
                out.write("\t".join([path, sur, "", "缺文件", ""]) + "\n")
                continue
            try:
                doc = fitz.open(f)
                txt = "".join(p.get_text() for p in doc[:8])
                doc.close()
            except Exception as e:
                hits["err"] += 1
                out.write("\t".join([path, sur, "", f"ERR:{e}", ""]) + "\n")
                continue
            flat = re.sub(r"\s+", "", txt)
            cands, mode, frag = [], "", ""
            for pat, name in PATTERNS:
                m = re.search(pat + rf"([{CJK}]{{2,4}})", flat)
                if m and m.group(1)[0] == sur:
                    cands.append(m.group(1))
                    mode = name
                    frag = flat[max(0, m.start() - 10):m.start() + 30]
                    break
            if not cands:
                # 全篇按词块统计以该姓开头的 2–3 字串
                freq = collections.Counter()
                for r0 in RUN.findall(flat):
                    if len(r0) >= 2 and r0[0] == sur:
                        freq[r0[:3] if len(r0) >= 3 else r0] += 1
                # 也统计「姓 名」被空格拆开的情况
                for m in re.finditer(sur, flat):
                    tail = flat[m.start():m.start() + 3]
                    if len(tail) >= 2:
                        freq[tail] += 1
                cands = [w for w, c in freq.most_common(4)]
                mode = "全篇词频"
                frag = flat[:80]
            hits["hit" if cands else "none"] += 1
            out.write("\t".join([path, sur, ",".join(cands), mode, frag[:80]]) + "\n")
    print(dict(hits))

main()
