"""清理 Authors/<作者目录>/ 树的书目字段：
  · authors：目录名是权威作者；丢掉「书名当作者」的脏值；保留真合著者（出现在文件名作者段）
  · title：空标题/标题≈作者名 → 从文件名取书名；剥掉出版社后缀；剥掉标题里残留的人名标注

用法：python authors_tree_clean.py [--apply]
输出：reports/authors_tree_plan.tsv + reports/authors_tree_backup_<ts>.tsv
"""
import argparse, asyncio, csv, datetime, re, sys, unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

TITLE_WORDS = set("""a an the of in on for to and or with without from at by as is are was were
what who how why its his her their our your my this that these those new old great
theory life art world time history culture media mind nature music book notes essays
introduction reader handbook studies toward towards after before between beyond
de la le les des du el y von van der die das und""".split())

PUB = re.compile(
    r"\s*[-–—_]\s*(Continuum|Routledge|SAGE|Sage|MIT Press|Harvard University Press|Oxford University Press|"
    r"Cambridge University Press|Blackwell|Palgrave|Springer|Verso|Polity|Zone|Ballantine|Penguin|Stanford|"
    r"Columbia University Press|Yale|Duke|Minnesota|SUNY|Thames (and|&) Hudson|McPherson|Basic Books|Norton|Princeton|Belknap|Chicago|Johns Hopkins|Rowman|Farrar|Cornell|Edinburgh|Bloomsbury|Wiley|De Gruyter|University of [A-Z][\w ]*Press|"
    r"University of [A-Z][\w ]+|Univ\w* Press|[\w ]*Press)\b[\s,，]*(\(?\d{4}\)?)?.*$", re.I)
YEAR_TAIL = re.compile(r"[\(\[（]\s*(19|20)\d\d\s*[\)\]）]\s*$")
# 标题里残留人名的四类形态
LAT_TAIL = re.compile(r"_\s*(?=[^_]*[A-Z])[A-Za-z.\-']+(?:[ ,]+[A-Za-z.\-']+){0,12}\.?\s*$")
CJK_TAIL = re.compile(r"_\s*[\u4e00-\u9fff·;；,，、]{2,20}\s*$")
NAME_BRACKET_PREFIX = re.compile(r"^\[[A-Za-z_ .'\-]{3,40}\]\s*[_\-–—]?\s*")
CN_MARKER = re.compile(
    r"\s*[（(\[]+[美英法德俄日意荷奥西葡][）)\]]+\s*[\u4e00-\u9fff·]{2,14}"
    r"\s*(?:著|编|译|主编)?\s*[；;，,]?.*$")
CN_TAIL_ROLE = re.compile(r"\s*[\u4e00-\u9fff·]{2,12}\s*(?:著|译|编|主编)\s*[；;，,]?.*$")
# 中文尾巴里这些是副题/版本词，不是人名
CJK_KEEP = re.compile(r"(修订|第[一二三四五六七八九十\d]+版|全译|注释|中文版|上下册|全集|选集|文集|卷|册|新编|增订|精装|译本)")
ARTIFACT = re.compile(r"\s*[\(（]\s*(?:Z-?Library|无|z-lib\.?\w*|libgen|\d{1,3}(?:st|nd|rd|th)?\s*ed\.?)\s*[\)）]\s*$", re.I)

CN_AUTHOR_TAIL = re.compile(
    r"[（(\[][^）)\]]{0,6}[）)\]]?\s*[\u4e00-\u9fff]{2,6}\s*[（(][A-Za-z][^）)]*[）)]\s*著.*$|"
    r"[（(\[][美英法德俄日意荷奥][）)\]]\s*[\u4e00-\u9fff·]{2,12}\s*[（(][A-Za-z][^）)]*[）)]?.*$|"
    r"[_\s][（(][A-Za-z][^）)]{2,40}[）)]\s*著?.*$")


def nfd(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def toks(s):
    return [t for t in re.sub(r"[^A-Za-z0-9]+", " ", nfd(s).lower()).split() if t]


def dir_author(d):
    """'Adorno, Theodor' -> 'Theodor Adorno'；(无逗号) 原样"""
    d = d.strip()
    if "," in d:
        fam, giv = [x.strip() for x in d.split(",", 1)]
        return f"{giv} {fam}".strip()
    return d


def family_of(name):
    if "," in name:
        return toks(name.split(",")[0])
    t = toks(name)
    return [t[-1]] if t else []


SEPCHARS = set(" \t-\u2013\u2014,\uff0c_:\uff1a.;\uff1b")
CONNECT = re.compile(r"^(and|of|or|und|et|y|e)\b", re.I)


def strip_leading_name(t, au):
    """最长优先剥掉开头的作者名前缀；余下以连接词开头则视为标题一部分、整体不剥。"""
    idxs = sorted({i for i, ch in enumerate(t) if ch in SEPCHARS}, reverse=True)
    for i in idxs:
        core = t[:i].strip(" \t-\u2013\u2014,\uff0c_:\uff1a.;\uff1b")
        if not core:
            continue
        st = set(toks(core))
        if not st or not st <= au:
            continue
        rest = t[i:].lstrip(" \t-\u2013\u2014,\uff0c_:\uff1a.;\uff1b")
        if rest and not CONNECT.match(rest):
            return rest
        return t
    # 放宽档：前缀以「 - 」结尾、且过半词是作者词（Kenneth Cmiel_ John Durham Peters - …）
    for i in idxs:
        if not t[max(0, i - 1):i + 1].strip().startswith(("-", "\u2013", "\u2014")):
            continue
        core = t[:i].strip(" \t-\u2013\u2014,\uff0c_:\uff1a.;\uff1b")
        st = set(toks(core))
        if len(st) >= 2 and len(st & au) / len(st) >= 0.5:
            rest = t[i:].lstrip(" \t-\u2013\u2014,\uff0c_:\uff1a.;\uff1b")
            if rest and not CONNECT.match(rest):
                return rest
    return t


def strip_name_tail(t, au):
    """从右往左找分隔符（_ / -- / -），右侧文本含作者词就截断——容错 &、重音、数字等"""
    for m in reversed(list(re.finditer(r"_{1,2}|[\u2013\u2014-]{1,2}", t))):
        tail = t[m.end():].strip()
        if not tail or len(tail) > 90:
            continue
        tt = set(toks(tail))
        if tt and (tt & au):
            return t[:m.start()].strip(" -\u2013\u2014_,&\uff0c")
    return t


def name_like(x):
    t = toks(x)
    if "," in x:
        return True
    if len(t) < 2:
        return False
    if all(w.isupper() for w in re.findall(r"[A-Za-z]+", x)) and len(t) > 1:
        return False
    return True


def clean_title(title, stem, dir_a, authors):
    """返回 (新标题, 来源标记)"""
    t = title.strip()
    src = ""
    parts = stem.split(" - ", 1)
    L, R = (parts[0], parts[1]) if len(parts) == 2 else (stem, "")
    # ① 标题空 或 标题≈作者名 → 从文件名取书名
    dt, tt = set(toks(dir_a)), set(toks(t))
    same_person_title = bool(tt) and bool(dt) and tt <= dt
    if not t or same_person_title or CN_AUTHOR_TAIL.fullmatch(t):
        cand = ""
        if len(parts) == 2:
            la = set(toks(" ".join(authors)))
            cand = R if (set(toks(L)) & la or set(toks(L)) & dt) else L
        if not cand:
            cand = re.sub(r"^[^ ]*(, ?[^ ]+)? *- *", "", stem)
        t, src = cand.strip(), "文件名"
    # ②-a 括注里的「年份, 出版社」
    t = re.sub(r"\s*[\(（]\s*(?:\d{4}\s*,\s*)?[^()（）]{0,60}?"
               r"(?:Press|Universit\w*|Verlag|Publishing|Books|Routledge|Springer|Palgrave|Polity|Sage)\b"
               r"[^()（）]{0,30}[\)）]\s*$", "", t, flags=re.I)
    # ② 剥出版社/年份后缀
    t2 = PUB.sub("", t).strip(" -–—_,")
    if t2 != t:
        t, src = t2, src or "去出版社"
    t = YEAR_TAIL.sub("", t).strip(" -–—_,")
    au_tokens = set(toks(" ".join(authors))) | set(toks(dir_a))
    # ③-0 前导方括号块（系列名/杂串）与前导「人名 - 」
    t = re.sub(r"^\[[^\]]{6,}\]\s*[_\-\u2013\u2014]?\s*", "", t)
    # 前导圆括号系列块：(_Marquette Studies in Philosophy 10_) Paul Ricoeur-…
    t = re.sub(r"^[\(（][^()（）]{4,60}?(?:Studies|Series|Press|Edition|Vol\.?|Philosophy|History)\b[^()（）]{0,30}[\)）]\s*[_\-\u2013\u2014]?\s*", "", t)
    t = strip_leading_name(t, au_tokens)
    # ③-a 从「-- [美]XXX」/「(（美）XXX著」这类人名段直接截断，保留前面的真标题
    cut = re.search(r"\s*(?:--+\s*)?[（(\[]+[美英法德俄日意荷奥西葡][）)\]]+", t)
    if cut and cut.start() >= 2:
        t = t[:cut.start()].strip(" -–—_")
    t = re.sub(r"\s*--\s*\d{13}\s*--.*$", "", t)          # -- ISBN -- 杂串
    t = re.sub(r"\s*--\s*[0-9a-f]{20,}.*$", "", t, flags=re.I)
    t = re.sub(r"\s*--\s*Anna.s Archive.*$", "", t, flags=re.I)
    # ③ 尾部残留人名（只剥一轮，避免把 "Essays by Agnes Heller" 这种正当标题削平）
    t = strip_name_tail(t, au_tokens)
    # ③ 剥标题里残留的人名/杂质（多轮，直到稳定）
    for _ in range(4):
        before = t
        # 中文国别标记 + 姓名 + 著/译/编…
        t = CN_MARKER.sub("", t)
        # 以作者名开头的拉丁前缀（Arendt Hannah The Origins…）
        t = strip_leading_name(t, au_tokens)
        # [_人名] / _人名, 人名 / _中文名 尾巴
        # 只在尾巴确实含作者姓氏时才剥（避免切掉 "Wilhelm Meister_ Apprenticeship" 这类副题）
        for rx, use_tok in ((LAT_TAIL, True), (CJK_TAIL, False)):
            m = rx.search(t)
            if m:
                tail = m.group(0)
                hit = (set(toks(tail)) & au_tokens) if use_tok else not CJK_KEEP.search(tail)
                if hit:
                    t = t[:m.start()]
        t = NAME_BRACKET_PREFIX.sub("", t)
        # 纯中文「XX著；…」尾巴（仅当 XX 像人名且后面跟角色词）
        t = CN_TAIL_ROLE.sub(lambda mm: "" if re.match(r"^\s*[\u4e00-\u9fff·]{2,4}\s*(著|译|编)", mm.group(0)) else mm.group(0), t)
        t = CN_AUTHOR_TAIL.sub("", t)
        t = ARTIFACT.sub("", t)
        t = re.sub(r"\s{2,}", " ", t).strip(" -–—_,")
        if t == before:
            break
    t = re.sub(r"[\s.,，、;；:_\-–—]+$", "", t)
    if len(t) < 2:                      # 别把标题清空——回退原值
        return title.strip(), ""
    if t != title.strip():
        src = src or "去人名"
    return t, src


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, coalesce(authors,'{}') AS authors, coalesce(title,'') AS title "
            "FROM file_metadata WHERE path LIKE 'Authors/%' ORDER BY path")
        plan = []
        for r in rows:
            p = r["path"].split("/")
            if len(p) < 3:
                continue
            d, stem = p[1], re.sub(r"\.pdf$", "", p[-1], flags=re.I)
            da = dir_author(d)
            dset = set(toks(da))
            left = set(toks(stem.split(" - ", 1)[0]))
            old = list(r["authors"])
            keep, seen = [], set()
            for x in old:
                fam = set(family_of(x))
                if fam and fam <= dset:
                    if x not in seen:
                        keep.append(x); seen.add(x)
                    continue
                if name_like(x) and set(toks(x)) and set(toks(x)) <= left:
                    if x not in seen:
                        keep.append(x); seen.add(x)
            # 目录作者放首位（已有同人写法则用它更全的形式）
            if not any(set(family_of(x)) <= dset for x in keep):
                keep.insert(0, da)
            new_au = keep
            nt, src = clean_title(r["title"], stem, da, new_au or old)
            ch_au = new_au != old
            ch_ti = nt != r["title"]
            if ch_au or ch_ti:
                plan.append((r["id"], r["path"], "|".join(old), "|".join(new_au),
                             r["title"], nt, src, "作者+标题" if ch_au and ch_ti else ("作者" if ch_au else "标题")))
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f"/app/batch_operation/reports/authors_tree_plan.tsv", "w",
                  encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "path", "old_authors", "new_authors", "old_title", "new_title", "src", "动作"])
            for i, p_, oa, na, ot, nt, s, act in plan:
                w.writerow([i, p_, oa, na, ot, nt, s, act])
        with open(f"/app/batch_operation/reports/authors_tree_backup_{stamp}.tsv", "w",
                  encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "path", "authors", "title"])
            for r in rows:
                w.writerow([r["id"], r["path"], "|".join(r["authors"]), r["title"]])
        print(f"计划 {len(plan)} 行 / 共 {len(rows)}；备份 authors_tree_backup_{stamp}.tsv")
        import collections
        print(dict(collections.Counter(x[7] for x in plan)))
        if not a.apply:
            print("[dry-run] 未写库")
            return
        n = 0
        async with conn.transaction():
            for i, p_, oa, na, ot, nt, s, act in plan:
                await conn.execute(
                    "UPDATE file_metadata SET authors=$2::text[], title=$3 "
                    "WHERE id=$1 AND coalesce(array_to_string(authors,'|'),'')=$4",
                    i, na.split("|") if na else [], nt, oa)
                n += 1
        left_au = await conn.fetchval(
            "SELECT count(*) FROM file_metadata WHERE path LIKE 'Authors/%' "
            "AND (authors IS NULL OR cardinality(authors)=0)")
        left_ti = await conn.fetchval(
            "SELECT count(*) FROM file_metadata WHERE path LIKE 'Authors/%' "
            "AND coalesce(btrim(title),'')=''")
        print(f"[apply] 写 {n} 行；Authors/ 空作者 {left_au}；空标题 {left_ti}")
    finally:
        await conn.close()

asyncio.run(main())
