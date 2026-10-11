"""authors_quality_fix.py — 定点清洗 authors 里含数字 / 括号 / 标点的行（2026-10-11 用户指令）。

第 0 节质检已定位：有 authors 的 27,311 文件 / 31,360 元素中，
命中 `含数字 ∪ 含括号 ∪ 含标点` 的只有 241 文件 / ~275 元素。
本脚本只在这一个范围内出改动方案，不碰其余 27,069 个健康文件。

分桶与修法（对应 author-normalization.md 第 0 节根因表）：
  drop_digit    纯数字 ID / 卷册号 / 域名 —— 文件名序号，直接删该元素
  clean_elem    剥编者标记 (eds.)/(editor)、脚注数字、尾随生年、竖线→逗号
  swap          title 与 authors 颠倒（书名在 authors、人名在 title）→ 互换
  split         多作者未拆（、/；/两个以上人名间的逗号）→ 拆成单人
  drop_empty    删光后无作者可回填 → 置空数组（列 T2 待复核）
  unresolved    丛书名/无作者书、双向都像人名、书名残片 → 原样保留 + 人工清单
  keep          合规（姓名(国籍) 等），不动

T1 = 高置信、--apply 会执行；T2 = 只列清单、--apply 也不动。

dry-run 默认；--apply 写库（先落 reports/authors_quality_backup_<date>.tsv）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402
import authors_normalize as AN  # noqa: E402  （复用其人名判定/倒装名工具）

log = C.setup_logging(
    "authors_quality_fix", "batch_operation/reports/authors_quality_fix.log"
)

SCOPE_SQL = """
    SELECT id, path, coalesce(title, '') AS title, authors
    FROM file_metadata
    WHERE authors IS NOT NULL AND cardinality(authors) > 0
      AND EXISTS (SELECT 1 FROM unnest(authors) a
                  WHERE a ~ '[0-9]' OR a ~ '[()]'
                     OR a ~ '[,，、;；:：/|&]')
"""

# ------------------------------------------------------------------ 正则
RE_PURE_DIGIT = re.compile(r"^\d+$")
RE_VOLUME = re.compile(r"^(?:Volume|Vol\.)\s*\d+$", re.I)
RE_DOMAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]*\.(?:com|cn|net|org|edu|io|me)$", re.I)
RE_EDITOR = re.compile(
    r"\s*[\(\[]\s*(?:eds?\.|editors?|Ed\.)\s*[\)\]]\s*", re.I)
RE_EDITOR_TAIL = re.compile(
    r"\s*[,，]?\s*(?:editors?|editor|authors?|author|eds?\.?|"
    r"等?(?:编著|编译|著|编|译|校))\.?\s*$", re.I)
RE_FOOTNOTE_DIGIT = re.compile(r"(?<=[一-龥])\d(?=\s|$)")
RE_TRAIL_YEAR = re.compile(r"\s*[-–—]?\s*\(\s*(?:1[5-9]|20)\d{2}(?:[-–_]\d{2,4})?\s*\)\s*$")
RE_TRAIL_BIRTH = re.compile(r"\s+\d{4}\s*[-–—]?\s*$")
RE_PIPE = re.compile(r"\s*\|\s*")

RE_YEAR_PAREN = re.compile(r"\((?:1[5-9]|20)\d{2}(?:[-–_]\d{2,4})?\)")
RE_SERIES_PAREN = re.compile(
    r"\((?:诗苑译林|大师诗选|域外诗丛|读书文丛|黑皮诗丛|人文悦读|巨擘书库|醒客阅读|"
    r"黑塞文集|海德格尔文集|现代西方学术文库|当代激进思想家译丛|社会理论译丛|"
    r"Inside Technology|Internationalizing Media Studies|Materializing Culture|"
    r"One World Archaeology|Galaxy Books|Heretical Thought)\)")
RE_PUB = re.compile(
    r"出版社|书店|印书馆|文献出版社|Press|Publisher|Routledge|Polity|Cotta|Twayne|"
    r"Gingko|Tor Books|Springer|Oxford|Brill|MIT\b|GRKG|Beiheft|Wintersemester")
RE_CJK_TITLE_MARK = re.compile(r"[：:《》「」]")
RE_LEADING_TAG = re.compile(
    r"^\s*(?:\([^\(\)]{1,60}\)|\[[^\[\]]{1,24}\]|〔[^〔〕]{1,10}〕)\s*")
RE_SERIES_IN_TEXT = re.compile(r"\s*[(（][^)）]{1,20}[)）]\s*$")

ROLE_WORD = re.compile(r"(?:编|译|著|校|编著|编译)\s*$")


# ------------------------------------------------------------------ 工具
def fname_stem(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0]


def strip_tags(s: str) -> str:
    """去掉开头的丛书名 / 国别 / 译丛标记，得到干净的人名或书名。"""
    prev = None
    while prev != s:
        prev = s
        s = RE_LEADING_TAG.sub("", s)
    return s.strip()


RE_LATINISH = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .'\-]*$")


def looks_person(s: str) -> bool:
    """判断一个字符串像不像『单个人名』。
    只允许剥开头的丛书名/国别标签和结尾的**国籍**括号；其余括号一律否——
    '(1987)'、'(诗苑译林)' 一旦被剥掉，'The Truth in Painting' 就会被当成人名，
    颠倒判定就整个失效（实测教训）。"""
    s = strip_tags((s or "").strip())
    if not s:
        return False
    s = RE_EDITOR.sub(" ", s)
    s = RE_NATION_TAIL.sub("", s).strip()
    if not s or ROLE_WORD.search(s):
        return False
    if re.search(r"[\(\)\[\]（）〈〉]", s):
        return False
    if AN.CJK_RE.search(s) or AN.KANA_RE.search(s) or AN.HANGUL_RE.search(s):
        return 1 <= len(s) <= 15
    if not RE_LATINISH.fullmatch(s):
        return False
    return 1 <= len(s.split()) <= 6


RE_NATION_TAIL = re.compile(
    r"[（(](?:奥|苏|古罗马|西|希腊|保加利亚|塞尔维亚|格鲁吉亚|葡|澳|黎|保|荷|"
    r"中|美|英|法|德|日|意|俄|匈|波|丹|挪|芬|瑞典|瑞士|印度|土耳其|韩国|朝鲜)[)）]$")


def is_bookish(e: str) -> bool:
    """元素整体像书名 / 出版社串 / 丛书名，而不是人名。
    国籍括号（里尔克(奥)）永远不算书名，必须先于年份括号判。"""
    e = (e or "").strip()
    if not e:
        return False
    if RE_NATION_TAIL.search(e):
        return False
    if RE_SERIES_PAREN.search(e) or RE_PUB.search(e):
        return True
    if RE_YEAR_PAREN.search(e):           # 带年份括号的元素一律当书名/出版信息
        return True
    if RE_CJK_TITLE_MARK.search(e) and len(e) >= 8 and not looks_person(e):
        return True
    return False


def drop_reason(e: str) -> str | None:
    """该元素是否属于『直接删掉』的垃圾；返回原因或 None。"""
    t = e.strip()
    if RE_PURE_DIGIT.match(t):
        return "纯数字ID"
    if RE_VOLUME.match(t):
        return "卷册号"
    if RE_DOMAIN.match(t):
        return "域名"
    return None


def clean_element(e: str) -> tuple[str, str | None]:
    """返回 (清理后的值, 触发的动作标签 or None)。值为空串表示应删除。"""
    orig = e
    e = (e or "").strip()
    if not e:
        return "", None
    act = None

    if "|" in e:                       # 'Robert T. Tally| Jr.'
        e = RE_PIPE.sub(", ", e).strip()
        act = "竖线转逗号"
    if r := drop_reason(e):
        return "", f"删[{r}]"
    e = RE_EDITOR.sub(" ", e)          # (eds.) / (editor)
    e2 = RE_EDITOR_TAIL.sub("", e).strip()
    if e2 != e:
        e, act = e2, (act or "剥编者标记")
    e = e.strip(" ,;、")
    if not e:
        return "", "删[空值]"
    if r := drop_reason(e):
        return "", f"删[{r}]"

    if is_bookish(e):                    # 书名/出版社/丛书串 → 交行级判定
        return e, (act or "书名残留")

    if AN.CJK_RE.search(e):            # 中文值里的脚注数字
        e2 = RE_FOOTNOTE_DIGIT.sub("", e)
        e2 = re.sub(r"\s+", " ", e2).strip()
        if e2 != e:
            e, act = e2, (act or "去脚注数字")

    # 含数字但去掉尾随生年后是人名 → 保留清理值
    cand = RE_TRAIL_BIRTH.sub("", RE_TRAIL_YEAR.sub("", e)).strip()
    if re.search(r"\d", e) and cand and AN.looks_like_person_name(cand):
        if cand != e:
            e, act = cand, (act or "去尾随年份")
    elif re.search(r"\d", e):
        if is_bookish(e):
            return e, "书名残留"      # 交行级判定（swap/unresolved）
        if r := drop_reason(e):
            return "", f"删[{r}]"
        return "", "删[含数字非人名]"
    return (e, act) if act else (e, None)


def split_authors(e: str) -> list[str] | None:
    """多作者拆分。返回 None 表示不拆。逗号两侧必须都是像样的人名，
    且 Jr./Sr. 后的逗号永不切（author-normalization.md 坑）。"""
    e = (e or "").strip()
    if not e:
        return None
    if (pairs := AN.parse_lastname_pairs(e)) and len(pairs) >= 2:
        return pairs
    if re.search(r"\b(?:Jr|Sr)\.?$", e):        # 后缀保护
        return None
    if any(ch in e for ch in "、；;"):
        parts = re.split(r"[、；;]", e)
    elif "," in e or "，" in e:
        parts = re.split(r"[,，]", e)
    else:
        return None
    parts = [p.strip(" .") for p in parts if p.strip(" .")]
    if len(parts) < 2:
        return None
    if not all(looks_person(p) for p in parts):
        return None
    out: list[str] = []
    for p in parts:
        p = re.sub(r"\s+(?:等|et al\.?)$", "", p).strip()
        out.extend(AN.parse_lastname_pairs(p) or
                   [AN.to_western(p) if "," in p else p])
    out = [x for x in dict.fromkeys(out) if x]
    return out if len(out) >= 2 else None


def person_from_segment(seg: str) -> list[str] | None:
    """文件名段 → 人名列表；取不到像样的人名返回 None。
    中西双名并存（大卫·哈维_ David Harvey / 克拉里, Jonathan Crary）
    按格式约定取原文名（Latin），避免同一个人占两个元素。"""
    s = strip_tags(seg.strip())
    if not s:
        return None
    if (pairs := AN.parse_lastname_pairs(s)):
        return pairs
    if "_" in s:
        ps = [p.strip() for p in s.split("_") if p.strip()]
        if len(ps) >= 2 and all(looks_person(p) for p in ps):
            return prefer_latin([AN.to_western(p) if "," in p else p for p in ps])
    if " and " in s.lower():
        ps = [p.strip() for p in re.split(r"\s+and\s+", s, flags=re.I) if p.strip()]
        if len(ps) >= 2 and all(looks_person(p) for p in ps):
            return [AN.to_western(p) if "," in p else p for p in ps]
    if "," in s:
        ps = [p.strip() for p in re.split(r"[,，]", s) if p.strip()]
        if len(ps) >= 2 and all(looks_person(p) for p in ps) \
                and not re.search(r"\b(?:Jr|Sr)\.?$", s):
            return prefer_latin([AN.to_western(p) for p in ps])
        if looks_person(s):
            return [AN.to_western(s) if AN.LATIN_NAME_RE.match(
                s.split(",", 1)[0].strip()) else s]
        return None
    return [s] if looks_person(s) else None


def prefer_latin(names: list[str]) -> list[str]:
    """同一段里既有中文名又有原文名 → 只留原文名（西方人统一原文名）。"""
    cjk = [n for n in names if AN.name_shape(n) == "cjk"]
    lat = [n for n in names if AN.name_shape(n) == "latin"]
    if cjk and lat:
        return lat
    return names


# ------------------------------------------------------------------ 主流程
def plan_row(row: dict) -> dict:
    fid, path, title = row["id"], row["path"], (row["title"] or "").strip()
    authors = [a for a in row["authors"] if a and a.strip()]

    acts: list[str] = []
    tiers: list[str] = []
    new_authors = list(authors)
    new_title = title

    # ---------- 行级：title / authors 是否颠倒 --------------------
    stem = fname_stem(path)
    segs = [s.strip() for s in stem.split(" - ")] if " - " in stem else []
    if len(segs) >= 2:
        head, tail = segs[0], " - ".join(segs[1:])
        head_clean = strip_tags(head)
        tail_clean = strip_tags(tail)
        same_book = AN.norm_title(head_clean) == AN.norm_title(
            RE_SERIES_IN_TEXT.sub("", tail_clean).strip())
        head_persons = person_from_segment(head)
        tail_persons = person_from_segment(tail)

        # 目录名是作者时（OneDrive/<作者>/、Supplementary Books/<作者>/），
        # 尾段与目录名吻合 ⇒ 文件名方向本来就对，绝不能反向互换。
        seg_p = path.split("/")
        dir_author = (seg_p[1] if len(seg_p) >= 3 and seg_p[0] in
                      ("OneDrive", "Supplementary Books", "Authors") else "")
        tail_is_dir_author = False
        if dir_author and not tail_persons:
            da, tl = AN.norm_title(strip_tags(dir_author)), AN.norm_title(tail)
            if da and (da in tl or tl in da):
                tail_is_dir_author = True
                if looks_person(strip_tags(dir_author)):
                    tail_persons = [strip_tags(dir_author)]

        if same_book:
            acts.append("unresolved[无作者书/丛书名]")
            tiers.append("T2")
        elif head_persons and not tail_persons and not tail_is_dir_author \
                and (title == head or title.startswith(head + " - ")
                     or looks_person(title)):
            # 题名段是人名、书名段不是人名 ⇒ 切分方向反了 → 互换
            new_title = tail
            new_authors = head_persons
            acts.append("swap[题名↔作者互换]")
            tiers.append("T1")
        elif title == head and not head_persons and tail_persons:
            # 方向本来就对，authors 却是书名 → 用文件名里的人名回填
            if any(is_bookish(a) for a in authors):
                new_authors = tail_persons
                acts.append("回填[文件名作者]")
                tiers.append("T1")
        elif any(is_bookish(a) for a in authors):
            acts.append("unresolved[书名残片]")
            tiers.append("T2")

    # ---------- 元素级清洗 / 拆分 ---------------------------------
    cleaned: list[str] = []
    seen_persons = set()
    # 同一行里已有人名时，书名残片属于污染 → 直接删（GRKG 之类无真作者的行
    # person_elems 为空，书名串保留并落 T2）
    person_elems = [x for x in new_authors if looks_person(x)]
    for e in new_authors:
        val, act = clean_element(e)
        if val and is_bookish(val) and person_elems and not looks_person(val):
            acts.append(f"删[书名残片与真作者并存]:{e[:40]}")
            tiers.append("T1")
            continue
        if act:
            acts.append(f"{act}:{e[:40]}")
            if act.startswith("书名残留"):
                tiers.append("T2")
                val = e
        if not val:
            tiers.append("T1")
            continue
        if is_bookish(val) and person_elems and not looks_person(val):
            acts.append(f"删[书名残片与真作者并存]:{val[:40]}")
            tiers.append("T1")
            continue
        if (split := split_authors(val)) and split != [val]:
            acts.append(f"拆分:{val[:40]}→{len(split)}人")
            tiers.append("T1")
            for p in split:
                if p not in seen_persons:
                    seen_persons.add(p)
                    cleaned.append(p)
            continue
        if (re.search(r"[()（）]", val) or re.search(r"\d", val)) \
                and not RE_NATION_TAIL.search(val):
            acts.append(f"残留[括号/数字待判]:{val[:40]}")
            tiers.append("T2")
        if val not in seen_persons:
            seen_persons.add(val)
            cleaned.append(val)

    new_authors = cleaned

    # ---------- 收尾：删空但文件名里有人 → 回填 -------------------
    if not new_authors:
        if segs:
            persons = person_from_segment(segs[-1]) or person_from_segment(segs[0])
            if persons:
                new_authors = persons
                acts.append("回填[删空后文件名作者]")
                tiers.append("T1")
        if not new_authors:
            if acts and all(re.match(r"删\[(?:纯数字ID|卷册号|域名|空值)\]", a)
                            for a in acts):
                acts.append("drop_empty[全是ID/卷册号→置空]")
                tiers.append("T1")
            else:
                # 丢掉的是书名/说明文字这类可回收信息 → 交人工，别静默清空
                acts.append("drop_empty[含可回收信息→置空]")
                tiers.append("T2")

    changed = (new_authors != authors) or (new_title != title)
    if not changed:
        if "T2" in tiers:               # 有问题但本次不动 → 上人工清单
            return {"id": str(fid), "path": path, "action": "unresolved",
                    "tier": "T2", "old_title": title, "new_title": title,
                    "old_authors": authors, "new_authors": authors,
                    "acts": acts + ["T2需人工:本次无自动改动"]}
        return {"id": str(fid), "path": path, "action": "keep", "tier": "-",
                "old_title": title, "new_title": title,
                "old_authors": authors, "new_authors": authors,
                "acts": ["keep[合规不动]"]}

    tier = "T2" if "T2" in tiers else "T1"
    return {"id": str(fid), "path": path,
            "action": (acts[0].split("[")[0] if acts else "clean"), "tier": tier,
            "old_title": title, "new_title": new_title,
            "old_authors": authors, "new_authors": new_authors,
            "acts": acts or ["元素清理"]}


async def run(apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(SCOPE_SQL)
        log.info("[scope] %d rows", len(rows))

        plans = [plan_row(dict(r)) for r in rows]
        t1 = [p for p in plans if p["tier"] == "T1"]
        t2 = [p for p in plans if p["tier"] == "T2"]
        keep = [p for p in plans if p["tier"] == "-"]

        by_act: dict[str, int] = {}
        for p in plans:
            for a in p["acts"]:
                k = a.split(":")[0]
                k = k.split("[")[0] if "[" in k else k
                by_act[k] = by_act.get(k, 0) + 1

        summary = {"applied": apply, "scope_rows": len(plans),
                   "T1_可自动改": len(t1), "T2_仅列清单": len(t2),
                   "keep_不动": len(keep), "actions": by_act}

        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/authors_quality_plan.json", "w",
                  encoding="utf-8") as f:
            json.dump({"summary": summary, "plans": plans}, f,
                      ensure_ascii=False, indent=1)
        with open("batch_operation/reports/authors_quality_plan.tsv", "w",
                  encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["tier", "action", "path", "old_title", "new_title",
                        "old_authors", "new_authors", "acts"])
            for p in sorted(plans, key=lambda x: (x["tier"], x["path"])):
                w.writerow([p["tier"], p["action"], p["path"], p["old_title"],
                            p["new_title"],
                            "{" + ",".join(p["old_authors"]) + "}",
                            "{" + ",".join(p["new_authors"]) + "}",
                            " | ".join(p["acts"])])

        if apply and t1:
            day = dt.date.today().strftime("%Y%m%d")
            bak = f"batch_operation/reports/authors_quality_backup_{day}.tsv"
            with open(bak, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["id", "path", "title", "authors"])
                for p in t1:
                    w.writerow([p["id"], p["path"], p["old_title"],
                                "{" + ",".join(p["old_authors"]) + "}"])
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2, title = $3 "
                "WHERE id = $1 AND coalesce(title,'') = $4",
                [(p["id"], p["new_authors"], p["new_title"], p["old_title"])
                 for p in t1])
            log.info("[apply] T1 updated=%d backup=%s", len(t1), bak)

        log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
        return summary
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条改动样例")
    args = ap.parse_args()
    summary = asyncio.run(run(apply=args.apply))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.show:
        data = json.load(open(
            "batch_operation/reports/authors_quality_plan.json",
            encoding="utf-8"))
        for p in data["plans"][:args.show]:
            print("-", p["tier"], p["path"])
            print("    title  :", p["old_title"], "=>", p["new_title"])
            print("    authors:", p["old_authors"], "=>", p["new_authors"])
            print("    acts   :", p["acts"])


if __name__ == "__main__":
    main()
