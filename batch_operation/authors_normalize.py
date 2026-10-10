"""file_metadata.authors 规范化（2026-10-10 用户指令，A–D 四阶段）。

  A split   用 and / & 连接的多个作者拆开（拆完回炉再过 B/C 判定）
  B long    过长作者：整条是书名的回退 Zotero/作者目录；`Last, First_…` 成对
            结构按对拆；其余按分隔符拆并逐项校验（不得与 title 重合）
  C short   过短作者（中文单姓、西文单 token）→ Zotero creators 拼全名
            （西文『名 姓』、中文『姓+名』）→ 未命中用作者目录名兜底
  D variant 同人异写统一：归一化键聚类 + 作者目录名做跨语种证据；
            西方人→原文名，日韩→简体中文

dry-run 默认；--apply 写库（先落 reports/authors_backup_20261010.tsv）。
Zotero 条目库：reports/zotero_items.json（fetch_zotero_items.py 产出）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import common as C

log = C.setup_logging(
    "authors_normalize", "batch_operation/reports/authors_normalize.log"
)

ZOTERO_ITEMS = "/app/batch_operation/reports/zotero_items.json"
ZOTERO_ATTACH = "/app/batch_operation/reports/zotero_attachments.json"

SELECT_SQL = """
    SELECT id, path, authors, coalesce(title, '') AS title
    FROM file_metadata
    WHERE authors IS NOT NULL AND array_length(authors, 1) > 0
"""

SPLIT_RE = re.compile(r"\s+(?:and|&)\s+", re.I)
YEAR_PUB_RE = re.compile(r"[-–—_]\s*[^,;]{2,40}?\s*\(\s*(?:1[5-9]|20)\d{2}\s*\)")
DIGIT_RE = re.compile(r"\d")
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
KANA_RE = re.compile(r"[぀-ヿ]")
HANGUL_RE = re.compile(r"[가-힯]")
LATIN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")
ROLE_RE = re.compile(
    r"\s*[\(\[]\s*(?:editor|editors|ed\.?|eds\.?|auth\.?|author|"
    r"translator|trans\.?|with|foreword|introduction)\s*[\)\]]\s*",
    re.I)
PUB_YEAR_TAIL_RE = re.compile(
    r"-\s*[A-Za-z][A-Za-z &\.']{1,40}\s*\(\s*(?:1[5-9]|20)\d{2}\s*\)\s*$")
# 目录名里明显不是人名的占位（不做跨语种证据）
JUNK_FOLDERS = {"未知", "unknown", "佚名", "无名", "其他", "none", "n.a.",
                "na", "tmp", "test", "未署名", "无作者", "?", "-", "佚名氏"}


def path_stem(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0]


# ------------------------------------------------------------------ 工具
def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)


def norm_name_key(s: str) -> str:
    """归一化键：去空白/标点、小写；西文 'Last, First' → 'first last'。"""
    s = unicodedata.normalize("NFKC", s or "").strip()
    if "," in s and LATIN_NAME_RE.fullmatch(s.split(",", 1)[0].strip()):
        a, b = s.split(",", 1)
        s = f"{b.strip()} {a.strip()}"
    return re.sub(r"[\s.\-'’]+", "", s.lower())


def name_shape(s: str) -> str:
    if KANA_RE.search(s):
        return "kana"
    if HANGUL_RE.search(s):
        return "hangul"
    if CJK_RE.search(s):
        return "cjk"
    if LATIN_NAME_RE.fullmatch(s or ""):
        return "latin"
    return "other"


def looks_like_person_name(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) > 60 or DIGIT_RE.search(s):
        return False
    if any(ch in s for ch in "()[]{}「」《》") or "_" in s:
        return False
    shape = name_shape(s)
    if shape == "cjk":
        return 1 <= len(s) <= 15
    if shape in ("kana", "hangul"):
        return len(s) <= 20
    if shape == "latin":
        return 1 <= len(s.split()) <= 6
    return False


def is_short(s: str) -> bool:
    """中文/日文单姓（1 字）或西文单 token（只有姓）。"""
    s = (s or "").strip()
    if not s:
        return False
    shape = name_shape(s)
    if shape in ("cjk", "kana", "hangul"):
        return len(s) <= 1
    if shape == "latin":
        return len(s.split()) == 1
    return False


def to_western(s: str) -> str:
    """'Roy, Arundhati' → 'Arundhati Roy'。"""
    s = (s or "").strip()
    if "," in s and LATIN_NAME_RE.fullmatch(s.split(",", 1)[0].strip()):
        a, b = s.split(",", 1)
        return f"{b.strip()} {a.strip()}".strip()
    return s


def folder_author(path: str) -> str | None:
    seg = path.split("/")
    top = seg[0] if seg else ""
    if top == "OneDrive" and len(seg) >= 3:
        return seg[1]
    if top == "Authors" and len(seg) >= 3:
        return to_western(seg[1])
    if top == "Supplementary Books" and len(seg) >= 3:
        return seg[1]
    return None


# ------------------------------------------------------------- Zotero 库
def title_index(items: list) -> dict:
    idx = defaultdict(list)
    for it in items:
        k = norm_title(it.get("title", ""))
        if k:
            idx[k].append(it)
    return idx


def fmt_creators(creators: list) -> list[str]:
    out = []
    for c in creators:
        first = (c.get("firstName") or "").strip()
        last = (c.get("lastName") or "").strip()
        if not first and not last:
            continue
        if not first:
            out.append(last)
        elif not last:
            out.append(first)
        elif name_shape(last + first) in ("cjk", "kana", "hangul"):
            out.append(last + first)              # 姓+名
        else:
            out.append(f"{first} {last}")          # 名 姓
    return out


def zotero_lookup(idx: dict, title: str, fuzzy: bool = True) -> list[str]:
    """按 title 查 Zotero creators。fuzzy=False 只做精确匹配（B 阶段用，
    避免 'Zum Wesen…' 这类短标题被前缀兜底误配到别人名下）。"""
    k = norm_title(title)
    if not k:
        return []
    for cand in idx.get(k, []):
        if names := fmt_creators(cand.get("creators", [])):
            return names
    if not fuzzy:
        return []
    for key, cands in idx.items():                 # 标题截断的前缀兜底（≥25 字符）
        if len(k) < 25 or len(key) < 25:
            continue
        if key.startswith(k[:25]) or k.startswith(key[:25]):
            for cand in cands:
                if names := fmt_creators(cand.get("creators", [])):
                    return names
    return []


def build_prefix_index(idx: dict) -> dict:
    """归一化标题前 12 字符 → 候选键（截断标题也能落进同一桶）。"""
    pi: dict[str, list] = defaultdict(list)
    for k in idx:
        if len(k) >= 12:
            pi[k[:12]].append(k)
        elif len(k) >= 6:
            pi[k[:6]].append(k)
    return pi


def zotero_fuzzy(idx: dict, pi: dict, title: str) -> list[str]:
    """前缀桶 + difflib 模糊（cutoff 0.86）——文件名截断的标题。"""
    import difflib
    k = norm_title(title)
    if len(k) < 8:
        return []
    for n in (12, 10, 8, 6):
        if len(k) < n:
            break
        cands = pi.get(k[:n], [])
        if not cands:
            continue
        m = difflib.get_close_matches(k, cands, n=1, cutoff=0.86)
        if m:
            for c in idx[m[0]]:
                if names := fmt_creators(c.get("creators", [])):
                    return names
    # 短标题（中文常见）：包含匹配，取长度差最小的键
    if 3 <= len(k) <= 40:
        best = None
        for key in idx:
            if k in key or (len(key) >= 3 and key in k):
                if best is None or abs(len(key) - len(k)) < abs(len(best) - len(k)):
                    best = key
        if best:
            for c in idx[best]:
                if names := fmt_creators(c.get("creators", [])):
                    return names
    return []


def zotero_by_filename(att_idx: dict, items_by_key: dict, path: str) -> list[str]:
    """按附件文件名反查父条目 creators（title 对不上时的第二条路）。"""
    fn = os.path.basename(path)
    parent = att_idx.get(path) or att_idx.get(fn)
    if parent and parent in items_by_key:
        return fmt_creators(items_by_key[parent].get("creators", []))
    # 兜底：文件名被改过（去空格/大小写）时再试一次归一化比对
    nfn = norm_title(fn)
    for a_fn, p_key in att_idx.items():
        if norm_title(a_fn) == nfn and p_key in items_by_key:
            if names := fmt_creators(items_by_key[p_key].get("creators", [])):
                return names
    return []


# ---------------------------------------------------------------- B 阶段
def parse_lastname_pairs(entry: str) -> list[str]:
    """'Kozinets, Robert V._Belk, Russell W._Fischer, Eileen' → 3 个人名。"""
    segs = [s.strip(" .;:") for s in entry.split("_") if s.strip(" .;:")]
    if len(segs) < 2 or not all("," in s for s in segs):
        return []
    names = []
    for s in segs:
        a, b = s.split(",", 1)
        if LATIN_NAME_RE.fullmatch(a.strip()) and looks_like_person_name(b.strip()):
            names.append(to_western(s))
        else:
            return []
    return names


def clean_long(entry: str, file_title: str, path: str = "") -> tuple[list[str], str]:
    e = entry.strip()
    nt, nn = norm_title(e), norm_title(file_title)
    npath = norm_title(path_stem(path)) if path else ""
    if nn and len(nn) > 8 and (nn in nt or nt in nn):
        return [], "title_junk"                    # 整条是书名 → 回退
    if PUB_YEAR_TAIL_RE.search(e) and "," not in e:
        return [], "title_junk"                    # 'Title-Publisher (2017)'
    e = ROLE_RE.sub(",", e)                        # 去 (editor)/(auth.) 等角色标记
    if names := parse_lastname_pairs(e):
        return names, "pairs"
    e = YEAR_PUB_RE.sub(",", e)
    e = re.sub(r"[_\u2013\u2014]+", ",", e)
    parts = [p.strip(" .;:") for p in re.split(r"[,;、]| and ", e) if p.strip()]
    names = []
    for p in parts:
        cand = to_western(p) if "," in p else p
        if not looks_like_person_name(cand):
            continue
        nk = norm_name_key(cand)
        if nn and nk and len(nk) > 6 and nk in nn:      # 残留书名片段
            continue
        if npath and nk and len(nk) > 6 and nk in npath:  # 文件名里的书名
            continue
        names.append(cand)
    if names:
        return names, "cleaned"
    return [], "all_junk"


# --------------------------------------------------------------- 主流程
async def run(apply: bool = False, stages: str = "A,B,C,D") -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(SELECT_SQL)
        items = json.load(open(ZOTERO_ITEMS)) if os.path.exists(ZOTERO_ITEMS) else []
        atts = (json.load(open(ZOTERO_ATTACH))
                if os.path.exists(ZOTERO_ATTACH) else [])
        tidx = title_index(items)
        pidx = build_prefix_index(tidx)
        att_idx = {a["filename"]: a["parent"] for a in atts if a.get("filename")}
        # 附件 path 也是可比对的本地路径（Books/ ↔ OneDrive/）
        for a in atts:
            path_ = a.get("path") or ""
            if not path_:
                continue
            q = path_[len("attachments:"):] if path_.startswith("attachments:") else path_
            for v in {path_, q,
                      ("OneDrive/" + q[len("Books/"):]
                       if q.startswith("Books/") else ""),
                      ("Books/" + q[len("OneDrive/"):]
                       if q.startswith("OneDrive/") else "")}:
                if v:
                    att_idx.setdefault(v, a["parent"])
        items_by_key = {it["key"]: it for it in items if it.get("key")}
        log.info("rows=%d zotero_items=%d attachments=%d",
                 len(rows), len(items), len(atts))

        buckets: Counter = Counter()
        samples: dict[str, list] = defaultdict(list)
        plan: dict[str, list] = {}

        def note(bucket, info, limit=10):
            buckets[bucket] += 1
            if len(samples[bucket]) < limit:
                samples[bucket].append(info)

        def fix_entry(entry, path, title, depth=0):
            entry = (entry or "").strip()
            if not entry:
                return []
            if "A" in stages and SPLIT_RE.search(entry):
                parts = [p.strip() for p in SPLIT_RE.split(entry) if p.strip()]
                note("A_split", (path, entry, parts))
                out = []
                for p in parts:
                    out.extend(fix_entry(p, path, title, depth + 1))
                return out
            if "B" in stages and len(entry) >= 40:
                names, how = clean_long(entry, title, path)
                if names:
                    note(f"B_{how}", (path, entry, names))
                    return names
                note(f"B_{how}", (path, entry))
                if z := zotero_lookup(tidx, title, fuzzy=False):
                    note("B_zotero", (path, entry, z))
                    return z
                fa = folder_author(path)
                if fa and looks_like_person_name(fa):
                    note("B_folder", (path, entry, fa))
                    return [fa]
                note("B_kept_unresolved", (path, entry))
                return [entry]          # 拿不到就保留原值，不静默清空
            return [entry]

        for r in rows:
            fid, path, authors, title = r["id"], r["path"], list(r["authors"]), r["title"]
            new: list[str] = []
            for e in authors:
                new.extend(fix_entry(e, path, title))
            new = [re.sub(r"\s+", " ", x).strip() for x in new if x and x.strip()]

            if "C" in stages and any(is_short(x) for x in new):
                # 强证据（标题精确命中 / 附件路径命中）直接采用；
                # 弱证据（前缀+模糊、包含匹配）须过文件名姓氏守卫。
                src, z = "exact", zotero_lookup(tidx, title, fuzzy=False)
                if not z:
                    z, src = zotero_fuzzy(tidx, pidx, title), "fuzzy"
                if not z:
                    z, src = zotero_by_filename(att_idx, items_by_key, path), "attach"
                if z and src == "fuzzy":
                    m = re.match(r"^([^_]{1,12})_(\d{4})_", os.path.basename(path))
                    if m and not re.search(re.escape(m.group(1)), " ".join(z), re.I):
                        note("C_filename_mismatch", (path, new, z))
                        z = []
                if z:
                    note("C_zotero", (path, new, z))
                    new = z
                else:
                    fa = folder_author(path)
                    if fa and len(new) == 1 and looks_like_person_name(fa) \
                            and norm_name_key(fa) != norm_name_key(new[0]):
                        note("C_folder", (path, new, fa))
                        new = [fa]
                    else:
                        note("C_unresolved", (path, new))

            new = list(dict.fromkeys(new))
            if new != list(r["authors"]):
                plan[str(fid)] = new

        # ---------------------------------------------- D 同人异写统一
        variant_map: dict[str, str] = {}
        if "D" in stages:
            freq: Counter = Counter()
            all_names: set[str] = set()
            for r in rows:
                for e in r["authors"]:
                    e = (e or "").strip()
                    if e:
                        all_names.add(e)
                        freq[e] += 1

            parent: dict[str, str] = {}

            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for n in all_names:
                find(norm_name_key(n))
            csize: dict[str, int] = {find(norm_name_key(n)): 1 for n in all_names}

            def union_capped(a: str, b: str, cap: int = 5) -> bool:
                """带簇上限的合并：防止『未知』这类目录把一串互不相干的人
                串成巨簇（实测教训：加缪→Doris Lessing）。"""
                ra, rb = find(a), find(b)
                if ra == rb:
                    return True
                sa, sb = csize.get(ra, 0), csize.get(rb, 0)
                if sa + sb > cap:
                    return False
                parent[ra] = rb
                csize[rb] = sa + sb
                return True

            # 作者目录名做跨语种证据，三条硬约束：
            #  ① 只用 OneDrive / Supplementary Books（Calibre 按著者建目录）；
            #     **排除 Authors/** —— 那棵树按主题人物组织，目录人 ≠ 著者
            #     （实测教训：Joe Hughes 被并进 Gilles Deleuze）
            #  ② 目录名与作者名都必须是像样的人名，且都不是 未知/Unknown 占位
            #  ③ 合并带簇上限（防巨簇）
            evidence_used = 0
            for r in rows:
                if r["path"].split("/")[0] == "Authors":
                    continue
                fa = folder_author(r["path"])
                if not fa or fa.strip().lower() in JUNK_FOLDERS:
                    continue
                if a0 := (next((x.strip() for x in r["authors"]
                                if x and x.strip()), "")):
                    if a0.strip().lower() in JUNK_FOLDERS:
                        continue
                if not looks_like_person_name(fa):
                    continue
                names = [x.strip() for x in r["authors"] if x and x.strip()]
                if len(names) != 1:
                    continue
                a = names[0]
                if not looks_like_person_name(a):
                    continue
                if norm_name_key(a) == norm_name_key(fa):
                    continue
                if union_capped(norm_name_key(a), norm_name_key(fa)):
                    evidence_used += 1
            note("D_evidence_used", evidence_used)

            clusters: dict[str, set] = defaultdict(set)
            for n in all_names:
                clusters[find(norm_name_key(n))].add(n)

            def choose(members: set) -> str:
                persons = [m for m in members if looks_like_person_name(m)]
                if persons:
                    members = set(persons)
                shapes = {name_shape(m) for m in members}
                if "latin" in shapes:                       # 西方人 → 原文名
                    cands = [m for m in members if name_shape(m) == "latin"]
                elif shapes & {"cjk", "kana", "hangul"}:     # 日韩 → 简体中文
                    cjk = [m for m in members if name_shape(m) == "cjk"]
                    cands = cjk or [m for m in members
                                    if name_shape(m) in ("kana", "hangul")]
                else:
                    cands = list(members)
                return sorted(cands, key=lambda m: (-freq[m], -len(m)))[0]

            multi = {k: v for k, v in clusters.items() if len(v) > 1}
            for members in multi.values():
                canon = choose(members)
                for m in members:
                    if m != canon:
                        variant_map[m] = canon
            note("D_clusters", {"clusters": len(multi), "variants": len(variant_map),
                                "cross_script": sum(
                                    1 for m in variant_map
                                    if name_shape(m) != name_shape(variant_map[m]))})

        if variant_map:
            for r in rows:
                fid = str(r["id"])
                cur = plan.get(fid, [x.strip() for x in r["authors"]
                                     if x and x.strip()])
                conv = list(dict.fromkeys(variant_map.get(x, x) for x in cur))
                if conv != cur:
                    plan[fid] = conv
                    note("D_applied", (r["path"], cur, conv))

        summary = {"applied": apply, "rows_scanned": len(rows),
                   "files_to_update": len(plan), "buckets": dict(buckets),
                   "zotero_items": len(items)}
        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/authors_plan.json", "w",
                  encoding="utf-8") as f:
            json.dump({"summary": summary, "plan": plan,
                       "variant_map": variant_map,
                       "samples": {k: v[:10] for k, v in samples.items()}},
                      f, ensure_ascii=False, indent=1)

        if apply and plan:
            with open("batch_operation/reports/authors_backup_20261010.tsv",
                      "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["id", "path", "authors"])
                for r in rows:
                    if str(r["id"]) in plan:
                        w.writerow([r["id"], r["path"],
                                    "{" + ",".join(r["authors"]) + "}"])
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2 WHERE id = $1",
                [(k, v) for k, v in plan.items()])
            log.info("[apply] updated %d rows", len(plan))

        log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
        for b in sorted(buckets):
            log.info("[%s] %d e.g. %s", b, buckets[b],
                     json.dumps(samples[b][:2], ensure_ascii=False)[:400])
        return summary
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--stages", default="A,B,C,D")
    args = ap.parse_args()
    import asyncio
    asyncio.run(run(apply=args.apply, stages=args.stages))


if __name__ == "__main__":
    main()
