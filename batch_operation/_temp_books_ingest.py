#!/usr/bin/env python3
"""workspace/Temp/书籍 → jindai 库：只搬「库里没有」的书，搬完登记进 file_metadata。

流程（--apply 才动文件）：
  1) 列源文件；清洗 Z-Library 噪声得候选题名；
  2) 与书库（file_metadata title/文件名）比对：精确 或「前缀互含≥6字」或「互含且较短侧
     ≥较长侧 50%」→ 判已有；否则缺失；
  3) 缺失集内去重（pdf > epub > mobi/docx > rar，同题只留一份）；词典类改投 Tool Books/；
  4) 解析 作者/书名 → 目标 `OneDrive/<作者>/<书名> (<id>)/<原名>`，id 从 20400 起；
  5) shutil.move 搬运 + INSERT file_metadata（title/authors/extension/size/item_type）。

只读比对用 --dry-run（默认）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import unicodedata

SRC = "/mnt/repo/workspace/Temp/书籍"
LIB = "/home/zhuth/.hermes/profiles/collector/cache/scratch/lib_titles.tsv"
STORAGE = "/misc/Public/jindai-storage"          # 容器 /storage 的宿主挂载
REPORT = "/home/zhuth/.hermes/profiles/collector/cache/scratch/temp_books_plan.json"
MIN_LEN = 6
TOOL_RE = re.compile(r"词典|辭典|字典|dictionary|dictionar|lexicon|wörterbuch|woerterbuch|"
                     r"glossar|thesaurus", re.I)
FMT_PREF = {"pdf": 0, "epub": 1, "mobi": 2, "azw3": 2, "docx": 3, "txt": 3, "rar": 9}
NEXT_ID = 20400

NOISE = [
    r"\[zhelper\.org\]\s*", r"\[Z-Library\]\s*", r"\[z-lib\.org\]\s*",
    r"\(z-lib\.org\)", r"\(Z-Library\)", r"\(https___zh\.z-lib\.org\)",
    r"https___zh\.z-lib\.org", r"\(elib\.cc\)", r"ePUBw\.COM", r"elib\.cc",
    r"\[zhelper\]", r"z-lib\.org", r"\( etc\.?\)", r"\(etc\.?\)",
    r"\(vol\.?_?no\.?\)",
]


def strip_ext(name: str) -> str:
    return re.sub(r"\.(pdf|epub|mobi|docx|rar|azw3?)$", "", name, flags=re.I)


def clean(stem: str) -> str:
    s = unicodedata.normalize("NFKC", stem)
    s = re.sub(r"^\[\d+\]\s*", "", s)
    for pat in NOISE:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"\s*\(https?__?_?[^\)]{0,80}\)", " ", s, flags=re.I)
    s = re.sub(r"\s*\([^\)]{0,12}z[- ]?lib[^\)]{0,60}\)", " ", s, flags=re.I)
    s = re.sub(r"\.(pdf|epub)_[^ ]{0,120}\.(pdf|epub)", " ", s, flags=re.I)  # 重复残片
    s = re.sub(r"\s*[（(]\s*[^（()）]{2,120}\.(pdf|epub)\s*[)）]", " ", s, flags=re.I)
    s = re.sub(r"\s*\.(pdf|epub)\s*$", "", s, flags=re.I)                 # 题名里残留扩展名
    s = re.sub(r"[_]{2,}", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ._-")
    return s


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^\w一-鿿]+", "", s)


def same_work(a: str, b: str) -> bool:
    """题名归一化后判同一本书：精确 / 短侧是长侧前缀(≥6字) / 互含且较短≥50%。"""
    if not a or not b:
        return False
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    if len(lo) < MIN_LEN:
        return False
    if hi.startswith(lo):
        return True
    if lo in hi or hi in lo:
        return len(lo) >= 8 and len(lo) / len(hi) >= 0.5
    return False


# ---------------- 书库索引
lib_titles, lib_paths, lib_tkeys = {}, {}, []
with open(LIB, encoding="utf-8") as f:
    for line in f:
        if "\t" not in line:
            continue
        title, path = line.rstrip("\n").split("\t", 1)
        base = strip_ext(path.rsplit("/", 1)[-1])
        if title:
            lib_titles.setdefault(norm(title), path)
            lib_tkeys.append((norm(title), path))
        if base:
            lib_paths.setdefault(norm(base), path)
print(f"library: {len(lib_titles)} titles / {len(lib_paths)} basenames / "
      f"{len(lib_tkeys)} entries")

# 前缀桶加速（候选题名查库题名前缀方向）
buckets = collections.defaultdict(list)
for nt, p in lib_tkeys:
    if len(nt) >= MIN_LEN:
        buckets[nt[:3]].append((nt, p))


def in_library(cleaned: str, stem: str):
    nc, ns = norm(cleaned), norm(stem)
    if nc in lib_titles:
        return "exact_title", lib_titles[nc]
    if ns and ns in lib_paths:
        return "exact_file", lib_paths[ns]
    if len(nc) >= MIN_LEN:
        seen = set()
        for i in range(len(nc) - 2):
            for nt, p in buckets.get(nc[i:i + 3], ()):
                if nt in seen:
                    continue
                seen.add(nt)
                if same_work(nt, nc):
                    return "contain", p
        for nt, p in lib_tkeys:                 # 中段互含兜底
            if same_work(nt, nc):
                return "contain", p
    return None, None


# ---------------- 作者/题名解析
BAD_AUTHOR = re.compile(
    r"出版社|出版|書局|书局|译|譯|主编|总编|編|编|著|教材|系列|丛书|文库|译丛|研究组|课题组|"
    r"研究院|编辑部|协会|大学|学院|公司|机构|图片库|健康組|飲食組|作者|著者|译者|"
    r"press|university|institute|20\d\d|19\d\d|第?\s*\d+\s*版|etc\b|ffilm",
    re.I)


def parse_author(stem: str) -> tuple[str, str]:
    """返回 (author, title)。只认三种可靠形态，判不准就 '未知'（宁缺毋滥）。"""
    s = stem.strip()
    s = re.sub(r"^\d{1,2}\s*[-–—]\s*", "", s)                          # 1-韦伯：…
    s = re.sub(r"^[（(][\u4e00-\u9fff]{1,3}[)）]\s*", "", s)           # （德）尼采：…
    m = re.match(r"^(?P<a>[^—-]{2,30}?)\s*--\s*(?P<t>.+)$", s)          # 作者--题名
    if m and re.search(r"[\u4e00-\u9fffA-Za-z]", m.group("a")):
        return m.group("a").strip(), m.group("t").strip()
    m = re.match(r"^【(?P<a>[^】]{2,20})】(?P<t>.+)$", s)                  # 【作者】题名
    if m:
        return m.group("a").strip(), m.group("t").strip()
    # 注意：**不用**「作者：题名」规则——中文题名里冒号太常见（传播与社会：…、
    # 免费文化：…、反美学：…），左侧常常是题名词而非人名，误判会污染作者目录。
    m = re.match(r"^(?P<t>.+?)\s*[（(\[](?P<a>[^)）\]]{2,40})[)）\]]\s*$", s)  # 题名 (作者)
    if m and not BAD_AUTHOR.search(m.group("a")) and "（" not in m.group("a") \
            and "(" not in m.group("a") and len(m.group("a")) <= 30:
        a = re.sub(r"^\[[\u4e00-\u9fff]{1,4}\]\s*", "", m.group("a")).strip(" ·．.,，")
        if len(a) >= 2:
            return a, m.group("t").strip()
    m = re.match(r"^(?P<t>.+?)\s*[_\-—]\s*(?P<a>[\u4e00-\u9fff·]{2,8})$", s)  # 题名_作者
    if m and len(m.group("t")) >= 4 and not BAD_AUTHOR.search(m.group("a")):
        t = re.sub(r"\s*[_\-—]\s*" + re.escape(m.group("a")) + r"$", "", m.group("t"))
        return m.group("a"), t.strip()
    return "未知", s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = []
    for dirpath, _d, fns in os.walk(SRC):
        for fn in fns:
            full = os.path.join(dirpath, fn)
            stem = strip_ext(fn)
            ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
            cleaned = clean(stem)
            level, hit = in_library(cleaned, stem)
            files.append({"full": full, "rel": os.path.relpath(full, SRC),
                          "fn": fn, "ext": ext, "cleaned": cleaned,
                          "level": level, "hit": hit,
                          "size": os.path.getsize(full)})

    # 缺失集内去重：按题名互含分组，组内按 格式偏好+大小 选一个
    missing = [f for f in files if not f["level"]]
    missing.sort(key=lambda f: (FMT_PREF.get(f["ext"], 5), -f["size"]))
    keep, dropped = [], []
    for f in missing:
        nc = norm(f["cleaned"])
        dup = next((k for k in keep if same_work(norm(k["cleaned"]), nc)), None)
        if not dup and f["ext"] == "rar" and len(nc) >= 3:
            # 归档包（如 十二国EPUB.rar）：源里已有同前缀单册 → 视为重复
            dup = next((k for k in keep
                        if norm(k["cleaned"])[:3] == nc[:3]
                        or nc[:3] in norm(k["cleaned"])), None)
        if dup:
            f["dup_of"] = dup["fn"]
            dropped.append(f)
        else:
            keep.append(f)

    plan = []
    seq = NEXT_ID
    for f in keep:
        toolbook = bool(TOOL_RE.search(f["fn"]))
        if toolbook:
            author, title = "Tool Books", strip_ext(f["fn"])
            sub = "英语" if re.search(r"牛津|朗文|习语|英语|oxford|longman", f["fn"], re.I) \
                else "词典"
            target_dir = f"{STORAGE}/Tool Books/{sub}"
            target = f"{target_dir}/{f["fn"]}"
        else:
            author, title = parse_author(f["cleaned"])
            target_dir = f"{STORAGE}/OneDrive/{author}/{title} ({seq})"
            target = f"{target_dir}/{f['fn']}"
            seq += 1
        plan.append({**f, "author": author, "title": title,
                     "target_dir": target_dir, "target": target,
                     "toolbook": toolbook, "id": None if toolbook else seq - 1})

    json.dump({"plan": [{k: v for k, v in p.items() if k != "full"} for p in plan],
               "already": [{"fn": f["fn"], "level": f["level"], "hit": f["hit"]}
                           for f in files if f["level"]],
               "source_dup": [{"fn": f["fn"], "dup_of": f["dup_of"]}
                              for f in dropped]},
              open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n总计 {len(files)} | 已有 {sum(1 for f in files if f['level'])} | "
          f"源内重复 {len(dropped)} | 待搬 {len(plan)}")
    print("\n== 待搬计划（作者/书名/目标目录）==")
    for p in plan:
        tag = "TOOL" if p["toolbook"] else p["id"]
        print(f"[{tag}] {p['ext']:4s} {p['size']:>10}  {p['title'][:46]}"
              f"  ← {p['author']}  | {p['fn'][:52]}")

    if not args.apply:
        print("\n(dry-run，未动文件)")
        return 0

    # ---------------- 搬运 + 登记
    import subprocess
    ok, fail = 0, 0
    rows = []
    for p in plan:
        try:
            os.makedirs(p["target_dir"], exist_ok=True)
            if os.path.exists(p["target"]):
                raise FileExistsError(p["target"])
            shutil.move(p["full"], p["target"])
            ok += 1
        except Exception as ex:
            fail += 1
            print(f"[MOVE FAIL] {p['fn']}: {ex}")
            continue
        rel = os.path.relpath(p["target"], STORAGE).replace(os.sep, "/")
        rows.append((rel, p["ext"] or None, p["size"],
                     None if p["toolbook"] else "book",
                     p["title"],
                     [p["author"]] if not p["toolbook"] and p["author"] != "未知" else None))
    print(f"\nmoved ok={ok} fail={fail}; 待登记 {len(rows)} 行")
    json.dump(rows, open(REPORT + ".rows.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
