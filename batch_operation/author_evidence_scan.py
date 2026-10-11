"""author_evidence_scan.py — 扫描「中文作者名 → 外文原名」的证据对（只读，2026-10-11）。

四条证据源（都必须来自**同一件作品**，否则不采信）：
  A Calibre  书目 id（OneDrive/<书名> (N)/ 目录尾号）→ books/authors 的拉丁著者
  B Zotero   题名精确/前缀匹配 → 条目 creators 的拉丁著者
  C 作者目录 第 2 段本身是拉丁人名（OneDrive/Supplementary Books/Authors）
  D 题名/文件名带国别标记（〔德〕（英）[法] (苏) 等）→ 证明是外国人（但不给原名）

输出 reports/author_evidence_pairs.tsv：cjk | latin | 证据类型 | 证据强度 | 命中行数 | 样例路径
供人工审定后写进 author_unify.py 的映射表。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

log = C.setup_logging(
    "author_evidence_scan", "batch_operation/reports/author_evidence_scan.log")

LATIN_NAME = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .'\-]+(?:, [A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .'\-]+)?$")
CJK_PURE = re.compile(r"^[一-鿿·]+$")
NATION_TAIL = re.compile(
    r"[（(](奥|苏|古罗马|西|希腊|保加利亚|塞尔维亚|格鲁吉亚|葡|澳|黎|保|荷|中|美|英|法|德|日|意|俄|匈|波|丹|挪威|芬|瑞典|瑞士|印度|土耳其|韩国|朝鲜|以色列)[)）]\s*$")
NATION_MARK = re.compile(r"[〔\[（(]\s*(?:美|英|法|德|日|俄|意|西|奥|苏|荷兰|比利时|瑞典|挪威|丹麦|芬兰|波兰|捷克|匈牙利|希腊|葡萄牙|瑞士|以色列|加拿大|澳大利亚|印度|韩国|朝鲜|墨西哥|阿根廷|巴西|智利)\s*[\]）)]")
JP_KR_MARK = re.compile(r"[〔\[（(]\s*(?:日|日本|韩|韩国|朝鲜)\s*[\]）)]|日本|韩国|朝鲜")
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힯]")


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)


def fmt_creators(creators: list) -> list[str]:
    out = []
    for c in creators or []:
        first = (c.get("firstName") or "").strip()
        last = (c.get("lastName") or "").strip()
        if not first and not last:
            continue
        out.append(f"{first} {last}".strip() if first else last)
    return out


def is_person(s: str) -> bool:
    s = (s or "").strip()
    if not s or not LATIN_NAME.fullmatch(s):
        return False
    if KANA.search(s) or HANGUL.search(s):
        return False
    return 2 <= len(s.split()) <= 6


async def main() -> None:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, coalesce(title,'') AS title, authors "
            "FROM file_metadata WHERE authors IS NOT NULL AND cardinality(authors)>0")
        log.info("rows=%d", len(rows))
    finally:
        await conn.close()

    # ---------------- 证据 A：Calibre ----------------
    # 容器内从 CIFS 拷 metadata.db 会 PermissionError / `locking protocol`，
    # 改为宿主侧先导出 reports/calibre_authors.json（见本目录 _export_calibre_authors.py）
    cal_authors: dict[int, list[str]] = {}
    cal_json = "batch_operation/reports/calibre_authors.json"
    if os.path.exists(cal_json):
        cal_authors = {int(k): v for k, v in
                       json.load(open(cal_json, encoding="utf-8")).items()}
        log.info("calibre books with authors=%d", len(cal_authors))
    else:
        log.warning("缺少 %s —— Calibre 证据缺失", cal_json)

    # ---------------- 证据 B：Zotero ----------------
    zpath = "batch_operation/reports/zotero_items.json"
    z_items = json.load(open(zpath, encoding="utf-8")) if os.path.exists(zpath) else []
    z_by_title: dict[str, list[str]] = defaultdict(list)
    z_prefix: dict[str, list[str]] = defaultdict(list)
    for it in z_items:
        names = [n for n in fmt_creators(it.get("creators", [])) if is_person(n)]
        if not names:
            continue
        k = norm_title(it.get("title", ""))
        if k:
            z_by_title[k] = names
            if len(k) >= 12:
                z_prefix[k[:12]] = names
    log.info("zotero indexed=%d", len(z_by_title))

    pairs: dict[tuple[str, str], dict] = {}

    def add(cjk: str, latin: str, ev: str, path: str):
        if not cjk or not latin or cjk == latin:
            return
        key = (cjk, latin)
        d = pairs.setdefault(key, {"cjk": cjk, "latin": latin, "ev": set(),
                                   "n": 0, "sample": path,
                                   "jpkr": False})
        d["ev"].add(ev)
        d["n"] += 1
        if JP_KR_MARK.search(path) or KANA.search(latin) or HANGUL.search(latin):
            d["jpkr"] = True

    for r in rows:
        path, title, authors = r["path"], r["title"], list(r["authors"])
        cjk_vals = [a for a in authors if CJK_PURE.match(a.strip())]
        if not cjk_vals:
            continue
        seg = path.split("/")
        evid: list[tuple[str, str]] = []

        # A: Calibre
        if seg[0] == "OneDrive":
            m = re.search(r"\((\d+)\)", path)
            if m and (ca := cal_authors.get(int(m.group(1)))):
                if all(is_person(x) for x in ca):
                    evid.append(("Calibre", " ‖ ".join(ca)))
        # B: Zotero 题名
        k = norm_title(title)
        if k:
            if (zn := z_by_title.get(k)):
                evid.append(("Zotero精确", " ‖ ".join(zn)))
            elif len(k) >= 12 and k[:12] in z_prefix:
                evid.append(("Zotero前缀", " ‖ ".join(z_prefix[k[:12]])))
        # C: 目录名是拉丁人名
        if len(seg) >= 3 and seg[0] in ("OneDrive", "Supplementary Books", "Authors") \
                and is_person(seg[1]):
            evid.append(("目录名", seg[1]))

        if not evid:
            continue
        for ev, latin in evid:
            # 一个 CJK 值直接对一个原名；多值时整组对整组
            if len(cjk_vals) == 1:
                add(cjk_vals[0], latin, ev, path)
            else:
                for c in cjk_vals:
                    for L in latin.split(" ‖ "):
                        add(c, L, ev + "(组)", path)

    # ---------------- D：国别标记（只证明是外国人，不给原名） ----------------
    foreign_rows = [r for r in rows
                    if any(CJK_PURE.match(a.strip()) for a in r["authors"])
                    and (NATION_MARK.search(r["title"]) or NATION_MARK.search(r["path"])
                         or NATION_TAIL.search(" ".join(r["authors"])))]
    os.makedirs("batch_operation/reports", exist_ok=True)
    out = "batch_operation/reports/author_evidence_pairs.tsv"
    import csv
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cjk", "latin", "证据", "命中行数", "日韩嫌疑", "样例路径"])
        for (c, l), d in sorted(pairs.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow([c, l, "+".join(sorted(d["ev"])), d["n"],
                        "JPKR" if d["jpkr"] else "", d["sample"]])
    summary = {"rows": len(rows), "pairs": len(pairs),
               "cjk_values_with_evidence": len({c for c, _ in pairs}),
               "foreign_marked_rows": len(foreign_rows), "out": out}
    log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
