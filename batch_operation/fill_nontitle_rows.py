"""按文件名回填所有 title 为空的 PDF 行（2026-10-09 用户指令）。

item_type（当前为 NULL 才写）：
  Journals / Series → `proceedings`（合订本，暂且这么对应）
  Zotero            → `journalArticle`
  其余（Authors/Supplementary Books/`..`/…）→ `book`；Albums → `multimedia`
title 解析：
  Zotero  : `作者_…_YYYY_题名.pdf` → 题名取年份之后**原样**，作者取年份之前的 _ 段
  其余    : 文件名含 ` - ` → 题名=前段、作者=后段（Calibre 约定）；否则整名作题名
  OneDrive: 书名目录去 ` (id)`，作者=上级目录（与 fix_onedrive_empty_titles 同逻辑）
另写 `publication`=子目录名（Journals 刊名）、`series`=子目录名（Series 丛书名）。
幂等（谓词 `coalesce(btrim(title), '')=''`）、分批事务；dry-run 默认。

  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/fill_nontitle_rows.py [--apply] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys

import common as C

log = C.setup_logging("fill_nontitle", "batch_operation/reports/fill_nontitle.log")
OUT = "batch_operation/reports/fill_nontitle.json"

ITEM_TYPE = {"Journals": "proceedings", "Series": "proceedings",
             "Zotero": "journalArticle", "Albums": "multimedia"}
YEAR = re.compile(r"^\d{4}$")
BATCH = 2000


def parse(path: str) -> dict:
    seg = path.split("/")
    top = seg[0]
    base = seg[-1]
    stem = re.sub(r"\.(pdf|epub|djvu)$", "", base, flags=re.I)
    out: dict = {"item_type": ITEM_TYPE.get(top, "book")}

    if top == "Zotero":
        parts = stem.split("_")
        yi = next((i for i, t in enumerate(parts)
                   if i > 0 and YEAR.match(t) and 1000 <= int(t) <= 2099), None)
        if yi is not None:
            head = "_".join(parts[:yi + 1])
            title = stem[len(head) + 1:].strip()
            out["title"] = title or stem
            out["authors"] = [p for p in parts[:yi] if p]
        else:
            out["title"] = stem
    elif top == "OneDrive" and len(seg) >= 4:
        out["title"] = re.sub(r"\s*\(\d+\)\s*$", "", seg[2]).strip() or stem
        out["authors"] = [seg[1]]
        out["item_type"] = "book"
    elif " - " in stem:
        t, a = stem.rsplit(" - ", 1)
        out["title"] = t.strip() or stem
        if a.strip():
            out["authors"] = [a.strip()]
    else:
        out["title"] = stem

    if top == "Journals" and len(seg) >= 3:
        out["publication"] = seg[1]
    elif top == "Series" and len(seg) >= 3:
        out["series"] = seg[1]
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    conn = await C.get_conn()
    q = ("SELECT path FROM file_metadata "
         "WHERE lower(path) LIKE '%.pdf' AND coalesce(btrim(title), '') = '' "
         "ORDER BY path")
    rows = await conn.fetch(q)
    paths = [r["path"] for r in rows]
    if args.limit:
        paths = paths[:args.limit]
    by_dir = collections.Counter(p.split("/")[0] for p in paths)
    log.info("[plan] %d empty-title rows; by dir: %s", len(paths),
             json.dumps(dict(by_dir), ensure_ascii=False))

    parsed = [(p, parse(p)) for p in paths]
    bad = sum(1 for _p, d in parsed if not d.get("title"))
    applied = {"updated": 0, "failed": 0, "empty_title_after": bad}
    if args.apply:
        for i in range(0, len(parsed), BATCH):
            chunk = parsed[i:i + BATCH]
            try:
                sql = (
                    "UPDATE file_metadata SET title=$2, "
                    "item_type = COALESCE(item_type, $3::varchar), "
                    "authors = CASE WHEN authors IS NULL OR cardinality(authors)=0 "
                    "               THEN $4::text[] ELSE authors END, "
                    "publication = COALESCE(publication, $5::varchar), "
                    "series = COALESCE(series, $6::varchar) "
                    "WHERE path=$1 AND coalesce(btrim(title), '')=''")
                data = [(path, d["title"], d.get("item_type"),
                         d.get("authors") or [], d.get("publication"),
                         d.get("series")) for path, d in chunk]
                async with conn.transaction():
                    await conn.executemany(sql, data)
                applied["updated"] += len(data)
            except Exception as ex:
                applied["failed"] += len(chunk)
                log.warning("[upd] batch %d failed: %s", i, str(ex)[:200])
            log.info("[upd] %d/%d", i + len(chunk), len(parsed))

    left = await conn.fetchval(
        "SELECT count(*) FROM file_metadata "
        "WHERE lower(path) LIKE '%.pdf' AND coalesce(btrim(title), '') = ''")
    it = await conn.fetch(
        "SELECT coalesce(item_type,'(null)') t, count(*) FROM file_metadata "
        "WHERE lower(path) LIKE '%.pdf' GROUP BY 1 ORDER BY 2 DESC")
    summary = {"mode": "apply" if args.apply else "dry-run",
               "plan_by_dir": dict(by_dir), "applied": applied,
               "empty_title_after": int(left),
               "item_type_dist": {r["t"]: int(r["count"]) for r in it}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log.info("[done] %s", json.dumps(summary, ensure_ascii=False))
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
