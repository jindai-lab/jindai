"""按文件名回填 title 为空的 PDF 行（2026-10-09 用户指令；2026-10-29 从
``batch_operation/fill_nontitle_rows.py`` 迁入本包）。

为什么在 jindai 里：磁盘登记钩子（``.import_lists.scan_disk_upsert``）登记完新行
必须立刻回填空标题（2026-10-26 事故：引擎一次登记 8,234 行后没人回填，空标题挂了
一上午）。核心包**不能** subprocess 去拉 ``batch_operation/`` 下的脚本——那是运维
脚本目录（且大部分在 .gitignore 里）；顺带修正：旧实现漏了 ``import sys``，
``sys.executable`` 必抛 NameError 并被 except 吞掉，该补丁其实一直静默失效。

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

CLI（容器内）：
    python -m jindai.maintenance.fill_nontitle [--apply] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

from .import_lists import get_conn

log = logging.getLogger("jindai.import")

ITEM_TYPE = {"Journals": "proceedings", "Series": "proceedings",
             "Zotero": "journalArticle", "Albums": "multimedia"}
YEAR = re.compile(r"^\d{4}$")
BATCH = 2000


def parse_path(path: str) -> dict:
    """从存储相对路径推断 title / authors / item_type / publication / series。"""
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


async def fill_nontitle(
    out_path: str | Path,
    conn: Any = None, *, apply: bool = False, limit: int = 0,
) -> dict:
    """回填空 title 的 PDF 行。

    Args:
        conn: 复用已有 asyncpg 连接（None = 自建并在结束时关闭）。
        apply: False 只出计划（dry-run）。
        limit: 只处理前 N 行（抽样试跑）。
        out_path: summary 落点，默认 ``REPORTS/fill_nontitle.json``。

    Returns:
        summary dict（同时写入 out_path）。
    """
    own_conn = conn is None
    if own_conn:
        conn = await get_conn()
    try:
        q = ("SELECT path FROM file_metadata "
             "WHERE lower(path) LIKE '%.pdf' AND coalesce(btrim(title), '') = '' "
             "ORDER BY path")
        rows = await conn.fetch(q)
        paths = [r["path"] for r in rows]
        if limit:
            paths = paths[:limit]
        by_dir = collections.Counter(p.split("/")[0] for p in paths)
        log.info("[plan] %d empty-title rows; by dir: %s", len(paths),
                 json.dumps(dict(by_dir), ensure_ascii=False))

        parsed = [(p, parse_path(p)) for p in paths]
        bad = sum(1 for _p, d in parsed if not d.get("title"))
        applied = {"updated": 0, "failed": 0, "empty_title_after": bad}
        if apply:
            sql = (
                "UPDATE file_metadata SET title=$2, "
                "item_type = COALESCE(item_type, $3::varchar), "
                "authors = CASE WHEN authors IS NULL OR cardinality(authors)=0 "
                "               THEN $4::text[] ELSE authors END, "
                "publication = COALESCE(publication, $5::varchar), "
                "series = COALESCE(series, $6::varchar) "
                "WHERE path=$1 AND coalesce(btrim(title), '')=''")
            for i in range(0, len(parsed), BATCH):
                chunk = parsed[i:i + BATCH]
                try:
                    data = [(path, d["title"], d.get("item_type"),
                             d.get("authors") or [], d.get("publication"),
                             d.get("series")) for path, d in chunk]
                    async with conn.transaction():
                        await conn.executemany(sql, data)
                    applied["updated"] += len(data)
                except Exception as ex:  # noqa: BLE001 - 单批失败不打断整轮回填
                    applied["failed"] += len(chunk)
                    log.warning("[upd] batch %d failed: %s", i, str(ex)[:200])
                log.info("[upd] %d/%d", i + len(chunk), len(parsed))

        left = await conn.fetchval(
            "SELECT count(*) FROM file_metadata "
            "WHERE lower(path) LIKE '%.pdf' AND coalesce(btrim(title), '') = ''")
        it = await conn.fetch(
            "SELECT coalesce(item_type,'(null)') t, count(*) FROM file_metadata "
            "WHERE lower(path) LIKE '%.pdf' GROUP BY 1 ORDER BY 2 DESC")
        summary = {"mode": "apply" if apply else "dry-run",
                   "plan_by_dir": dict(by_dir), "applied": applied,
                   "empty_title_after": int(left),
                   "item_type_dist": {r["t"]: int(r["count"]) for r in it}}
    finally:
        if own_conn:
            await conn.close()

    target = Path(out_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    except OSError as ex:  # 报表写不了不该让回填结果丢失
        log.warning("[done] cannot write %s: %s", target, ex)
    log.info("[done] %s", json.dumps(summary, ensure_ascii=False))
    return summary


def fill_nontitle_sync(*, apply: bool = True, limit: int = 0,
                       out_path: Optional[str | Path] = None) -> dict:
    """同步入口：给 ``asyncio.to_thread`` / 运维脚本用（内部自建事件循环）。"""
    return asyncio.run(fill_nontitle(apply=apply, limit=limit, out_path=out_path))


async def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="真的写库（默认只出计划）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()
    summary = await fill_nontitle(out_path=args.out, apply=args.apply, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
