"""Backfill file_metadata for OneDrive rows with an empty title.

Target: `path LIKE 'OneDrive/%' AND coalesce(btrim(title),'')=''`
(all of them also have item_type NULL).  For each row:
  * item_type := 'book'
  * title/authors := Calibre metadata.db when the book id `(N)` at the end of
    the title directory resolves (title + ordered authors);
  * otherwise inferred from the path: author = path component 2
    (`OneDrive/<author>/<title (id)>/<file>`), title = the title directory
    stripped of its trailing ` (id)` (filename stem `Title - Author` as the
    last resort).

Safety: dry-run by default, single transaction on --apply, only touches rows
whose title is still empty (idempotent — a second run finds 0 targets).
metadata.db is opened read-only.

  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/fix_onedrive_empty_titles.py            # dry-run
  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/fix_onedrive_empty_titles.py --apply    # write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, "/app")
import common as C

log = C.setup_logging("empty_titles", "batch_operation/reports/empty_titles.log")

STORAGE = "/storage"
CALIBRE_DB = os.path.join(STORAGE, "OneDrive", "metadata.db")

TARGET_SQL = """
    SELECT id, path FROM file_metadata
     WHERE path LIKE 'OneDrive/%'
       AND coalesce(btrim(title), '') = ''
     ORDER BY path
"""
CALIBRE_SQL = """
    SELECT b.id, b.title,
           COALESCE(GROUP_CONCAT(a.name, char(31)), '')
      FROM books b
      LEFT JOIN books_authors_link bal ON bal.book = b.id
      LEFT JOIN authors a ON a.id = bal.author
     GROUP BY b.id, b.title
"""
UPDATE_SQL = """
    UPDATE file_metadata
       SET title = $2,
           authors = CASE WHEN authors IS NULL OR cardinality(authors) = 0
                          THEN $3::text[] ELSE authors END,
           item_type = 'book'
     WHERE id = $1 AND coalesce(btrim(title), '') = ''
"""

BOOK_ID_RE = re.compile(r"\((\d+)\)\s*$")
TITLE_ID_RE = re.compile(r"\s*\(\d+\)\s*$")


def calibre_index() -> dict[int, tuple[str, list[str]]]:
    con = sqlite3.connect(f"file:{CALIBRE_DB}?mode=ro", uri=True)
    try:
        rows = con.execute(CALIBRE_SQL).fetchall()
    finally:
        con.close()
    out: dict[int, tuple[str, list[str]]] = {}
    for bid, title, blob in rows:
        authors = [a for a in (blob or "").split("\x1f") if a]
        out[int(bid)] = (title or "", authors)
    return out


def infer(path: str, calibre: dict[int, tuple[str, list[str]]]) -> dict:
    """Return {source, title, authors, book_id} for one file_metadata.path."""
    parts = path.split("/")
    author_dir = parts[1] if len(parts) >= 2 else ""
    title_dir = parts[2] if len(parts) >= 3 else ""
    stem = os.path.splitext(os.path.basename(path))[0]

    book_id = None
    m = BOOK_ID_RE.search(title_dir)
    if m:
        book_id = int(m.group(1))

    if book_id is not None and book_id in calibre:
        title, authors = calibre[book_id]
        if title:
            return {"source": "calibre", "title": title,
                    "authors": authors or ([author_dir] if author_dir else []),
                    "book_id": book_id}

    title = TITLE_ID_RE.sub("", title_dir).strip()
    src = "path-dir"
    if not title:
        # filename stem `Title - Author` / `Title`
        title = stem.rsplit(" - ", 1)[0].strip() or stem
        src = "filename"
    authors = [author_dir] if author_dir else []
    return {"source": src, "title": title, "authors": authors,
            "book_id": book_id}


async def run(apply: bool, limit: int | None) -> dict:
    calibre = calibre_index()
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(TARGET_SQL)
        if limit:
            rows = rows[:limit]
        plan = []
        for r in rows:
            inf = infer(r["path"], calibre)
            plan.append({"id": str(r["id"]), "path": r["path"], **inf})

        by_src: dict[str, int] = {}
        for p in plan:
            by_src[p["source"]] = by_src.get(p["source"], 0) + 1
        no_title = [p for p in plan if not p["title"]]
        meta = {
            "targets": len(plan),
            "by_source": by_src,
            "empty_title_after_infer": len(no_title),
            "calibre_index_size": len(calibre),
            "applied": False,
        }
        log.info("plan: %s", json.dumps(meta, ensure_ascii=False))

        if apply and plan:
            async with conn.transaction():
                for p in plan:
                    if not p["title"]:
                        continue
                    await conn.execute(
                        UPDATE_SQL, p["id"], p["title"], p["authors"]
                    )
            meta["applied"] = True
            meta["updated"] = sum(1 for p in plan if p["title"])
            # read-back
            left = await conn.fetchval(
                "SELECT count(*) FROM file_metadata "
                "WHERE path LIKE 'OneDrive/%' AND coalesce(btrim(title),'')=''"
            )
            meta["targets_remaining_empty"] = left
            log.info("applied: %s", json.dumps(meta, ensure_ascii=False))
        C.dump_json("batch_operation/reports/empty_titles.json",
                    {"meta": meta, "sample": plan[:20]})
        return {"meta": meta, "sample": plan[:20]}
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write (default dry-run)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = asyncio.run(run(args.apply, args.limit))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
