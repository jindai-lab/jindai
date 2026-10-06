"""Re-resolve file_metadata rows whose stored path no longer exists on disk.

The 2026-10 audit found ~700 pending OneDrive PDF rows whose `file_metadata.path`
points at an author folder that has since been renamed (e.g. DB
`OneDrive/A.S.爱丁顿/物理世界的本质… (372)/…` vs disk
`OneDrive/A.S. Eddington/物理世界的本质… (372)/…`).  The pipeline matches paths
EXACTLY, so those rows are never imported and the pending counter never falls.

Fix, per stale row (all rows considered here have ZERO paragraphs, so no
paragraph history is at risk):
  rename  : a live Calibre file_path exists for the same book id `(N)` and no
            other file_metadata row holds that path -> UPDATE path in place
            (keeps dataset memberships).
  dup     : a live path exists but ANOTHER row already holds it -> stale row is
            a duplicate: move file_dataset memberships to the survivor, then
            DELETE the stale row (only when it has 0 paragraphs AND 0 queue rows).
  gone    : no live Calibre row / live file missing on disk -> DELETE the stale
            row under the same zero-reference guard; report these separately.

Safety: dry-run by default, single transaction, per-row guard re-checked right
before each write.  Calibre metadata.db is opened read-only.

  python batch_operation/fix_path_drift.py            # dry-run plan
  python batch_operation/fix_path_drift.py --apply    # write
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

log = C.setup_logging("path_drift", "batch_operation/reports/path_drift.log")

STORAGE = "/storage"
CALIBRE_DB = os.path.join(STORAGE, "OneDrive", "metadata.db")

PENDING_SQL = """
    SELECT f.id, f.path FROM file_metadata f
    WHERE f.path LIKE 'OneDrive/%' AND lower(f.path) LIKE '%.pdf'
      AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
    ORDER BY f.path
"""
PARA_SQL = "SELECT count(*) FROM paragraph WHERE source_id = $1"
QUEUE_SQL = "SELECT count(*) FROM embedding_pending_queue WHERE source_id = $1"
QUEUE_BY_ID_SQL = "SELECT count(*) FROM embedding_pending_queue WHERE id = $1"
DUP_SQL = "SELECT id FROM file_metadata WHERE path = $1"
MOVE_MEMBERS_SQL = """
    INSERT INTO file_dataset (file_id, dataset_id)
    SELECT $2, dataset_id FROM file_dataset WHERE file_id = $1
    ON CONFLICT DO NOTHING
"""
DROP_MEMBERS_SQL = "DELETE FROM file_dataset WHERE file_id = $1"
DELETE_FM_SQL = "DELETE FROM file_metadata WHERE id = $1"
RENAME_SQL = "UPDATE file_metadata SET path = $2 WHERE id = $1"

BOOK_ID_RE = re.compile(r"\((\d+)\)/[^/]+$")


def live_pdf_index() -> dict[int, str]:
    """book_id -> live `OneDrive/<author>/<title (id)>/<stem>.pdf` (exists on disk)."""
    con = sqlite3.connect(f"file:{CALIBRE_DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            SELECT b.id, b.path, d.name, d.format
              FROM books b JOIN data d ON d.book = b.id
             WHERE upper(d.format) = 'PDF'
            """
        ).fetchall()
    finally:
        con.close()
    out: dict[int, str] = {}
    for bid, bpath, name, _fmt in rows:
        rel = f"OneDrive/{bpath}/{name}.pdf"
        if os.path.exists(os.path.join(STORAGE, rel)):
            out[int(bid)] = rel
    return out


async def plan(conn) -> dict:
    live = live_pdf_index()
    rows = await conn.fetch(PENDING_SQL)
    stale = [r for r in rows if not os.path.exists(os.path.join(STORAGE, r["path"]))]
    out: dict[str, list] = {"rename": [], "dup": [], "gone": [], "uncertain": []}
    for r in stale:
        fid, path = r["id"], r["path"]
        m = BOOK_ID_RE.search(path)
        book_id = int(m.group(1)) if m else None
        new_path = live.get(book_id) if book_id is not None else None
        if not new_path:
            out["gone"].append({"id": str(fid), "path": path, "reason": "no live calibre file"})
            continue
        owner = await conn.fetchval(DUP_SQL, new_path)
        if owner is None:
            out["rename"].append({"id": str(fid), "from": path, "to": new_path})
        elif str(owner) == str(fid):
            out["uncertain"].append({"id": str(fid), "path": path, "reason": "row already owns live path"})
        else:
            out["dup"].append({"id": str(fid), "path": path, "survivor": str(owner), "to": new_path})
    meta = {
        "pending_total": len(rows),
        "stale": len(stale),
        "live_calibre_pdfs": len(live),
    }
    return {"meta": meta, **out}


async def guarded_delete(conn, fid, label, survivor=None) -> bool:
    paras = await conn.fetchval(PARA_SQL, fid)
    q_src = await conn.fetchval(QUEUE_SQL, fid)
    q_id = await conn.fetchval(QUEUE_BY_ID_SQL, fid)
    if (paras or 0) or (q_src or 0) or (q_id or 0):
        log.info("  skip %s %s: paras=%s queue(source)=%s queue(id)=%s",
                 label, fid, paras, q_src, q_id)
        return False
    if survivor:
        await conn.execute(MOVE_MEMBERS_SQL, fid, survivor)
    await conn.execute(DROP_MEMBERS_SQL, fid)
    await conn.execute(DELETE_FM_SQL, fid)
    return True


async def apply(conn, p: dict) -> dict:
    done = {"renamed": 0, "dup_deleted": 0, "gone_deleted": 0, "skipped": 0}
    for item in p["rename"]:
        try:
            await conn.execute(RENAME_SQL, item["id"], item["to"])
            done["renamed"] += 1
        except Exception as ex:  # noqa: BLE001
            log.error("rename failed %s -> %s: %s", item["from"], item["to"], ex)
            done["skipped"] += 1
    for item in p["dup"]:
        try:
            if await guarded_delete(conn, item["id"], "dup",
                                     survivor=item["survivor"]):
                done["dup_deleted"] += 1
            else:
                done["skipped"] += 1
        except Exception as ex:  # noqa: BLE001
            log.error("dup delete failed %s: %s", item["path"], ex)
            done["skipped"] += 1
    for item in p["gone"]:
        try:
            if await guarded_delete(conn, item["id"], "gone"):
                done["gone_deleted"] += 1
            else:
                done["skipped"] += 1
        except Exception as ex:  # noqa: BLE001
            log.error("gone delete failed %s: %s", item["path"], ex)
            done["skipped"] += 1
    return done


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="batch_operation/reports/path_drift.json")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        p = await plan(conn)
        meta = p.pop("meta")
        summary = {"meta": meta, **{k: len(v) for k, v in p.items()}}
        log.info("plan: %s", json.dumps(summary, ensure_ascii=False))
        for item in p["rename"][:5]:
            log.info("  rename: %s -> %s", item["from"], item["to"])
        for item in p["dup"][:5]:
            log.info("  dup: %s (survivor %s)", item["path"], item["survivor"])
        for item in p["gone"][:5]:
            log.info("  gone: %s", item["path"])
        if not args.apply:
            log.info("dry-run only (pass --apply to write)")
            C.dump_json(args.report, {"summary": summary, "plan": p})
            return 0

        async with conn.transaction():
            done = await apply(conn, p)
        summary.update(done)

        # verify
        left = await conn.fetchval("""
            SELECT count(*) FROM file_metadata f
             WHERE f.path LIKE 'OneDrive/%' AND lower(f.path) LIKE '%.pdf'
               AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
        """)
        still_stale = 0
        for row in await conn.fetch(PENDING_SQL):
            if not os.path.exists(os.path.join(STORAGE, row["path"])):
                still_stale += 1
        summary["pending_after"] = left
        summary["stale_after"] = still_stale
        C.dump_json(args.report, summary)
        log.info("done: %s", json.dumps(summary, ensure_ascii=False))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
