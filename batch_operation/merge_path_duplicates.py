"""Merge '../app/...' duplicate file_metadata rows into their clean twins.

Background: config.yaml used to say ``storage: /storage`` while the files
live under ``/app/OneDrive``, so every import stored paths as
``relpath('/app/...', '/storage')`` = ``../app/OneDrive/...``. The 2026-09
path migration (and other scripts) had inserted the *clean* path
``OneDrive/...`` separately, giving one physical file two rows: the clean
row (counted as "pending", 0 paragraphs) and the legacy row (holding the
paragraphs). FileMetadata.get matches paths EXACTLY, so the pipeline kept
writing to the legacy rows and the pending counter never fell.

Run AFTER config storage has been switched back to ``/app`` AND after the
in-flight import task finished (otherwise the task keeps minting legacy rows).

Case handling (c = clean row 'OneDrive/...', l = legacy row '../app/OneDrive/...'):
  a) c empty, l has paragraphs  -> rename l.path := c.path, delete empty c
  b) c has, l empty             -> delete empty l (c already authoritative)
  c) both have                  -> leave both, report (real duplicates)
  d) both empty                 -> delete l (clean c stays for the pipeline)
  e) l exists, no c             -> rename l.path (strip '../app/')
file_dataset memberships are moved onto the surviving row before deletes.

  python batch_operation/merge_path_duplicates.py --dry-run
  python batch_operation/merge_path_duplicates.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import common as C

log = C.setup_logging("merge_paths", "batch_operation/reports/merge_paths.log")

PAIRS_SQL = """
    SELECT c.id AS cid, l.id AS lid, c.path AS cpath
    FROM file_metadata c
    JOIN file_metadata l ON l.path = '../app/' || c.path
    WHERE c.path LIKE 'OneDrive/%' AND lower(c.path) LIKE '%.pdf'
"""
LEGACY_ONLY_SQL = """
    SELECT l.id, l.path FROM file_metadata l
    WHERE l.path LIKE '../app/%'
      AND NOT EXISTS (SELECT 1 FROM file_metadata c
                      WHERE c.path = substr(l.path, 8))
"""
PARA_COUNT_SQL = "SELECT count(*) FROM paragraph WHERE source_id = $1"
MOVE_MEMBERS_SQL = """
    INSERT INTO file_dataset (file_id, dataset_id)
    SELECT $2, dataset_id FROM file_dataset WHERE file_id = $1
    ON CONFLICT DO NOTHING
"""
DROP_MEMBERS_SQL = "DELETE FROM file_dataset WHERE file_id = $1"
DELETE_FM_SQL = "DELETE FROM file_metadata WHERE id = $1"
RENAME_SQL = "UPDATE file_metadata SET path = $2 WHERE id = $1"


async def _has_paras(conn, fid) -> bool:
    return (await conn.fetchval(PARA_COUNT_SQL, fid)) > 0


async def plan(conn) -> dict:
    out = {
        "rename_legacy_over_clean": [],   # case a
        "drop_empty_legacy": [],          # cases b, d
        "both_have_report": [],           # case c
        "rename_legacy_only": [],         # case e
    }
    for row in await conn.fetch(PAIRS_SQL):
        cid, lid, cpath = row["cid"], row["lid"], row["cpath"]
        c_has = await _has_paras(conn, cid)
        l_has = await _has_paras(conn, lid)
        if not c_has and l_has:
            out["rename_legacy_over_clean"].append(
                {"legacy": str(lid), "clean": str(cid), "path": cpath})
        elif c_has and not l_has:
            out["drop_empty_legacy"].append(
                {"legacy": str(lid), "clean": str(cid), "path": cpath})
        elif c_has and l_has:
            out["both_have_report"].append(
                {"legacy": str(lid), "clean": str(cid), "path": cpath})
        else:  # both empty
            out["drop_empty_legacy"].append(
                {"legacy": str(lid), "clean": str(cid), "path": cpath})
    for row in await conn.fetch(LEGACY_ONLY_SQL):
        clean = row["path"][len("../app/"):]  # strip '../app/'
        if not clean.startswith("OneDrive/"):
            log.warning("legacy path outside OneDrive, skipped: %s", row["path"])
            continue
        out["rename_legacy_only"].append(
            {"legacy": str(row["id"]), "path": clean})
    return out


async def apply(conn, p: dict) -> dict:
    done = {"a_renamed": 0, "b_d_deleted": 0, "e_renamed": 0}
    # case a: legacy row takes the clean path; empty clean row is removed.
    # MOVE_MEMBERS_SQL moves $1's dataset links onto $2 -> pass (clean, legacy)
    # so the surviving legacy row inherits the clean row's memberships.
    for item in p["rename_legacy_over_clean"]:
        lid, cid = item["legacy"], item["clean"]
        # order matters: uniq_path forbids two rows holding the clean path,
        # so delete the empty clean row BEFORE renaming legacy (first attempt
        # crashed with UniqueViolationError renaming while cid still existed)
        try:
            await conn.execute(MOVE_MEMBERS_SQL, cid, lid)
            await conn.execute(DROP_MEMBERS_SQL, cid)
            await conn.execute(DELETE_FM_SQL, cid)
            await conn.execute(RENAME_SQL, lid, item["path"])
            done["a_renamed"] += 1
        except Exception as ex:  # noqa: BLE001 - keep going per item
            log.error("case-a failed for %s: %s", item["path"], ex)
    # cases b/d: legacy empty -> just remove it (clean row keeps its data)
    for item in p["drop_empty_legacy"]:
        try:
            await conn.execute(DROP_MEMBERS_SQL, item["legacy"])
            await conn.execute(DELETE_FM_SQL, item["legacy"])
            done["b_d_deleted"] += 1
        except Exception as ex:  # noqa: BLE001
            log.error("case-b/d failed for %s: %s", item["path"], ex)
    # case e: legacy without clean twin -> normalize its path
    for item in p["rename_legacy_only"]:
        try:
            await conn.execute(RENAME_SQL, item["legacy"], item["path"])
            done["e_renamed"] += 1
        except Exception as ex:  # noqa: BLE001
            log.error("case-e failed for %s: %s", item["path"], ex)
    return done


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only (default when --apply is absent)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="batch_operation/reports/merge_paths.json")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        p = await plan(conn)
        summary = {k: len(v) for k, v in p.items()}
        log.info("plan: %s", json.dumps(summary))
        for item in p["both_have_report"][:10]:
            log.info("  both-have (left alone): %s", item["path"])
        if not args.apply:
            for k, v in p.items():
                for item in v[:5]:
                    log.info("  %s: %s", k, item.get("path"))
            return 0
        done = await apply(conn, p)
        # verify: no '../app/' rows left under OneDrive, pending recomputed
        legacy = await conn.fetchval(
            "SELECT count(*) FROM file_metadata WHERE path LIKE '../app/%'")
        pending = await conn.fetchval(
            """SELECT count(*) FROM file_metadata f
               WHERE f.path LIKE 'OneDrive/%' AND lower(f.path) LIKE '%.pdf'
                 AND NOT EXISTS (SELECT 1 FROM paragraph p
                                 WHERE p.source_id = f.id)""")
        summary.update(done)
        summary["legacy_remaining"] = legacy
        summary["pending_pdf_after"] = pending
        C.dump_json(args.report, summary)
        log.info("done: %s", json.dumps(summary))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
