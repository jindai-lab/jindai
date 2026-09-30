"""Backfill 书库--{author} dataset links for OneDrive Calibre-layout files.

Why: DatasetFromField rendered paragraph.author=None as the literal bucket
"书库--None" (~1.6k files landed there). Correct author buckets follow the
Calibre folder layout:  <catalog>/<author>/<title (id)>/<file>.pdf

Does (idempotent, single transaction, safe to run while imports are active):
  1. normalize legacy '../app/' path prefixes
  2. ensure dataset 书库--<author> exists (dataset.name has a unique index)
  3. link every matchable file to its author bucket (ON CONFLICT DO NOTHING)
  4. unlink 书库--None and drop the emptied dataset row

  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/backfill_author_datasets.py            # dry run
  ... same command + --apply                                      # write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter

import common as C

log = C.setup_logging(
    "backfill_author_datasets", "batch_operation/reports/backfill_author_datasets.log"
)
REPORT = "batch_operation/reports/backfill_author_datasets.json"

FILES_SQL = """
    SELECT id, path FROM file_metadata
    WHERE path LIKE '%OneDrive/%' AND lower(path) LIKE '%.pdf'
"""
DATASETS_SQL = "SELECT id, name FROM dataset WHERE name LIKE '书库--%'"

_CLEAN_RE = re.compile(r"^(?:\.\./)+(?:app/)?")


def derive_author(path: str):
    """Calibre layout author (folder one level under the catalog root)."""
    p = _CLEAN_RE.sub("", path)
    parts = p.split("/")
    if len(parts) == 4 and parts[0] == "OneDrive" and p.lower().endswith(".pdf") and parts[1]:
        return parts[1]
    return None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()
    t0 = time.time()
    conn = await C.get_conn()
    try:
        files = await conn.fetch(FILES_SQL)
        buckets: dict[str, list] = {}
        skipped = Counter()
        skipped_samples: list[str] = []
        dirty_prefix = 0
        for row in files:
            raw = row["path"]
            if raw.startswith("../"):
                dirty_prefix += 1
            author = derive_author(raw)
            if author is None:
                reason = ("legacy_prefix_normalized" if raw.startswith("../")
                          else f"depth_{len(raw.split('/'))}")
                skipped[reason] += 1
                if len(skipped_samples) < 5:
                    skipped_samples.append(raw)
                continue
            buckets.setdefault(author, []).append(row["id"])

        existing = {r["name"]: r["id"] for r in await conn.fetch(DATASETS_SQL)}
        new_authors = sorted(a for a in buckets if f"书库--{a}" not in existing)
        none_id = existing.get("书库--None")
        none_links = await conn.fetchval(
            "SELECT count(*) FROM file_dataset WHERE dataset_id = $1", none_id
        ) if none_id else 0

        report = {
            "mode": "apply" if args.apply else "dry-run",
            "files_total": len(files),
            "matchable_files": sum(len(v) for v in buckets.values()),
            "authors": len(buckets),
            "datasets_to_create": len(new_authors),
            "links_to_add": sum(len(v) for v in buckets.values()),
            "none_bucket": {"id": str(none_id) if none_id else None,
                            "links_to_remove": none_links},
            "legacy_prefix_paths": dirty_prefix,
            "skipped": dict(skipped),
            "skipped_samples": skipped_samples,
        }

        if args.apply:
            async with conn.transaction():
                for author in new_authors:
                    await conn.execute(
                        "INSERT INTO dataset (name) VALUES ($1) "
                        "ON CONFLICT (name) DO NOTHING",
                        f"书库--{author}",
                    )
                ds = {r["name"]: r["id"] for r in await conn.fetch(DATASETS_SQL)}
                pairs = [(ds[f"书库--{a}"], fid)
                         for a, fids in buckets.items() for fid in fids]
                added = 0
                for i in range(0, len(pairs), 500):
                    chunk = pairs[i:i + 500]
                    ph = ",".join(
                        f"(${2 * j + 1}, ${2 * j + 2})" for j in range(len(chunk))
                    )
                    res = await conn.execute(
                        f"INSERT INTO file_dataset (dataset_id, file_id) "
                        f"VALUES {ph} ON CONFLICT DO NOTHING",
                        *[v for pair in chunk for v in pair],
                    )
                    added += int(res.rsplit(" ", 1)[-1])
                unlinked = 0
                meta_refs = 0
                dataset_dropped = False
                if none_id:
                    res = await conn.execute(
                        "DELETE FROM file_dataset WHERE dataset_id = $1", none_id
                    )
                    unlinked = int(res.rsplit(" ", 1)[-1])
                    meta_refs = await conn.fetchval(
                        "SELECT count(*) FROM file_metadata WHERE dataset_id = $1",
                        none_id,
                    )
                    if meta_refs:
                        await conn.execute(
                            "UPDATE file_metadata SET dataset_id = NULL "
                            "WHERE dataset_id = $1", none_id
                        )
                    await conn.execute("DELETE FROM dataset WHERE id = $1", none_id)
                    dataset_dropped = True
                report.update(
                    datasets_created=len(new_authors),
                    links_added=added,
                    none_links_removed=unlinked,
                    none_dataset_dropped=dataset_dropped,
                    file_metadata_refs_cleared=int(meta_refs or 0),
                )
        report["seconds"] = round(time.time() - t0, 1)
        with open(REPORT, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        log.info("done %s", json.dumps(report, ensure_ascii=False))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
