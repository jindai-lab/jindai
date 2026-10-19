#!/usr/bin/env python3
"""Detect which of the OneDrive PDFs lacking paragraphs have a native text
layer (importable without OCR) vs. pure scans (need OCR).

Usage: python detect_textlayer.py [--sample N] [--out FILE]
Writes a JSON with two lists: with_text (paths), scans (paths).
"""
import argparse, asyncio, json, os, sys

sys.path.insert(0, "/app")
import fitz  # PyMuPDF

import common as C

ONEDRIVE = "/storage"

MISSING_SQL = """
    SELECT f.id, f.path FROM file_metadata f
    WHERE f.path LIKE 'OneDrive/%' AND lower(f.path) LIKE '%.pdf'
      AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
    ORDER BY f.path
"""


def has_text_layer(path: str, sample: int = 10) -> bool:
    """Sample pages across the doc; >=20% of sampled pages with >=30 chars -> text layer."""
    try:
        doc = fitz.open(path)
    except Exception:
        return False
    try:
        n = doc.page_count
        if n == 0:
            return False
        idxs = sorted(set(
            [0, n // 2, n - 1] +
            [int(n * i / sample) for i in range(sample)]
        ))[: sample + 3]
        good = 0
        for i in idxs:
            try:
                t = doc[i].get_text().strip()
            except Exception:
                t = ""
            if len(t) >= 30:
                good += 1
        return good >= max(1, int(len(idxs) * 0.2))
    finally:
        doc.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--out", default="/app/batch_operation/reports/textlayer_split.json")
    ap.add_argument("--limit", type=int, default=0, help="only check first N files")
    args = ap.parse_args()

    conn = await C.get_conn()
    rows = await conn.fetch(MISSING_SQL)
    await conn.close()
    if args.limit:
        rows = rows[: args.limit]

    with_text, scans, errors = [], [], []
    for i, row in enumerate(rows):
        rel = row["path"]
        full = os.path.join(ONEDRIVE, rel)
        if not os.path.exists(full):
            errors.append(rel)
            continue
        try:
            ok = has_text_layer(full, args.sample)
        except Exception:
            ok = False
        (with_text if ok else scans).append(rel)
        if (i + 1) % 200 == 0:
            print(f"progress {i+1}/{len(rows)}  with_text={len(with_text)} scans={len(scans)}", flush=True)

    result = {
        "with_text": with_text, "scans": scans, "missing": errors,
        "counts": {"with_text": len(with_text), "scans": len(scans), "missing": len(errors)},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("DONE", result["counts"], "->", args.out)


if __name__ == "__main__":
    asyncio.run(main())

