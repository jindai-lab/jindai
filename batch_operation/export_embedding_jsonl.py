"""Export paragraphs still missing embeddings as OpenAI-style batch JSONL.

Each output line is one embedding request, byte-compatible with the backend's
own chunking (common.get_chunks -> TextEmbeddings.get_chunks(200/50)):

  {"custom_id": "<paragraph_uuid>:<chunk_id>", "method": "POST",
   "url": "/v1/embeddings",
   "body": {"model": "bge-m3", "input": "<chunk text>"}}

custom_id encodes (paragraph id, chunk_id) so a batch response can be joined
back to text_embeddings (id, source_id, chunk_id) on consume.

Scope = the authoritative anti-join (paragraph LEFT-ANTI text_embeddings),
same population embed_missing.py drains. Files largest-first, keyset-paged
per file, resumable via a state file (done file ids + cursor).

Output is split into files of at most --max-bytes (default 500 MB); a new
file is opened before a line that would overflow it. A manifest json lists
every produced file with line/byte counts and custom_id ranges.

Run (in container, cwd /app):
  python batch_operation/export_embedding_jsonl.py --dry-run
  python batch_operation/export_embedding_jsonl.py --limit-files 3
  python batch_operation/export_embedding_jsonl.py --scope onedrive \
      --max-bytes 500MB --out /tmp/embed_jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as C

log = C.setup_logging("export_embedding_jsonl",
                      "batch_operation/reports/export_embedding_jsonl.log")

MISSING_BY_FILE_SQL = """
    SELECT p.source_id, count(*)::bigint AS n
    FROM paragraph p
    WHERE NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
    GROUP BY 1
"""
PATHS_SQL = "SELECT id, path FROM file_metadata WHERE path LIKE $1"
PAGE_SQL = """
    SELECT p.id, p.source_id, p.content
    FROM paragraph p
    WHERE p.source_id = $1
      AND ($2::uuid IS NULL OR p.id > $2)
      AND NOT EXISTS (SELECT 1 FROM text_embeddings e WHERE e.id = p.id)
    ORDER BY p.id
    LIMIT $3
"""

PAGE_SIZE = 5000          # paragraphs per keyset round-trip
CHUNKS_PER_PARA_EST = 1.9
DEFAULT_MAX_BYTES = 500 * 1024 * 1024


def parse_bytes(s: str) -> int:
    """'500MB' / '2GB' / plain digits -> int bytes."""
    s = s.strip().upper()
    mult = 1
    for suffix, m in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
                      ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if s.endswith(suffix):
            mult = m
            s = s[: -len(suffix)]
            break
    return int(float(s) * mult)


class JsonlWriter:
    """Rolling writer: at most max_bytes per file, one request per line."""

    def __init__(self, outdir: str, prefix: str, max_bytes: int,
                 model: str, extra_body: dict):
        self.outdir = outdir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.model = model
        self.extra_body = extra_body
        self.index = 0
        self.fh = None
        self.bytes_written = 0
        self.lines_written = 0
        self.manifest = []
        os.makedirs(outdir, exist_ok=True)

    def _open_next(self):
        if self.fh:
            self.fh.close()
            log.info("  closed %s (%d lines, %.1f MB)",
                     self.manifest[-1]["file"], self.manifest[-1]["lines"],
                     self.manifest[-1]["bytes"] / 2**20)
        path = os.path.join(self.outdir, f"{self.prefix}_{self.index:05d}.jsonl")
        self.fh = open(path, "w", encoding="utf-8")
        self.index += 1
        self.bytes_written = 0
        self.lines_written = 0
        self.manifest.append({
            "file": os.path.basename(path), "path": path,
            "lines": 0, "bytes": 0,
            "first_custom_id": None, "last_custom_id": None,
        })

    def write(self, custom_id: str, text: str) -> None:
        body = {"model": self.model, "input": text}
        if self.extra_body:
            body.update(self.extra_body)
        line = json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": body,
        }, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        if self.fh is None or (
            self.bytes_written > 0
            and self.bytes_written + len(encoded) > self.max_bytes
        ):
            self._open_next()
        assert self.fh is not None  # _open_next guarantees an open handle
        self.fh.write(line)
        self.bytes_written += len(encoded)
        self.lines_written += 1
        m = self.manifest[-1]
        m["lines"] += 1
        m["bytes"] += len(encoded)
        if m["first_custom_id"] is None:
            m["first_custom_id"] = custom_id
        m["last_custom_id"] = custom_id

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"done_file_ids": []}


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", choices=["all", "onedrive"], default="all")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--max-bytes", default="500MB",
                    help="per-file size cap (e.g. 500MB, 2GB)")
    ap.add_argument("--out", default="batch_operation/reports/embedding_jsonl")
    ap.add_argument("--prefix", default="embed_req")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="stop after N files this run (0 = all)")
    ap.add_argument("--limit-paragraphs", type=int, default=0,
                    help="hard cap on paragraphs this run (0 = all)")
    ap.add_argument("--limit-chunks", type=int, default=0,
                    help="hard cap on emitted request lines (0 = all)")
    ap.add_argument("--max-files", type=int, default=0,
                    help="stop once N output files have been closed (0 = all)")
    ap.add_argument("--extra-body", default="",
                    help="JSON object merged into every request body")
    ap.add_argument("--resume", action="store_true",
                    help="skip file ids recorded in the state file")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest",
                    default="batch_operation/reports/embedding_jsonl_manifest.json")
    ap.add_argument("--state",
                    default="batch_operation/reports/embedding_jsonl_state.json")
    args = ap.parse_args()

    max_bytes = parse_bytes(args.max_bytes)
    extra_body = json.loads(args.extra_body) if args.extra_body else {}

    t0 = time.time()
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(MISSING_BY_FILE_SQL)
        missing = {r["source_id"]: r["n"] for r in rows}
        total_missing = sum(missing.values())
        log.info("paragraphs missing embeddings: %s across %d files",
                 f"{total_missing:,}", len(missing))

        if args.scope == "onedrive":
            path_rows = await conn.fetch(PATHS_SQL, C.ONEDRIVE_LIKE)
        else:
            path_rows = await conn.fetch("SELECT id, path FROM file_metadata")
        paths = {r["id"]: r["path"] for r in path_rows}
        files = [(fid, paths[fid], n) for fid, n in missing.items() if fid in paths]
        orphans = total_missing - sum(n for _, _, n in files)
        if orphans:
            log.info(
                "%s missing paragraphs belong to files OUTSIDE this scope filter "
                "(embeddable, just not selected; not orphan rows)",
                f"{orphans:,}",
            )
        files.sort(key=lambda x: -x[2])
        est_chunks = int(sum(n for _, _, n in files) * CHUNKS_PER_PARA_EST)
        log.info("scope=%s files=%d est_chunks=%s max_bytes=%s",
                 args.scope, len(files), f"{est_chunks:,}", f"{max_bytes:,}")

        state = load_state(args.state) if args.resume else {"done_file_ids": []}
        done = set(state["done_file_ids"])
        if done:
            before = len(files)
            files = [(f, p, n) for f, p, n in files if str(f) not in done]
            log.info("resume: %d files already done, %d remain",
                     before - len(files), len(files))

        if args.dry_run:
            for fid, path, n in files[:20]:
                log.info("  %8d  %s", n, path)
            return 0

        writer = JsonlWriter(args.out, args.prefix, max_bytes,
                             args.model, extra_body)
        para_done = chunk_done = 0
        files_done = 0
        limit_files = args.limit_files or len(files)
        try:
            for fid, path, _est in files:
                if files_done >= limit_files or (
                    args.limit_chunks and chunk_done >= args.limit_chunks
                ):
                    break
                last_id = None
                file_paras = file_chunks = 0
                try:
                    while True:
                        rows = await conn.fetch(PAGE_SQL, fid, last_id, PAGE_SIZE)
                        if not rows:
                            break
                        last_id = rows[-1]["id"]
                        page_paras = page_chunks = 0
                        for r in rows:
                            if args.max_files and writer.index > args.max_files:
                                break
                            chunks = C.get_chunks(r["content"])
                            if not chunks:
                                continue
                            file_paras += 1
                            page_paras += 1
                            for ci, chunk in enumerate(chunks, 1):
                                writer.write(f"{r['id']}:{ci}", chunk)
                                file_chunks += 1
                                page_chunks += 1
                            if args.limit_chunks and chunk_done + file_chunks \
                                    >= args.limit_chunks:
                                break
                        para_done += page_paras
                        chunk_done += page_chunks
                        if args.limit_paragraphs and para_done \
                                >= args.limit_paragraphs:
                            break
                        if args.limit_chunks and chunk_done >= args.limit_chunks:
                            break
                        if args.max_files and writer.index > args.max_files:
                            # writer.index counts OPENED files; index > N means
                            # file N rolled over to N+1, i.e. N files are full
                            break
                except Exception as ex:  # noqa: BLE001 - keep going per file
                    log.error("file failed: %s (%s: %s)", path,
                              type(ex).__name__, str(ex)[:300])
                    continue
                if args.max_files and writer.index > args.max_files:
                    log.info("reached --max-files %d, stopping", args.max_files)
                    break
                files_done += 1
                state["done_file_ids"].append(str(fid))
                if args.resume:
                    save_state(args.state, state)
                if files_done % 10 == 0 or files_done == len(files):
                    log.info("files %d | paragraphs %s | chunks %s | %.0f chunks/s",
                             files_done, f"{para_done:,}", f"{chunk_done:,}",
                             chunk_done / max(time.time() - t0, 1))
        finally:
            writer.close()
            if args.resume:
                save_state(args.state, state)

        manifest = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "scope": args.scope,
            "chunking": {"chunk_length": C.CHUNK_LEN, "overlap": C.CHUNK_OVERLAP},
            "extra_body": extra_body,
            "max_bytes": max_bytes,
            "files": writer.manifest,
            "totals": {
                "files": len(writer.manifest),
                "paragraphs": para_done,
                "requests": chunk_done,
                "bytes": sum(m["bytes"] for m in writer.manifest),
            },
        }
        os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        log.info("done: %d file(s), %s paragraphs -> %s requests, %.1f MB in %.0fs",
                 len(writer.manifest), f"{para_done:,}", f"{chunk_done:,}",
                 manifest["totals"]["bytes"] / 2**20, time.time() - t0)
        log.info("manifest: %s", args.manifest)
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main()))
