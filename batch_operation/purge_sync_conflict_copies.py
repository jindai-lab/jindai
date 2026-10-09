"""删除同步冲突产生的“带时间戳”PDF 副本：记录 + 磁盘文件一并删除（2026-10-09 用户指令）。

识别：文件名里含 ISO 时间戳（`2026-08-23T231841.730082Z` 形态）或
`YYYY-MM-DD HHMMSS` / `YYYYMMDD_HHMMSS` 形态 → 判为冲突副本。

处置规则（按序）：
  1) 同题原件在**盘上**存在（去掉时间戳后文件名完全一致），且其 file_metadata 行
     也存在 → 删副本：队列行 → 嵌入 → 段落 → file_metadata 行 → 磁盘文件。
  2) 同题原件在盘上存在但**没有** file_metadata 行 → 不删记录，改把副本行
     `UPDATE path` 指向原件（文本保留，原件行下次导入自动登记），并删副本文件。
  3) 副本有段落而原件行 0 段落（删了会丢文本）→ 跳过并报告，等用户定夺。
  4) 找不到同题原件 → 保留（规则：只有同题文件存在才删）。

dry-run 默认；--apply 落库；--limit N 调试。幂等。
  docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/purge_sync_conflict_copies.py [--apply] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import sys
import time

import common as C

log = C.setup_logging("purge_conflict_copies",
                      "batch_operation/reports/purge_conflict_copies.log")

OUT = "batch_operation/reports/purge_conflict_copies.json"
TS_PATTERNS = [
    re.compile(r" ?- \d{4}-\d{2}-\d{2}T\d{6}(?:\.\d+)?Z$"),       # ISO8601 后缀
    re.compile(r" \(\d{4}-\d{2}-\d{2}T\d{6}(?:\.\d+)?Z\)$"),
    re.compile(r" - \d{4}-\d{2}-\d{2}[ _]\d{6}$"),
    re.compile(r"[ _]\d{8}[_-]\d{6}$"),
    re.compile(r" \(\d{4}-\d{2}-\d{2}[ _T]\d{2}[-:]?\d{2}[-:]?\d{2}\)$"),
]
EMB_BATCH = 4000


def strip_ts(stem: str) -> str | None:
    for pat in TS_PATTERNS:
        out = pat.sub("", stem)
        if out != stem:
            return out
    return None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from jindai.storage import storage
    base = storage.safe_join("OneDrive")[: -len("OneDrive")]

    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT id, path FROM file_metadata WHERE lower(path) LIKE '%.pdf'")
    all_paths = {r["path"] for r in rows}

    # 全盘同名索引（basename -> [path]）：优先复用 6 小时内的缓存清单，省一次全走
    disk: dict[str, list[str]] = collections.defaultdict(list)
    cache = "batch_operation/reports/pdf_disk_paths.txt"   # bind-mounted，容器/宿主同路径
    used_cache = False
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 3600:
        with open(cache, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.lower().endswith(".pdf"):
                    disk[line.rsplit("/", 1)[-1]].append(line)
        used_cache = True
    if not disk:
        for dirpath, d, fns in os.walk(base):
            d[:] = [x for x in d if not x.startswith(".")]  # 跳过回收站等隐藏目录
            for fn in fns:
                if fn.lower().endswith(".pdf"):
                    rel = os.path.relpath(os.path.join(dirpath, fn), base)
                    disk[fn].append(rel.replace(os.sep, "/"))
    log.info("[walk] %d pdfs on disk (cache=%s)", sum(len(v) for v in disk.values()),
             used_cache)

    st: collections.Counter = collections.Counter()
    report: list[dict] = []

    cands = []
    for r in rows:
        path = r["path"]
        stem = path.rsplit("/", 1)[-1][:-4]
        base_stem = strip_ts(stem)
        if base_stem is None:
            continue
        cands.append((str(r["id"]), path, base_stem + path[-4:]))
    if args.limit:
        cands = cands[:args.limit]
    log.info("[cands] %d timestamp-named pdfs", len(cands))

    for fid, path, orig_name in cands:
        same_dir = os.path.join(os.path.dirname(os.path.join(base, path)), orig_name)
        hits = [p for p in disk.get(orig_name, []) if p != path]
        ent = {"copy": path, "orig_name": orig_name,
               "orig_on_disk": bool(hits) or os.path.exists(same_dir),
               "orig_paths": hits[:3]}
        if not ent["orig_on_disk"]:
            st["no_original_kept"] += 1
            ent["action"] = "kept_no_original"
            report.append(ent)
            continue
        orig_row = hits[0] if hits else path.rsplit("/", 1)[0] + "/" + orig_name
        orig_id = await conn.fetchval(
            "SELECT id FROM file_metadata WHERE path=$1", orig_row)
        n_copy = await conn.fetchval(
            "SELECT count(*) FROM paragraph WHERE source_id=$1::uuid", fid)
        n_orig = await conn.fetchval(
            "SELECT count(*) FROM paragraph WHERE source_id=$1::uuid", orig_id) \
            if orig_id else 0
        ent.update(orig_row=orig_row, copy_paras=int(n_copy or 0),
                   orig_paras=int(n_orig or 0))

        if orig_id is None:
            # 原件在盘上但没记录 → 副本行改指原件，只删副本文件
            if args.apply:
                async with conn.transaction():
                    tag = await conn.execute(
                        "UPDATE file_metadata SET path=$1 WHERE id=$2::uuid "
                        "AND NOT EXISTS (SELECT 1 FROM file_metadata WHERE path=$1)",
                        orig_row, fid)
                    if int(tag.split()[-1]):
                        try:
                            os.remove(os.path.join(base, path))
                            st["file_removed"] += 1
                        except OSError as ex:
                            log.warning("rm failed %s: %s", path[:70], str(ex)[:120])
                        st["repointed_to_original"] += 1
                if os.path.exists(os.path.join(base, path)):
                    st["repoint_failed_still_exists"] += 1
                    ent["action"] = "repoint_failed"
                else:
                    ent["action"] = "repointed"
            else:
                st["repointed_to_original"] += 1
                ent["action"] = "repointed"
            report.append(ent)
            continue

        if n_copy and not n_orig:
            st["skipped_orig_empty"] += 1
            ent["action"] = "skipped_original_has_no_text"
            report.append(ent)
            continue

        # 删除副本：记录 + 文件
        if args.apply:
            try:
                async with conn.transaction():
                    await conn.execute(
                        "DELETE FROM embedding_pending_queue WHERE source_id=$1::uuid", fid)
                    ids = [x["id"] for x in await conn.fetch(
                        "SELECT id FROM paragraph WHERE source_id=$1::uuid", fid)]
                    for j in range(0, len(ids), EMB_BATCH):
                        await conn.execute(
                            "DELETE FROM text_embeddings WHERE source_id=$1::uuid "
                            "AND id=ANY($2::uuid[])", fid, ids[j:j + EMB_BATCH])
                    await conn.execute(
                        "DELETE FROM paragraph WHERE source_id=$1::uuid", fid)
                    await conn.execute("DELETE FROM file_dataset WHERE file_id=$1::uuid", fid)
                    tag = await conn.execute(
                        "DELETE FROM file_metadata WHERE id=$1::uuid "
                        "AND NOT EXISTS (SELECT 1 FROM paragraph WHERE source_id=$1)", fid)
                    if int(tag.split()[-1]):
                        st["rows_deleted"] += 1
                        st["paras_deleted"] += len(ids)
                    else:
                        st["row_delete_guarded"] += 1
                if os.path.exists(os.path.join(base, path)):
                    os.remove(os.path.join(base, path))
                    st["file_removed"] += 1
                ent["action"] = "deleted"
            except Exception as ex:
                st["failed"] += 1
                ent["action"] = "failed"
                log.warning("[del] %s: %s", path[:70], str(ex)[:200])
        else:
            st["to_delete"] += 1
            ent["action"] = "to_delete"
        report.append(ent)

    # 盘上还有“没登记过记录”的带时间戳副本 → 只处理文件
    handled = {path for _, path, _ in cands}
    for fn, paths_ in disk.items():
        for p in paths_:
            if p in handled:
                continue
            stem = p.rsplit("/", 1)[-1][:-4]
            bs = strip_ts(stem)
            if bs is None:
                continue
            orig = bs + p[-4:]
            hits = [q for q in disk.get(orig, []) if q != p]
            ent = {"copy": p, "orig_name": orig, "orig_paths": hits[:3],
                   "orig_on_disk": bool(hits)}
            if not hits:
                st["no_original_kept"] += 1
                ent["action"] = "kept_no_original"
                report.append(ent)
                continue
            if args.apply:
                try:
                    os.remove(os.path.join(base, p))
                    st["file_removed_untracked"] += 1
                    ent["action"] = "file_removed_untracked"
                except OSError as ex:
                    st["failed"] += 1
                    ent["action"] = "rm_failed"
                    log.warning("[rm] %s: %s", p[:70], str(ex)[:150])
            else:
                st["file_to_remove_untracked"] += 1
                ent["action"] = "file_to_remove_untracked"
            report.append(ent)

    summary = {"mode": "apply" if args.apply else "dry-run",
               "counters": dict(st), "items": report}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log.info("[done] mode=%s counters=%s", summary["mode"],
             json.dumps(dict(st), ensure_ascii=False))
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
