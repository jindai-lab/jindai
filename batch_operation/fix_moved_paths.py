"""诊断 file_metadata 里“盘上找不到”的路径是否只是被移动（文件名多半没改）。

2026-10-08 用户指令：8,557 条磁盘缺失行（多为历史移动）——先检查、再修复。

思路：
  1. 读上一轮清单落盘的 reports/filemeta_missing_on_disk.txt。
  2. 全库（/storage 下所有顶层目录）走一遍 *.pdf，建 basename → [(relpath, size)] 索引。
  3. 每条缺失路径按文件名精确匹配：
       - 唯一 → safe（新路径若已有 file_metadata 行：目标行无段落则删空行再改，
                有段落则记 dup 只报告不自动改）
       - 多个 → 先按“路径里 (id) 相同”收敛，再按 size 相同收敛；仍多个 → multi 报告
       - 没有同名 → 用 size+ext 唯一匹配找“改名候选”（只报告，不自动改）
  4. dry-run 默认：产出 reports/filemeta_moved_analysis.tsv + 汇总 JSON；
     --apply 才落库（只改 safe / conflict_empty 两类）。

用法：
  python batch_operation/fix_moved_paths.py            # 只分析
  python batch_operation/fix_moved_paths.py --apply    # 应用安全修复
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import sys

import common as C

log = C.setup_logging("fix_moved_paths", "batch_operation/reports/fix_moved_paths.log")

ID_RE = re.compile(r"\((\d+)\)")


def id_token(p: str) -> str | None:
    m = ID_RE.search(p)
    return m.group(1) if m else None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from jindai.storage import storage

    missing = [l.strip() for l in open(
        "batch_operation/reports/filemeta_missing_on_disk.txt", encoding="utf-8")
        if l.strip()]
    log.info("[missing] %d rows", len(missing))

    od = storage.safe_join("OneDrive")
    base = od[: -len("OneDrive")]
    log.info("[walk] scanning %s for *.pdf ...", base)

    by_name: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    by_size: dict[int, list[str]] = collections.defaultdict(list)
    n_files = 0
    for dirpath, _dns, fns in os.walk(base):
        for fn in fns:
            if not fn.lower().endswith(".pdf"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, base).replace(os.sep, "/")
            try:
                sz = os.path.getsize(full)
            except OSError:
                sz = -1
            by_name[fn].append((rel, sz))
            by_size[sz].append(rel)
            n_files += 1
    log.info("[walk] %d pdf files indexed (%d distinct names)", n_files, len(by_name))

    conn = await C.get_conn()
    rows = await conn.fetch("SELECT path FROM file_metadata")
    has_row = {r["path"] for r in rows}
    # 目标行是否有段落（conflict 用）
    para_paths = {r["path"] for r in await conn.fetch(
        "SELECT DISTINCT f.path FROM file_metadata f "
        "JOIN paragraph p ON p.source_id = f.id")}

    classes: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for old in missing:
        fn = old.rsplit("/", 1)[-1]
        oldsz = None
        cands = by_name.get(fn, [])
        chosen = None
        cls = None
        if len(cands) == 1:
            chosen, cls = cands[0][0], "safe"
        elif len(cands) > 1:
            tok = id_token(old)
            sub = [c for c in cands if tok and id_token(c[0]) == tok]
            if len(sub) == 1:
                chosen, cls = sub[0][0], "safe_id"
            elif len(sub) > 1:
                cls = "multi_id"
            else:
                cls = "multi"
        else:
            # 无同名 → size 唯一匹配 = 改名候选（只报告）
            # 需要旧文件 size：老路径不存在，无法取 size；改从 old 行的 size_bytes 拿
            cls = "no_name_match"

        if chosen is None:
            classes[cls].append((old, ""))
            continue
        if chosen in has_row:
            if chosen in para_paths:
                classes["dup_conflict"].append((old, chosen))
            else:
                classes["conflict_empty"].append((old, chosen))
        else:
            classes[cls if cls.startswith("safe") else "safe"].append((old, chosen))

    # size 兜底（只报告）：对 no_name_match，用 file_metadata.size_bytes 找唯一同 size 的盘上文件
    if classes["no_name_match"]:
        sizes = {r["path"]: r["size_bytes"] for r in await conn.fetch(
            "SELECT path, size_bytes FROM file_metadata")
            if r["path"] in set(classes["no_name_match"][i][0]
                                for i in range(len(classes["no_name_match"])))}
        rest = []
        for old, _ in classes["no_name_match"]:
            sz = sizes.get(old)
            cands = [p for p in by_size.get(sz, [])] if sz else []
            if len(cands) == 1:
                classes["renamed_size1"].append((old, cands[0]))
            elif len(cands) > 1:
                classes["renamed_multi"].append((old, cands[0]))
            else:
                rest.append((old, ""))
        classes["truly_gone"] = rest
        del classes["no_name_match"]

    summary = {k: len(v) for k, v in sorted(classes.items())}
    summary["missing_total"] = len(missing)
    log.info("[classify] %s", json.dumps(summary, ensure_ascii=False))

    os.makedirs("batch_operation/reports", exist_ok=True)
    with open("batch_operation/reports/filemeta_moved_analysis.tsv", "w",
              encoding="utf-8") as f:
        f.write("class\told_path\tnew_path\n")
        for cls, pairs in sorted(classes.items()):
            for old, new in pairs:
                f.write(f"{cls}\t{old}\t{new}\n")

    applied = {"updated": 0, "deleted_empty_target": 0, "failed": 0}
    if args.apply:
        todo = classes["safe"] + classes["safe_id"]
        empty = classes["conflict_empty"]
        for old, new in todo:
            try:
                await conn.execute(
                    "UPDATE file_metadata SET path = $1 WHERE path = $2", new, old)
                applied["updated"] += 1
            except Exception as ex:
                applied["failed"] += 1
                log.warning("[apply] update failed %s: %s", old[:80], str(ex)[:200])
        for old, new in empty:
            try:
                async with conn.transaction():
                    tid = await conn.fetchval(
                        "SELECT id FROM file_metadata WHERE path = $1", new)
                    await conn.execute(
                        "DELETE FROM embedding_pending_queue WHERE source_id = $1", tid)
                    await conn.execute(
                        "DELETE FROM file_metadata WHERE id = $1 "
                        "AND NOT EXISTS (SELECT 1 FROM paragraph "
                        "                 WHERE source_id = $1)", tid)
                    await conn.execute(
                        "UPDATE file_metadata SET path = $1 WHERE path = $2",
                        new, old)
                applied["deleted_empty_target"] += 1
            except Exception as ex:
                applied["failed"] += 1
                log.warning("[apply] conflict_empty failed %s: %s", old[:80],
                            str(ex)[:200])
        log.info("[apply] %s", json.dumps(applied))

    with open("batch_operation/reports/fix_moved_paths_summary.json", "w",
              encoding="utf-8") as f:
        json.dump({"classify": summary, "apply": applied}, f,
                  ensure_ascii=False, indent=1)
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
