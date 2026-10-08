"""修复 file_metadata 中指向 OneDrive 的陈旧 PDF 路径（目录/文件改名漂移）。

陈旧 = `path LIKE 'OneDrive/%' AND lower(path) LIKE '%.pdf'` 但字符串在磁盘上已无此文件。
Calibre 停用后**以磁盘为准**：用路径倒数第二段（书名目录）末尾的 Calibre id `(N)`
反查磁盘上任意 `OneDrive/<作者>/<任意书名> (N)/` 目录（作者目录/书名目录改名都能救），
在命中目录里选文件：行内原文件名精确匹配 → 唯一 PDF → 与原文件名最相似者（阈值 0.5）。

分类：
  rename     唯一命中且目标路径无主      → UPDATE path
  dup        目标路径已被别的行占用      → 谁 0 段落谁冗余：冗余行（0 段落 + 0 队列行）
             才删（先移 file_dataset、清队列行），双方都有段落 → manual
  ambiguous  同 id 多个目录 / 目录里多个 PDF 无法唯一 → manual
  gone       磁盘上没有该 id 的目录      → 保留不删（可能整本已下架，另行处理）

dry-run 默认；--apply 单事务落库。跑完 pending 指标会随之下降（行重新可导入）。
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import sys
import time

import common as C

log = C.setup_logging("fix_stale_paths", "batch_operation/reports/fix_stale_paths.log")

ID_RE = re.compile(r"\((\d+)\)\s*$")


def build_disk_index():
    """{book_id: [rel_dir...]} 与 {rel_dir: [pdf 文件名...]}（容器 /storage 视角）。"""
    from jindai.storage import storage

    root = storage.safe_join("OneDrive")
    id_index: dict[str, list[str]] = {}
    dir_files: dict[str, list[str]] = {}
    n = 0
    for dirpath, _dirs, filenames in os.walk(root):
        rel = "OneDrive/" + os.path.relpath(dirpath, root).replace(os.sep, "/")
        parts = rel.split("/")
        pdfs = [f for f in filenames if f.lower().endswith(".pdf")]
        if pdfs:
            dir_files[rel] = pdfs
        if len(parts) == 3:  # OneDrive/<author>/<title (N)>
            m = ID_RE.search(parts[2])
            if m:
                id_index.setdefault(m.group(1), []).append(rel)
        n += 1
        if n % 5000 == 0:
            log.info("[index] walked %d dirs, %d id dirs", n, len(id_index))
    return id_index, dir_files


def resolve(stale: str, id_index, dir_files):
    rel = stale[len("OneDrive/"):] if stale.startswith("OneDrive/") else stale
    parts = rel.split("/")
    if len(parts) != 3:
        return "gone", None, f"not Calibre-shaped ({len(parts)} segs)"
    m = ID_RE.search(parts[1])
    if not m:
        return "gone", None, "no (id) in title dir"
    dirs = id_index.get(m.group(1), [])
    if not dirs:
        return "gone", None, f"id {m.group(1)} not on disk"
    if len(dirs) > 1:
        # 同 id 多目录（新旧作者目录并存等）：只有含 PDF 的目录才可能是真身
        withpdf = [d for d in dirs if dir_files.get(d)]
        if len(withpdf) == 1:
            dirs = withpdf
        else:
            return "ambiguous", None, f"id {m.group(1)} in {len(dirs)} dirs: {dirs[:3]}"
    d = dirs[0]
    pdfs = dir_files.get(d, [])
    if not pdfs:
        return "gone", None, f"dir empty of pdf: {d}"
    base = parts[2]
    if base in pdfs:
        return "rename", f"{d}/{base}", "same filename"
    if len(pdfs) == 1:
        return "rename", f"{d}/{pdfs[0]}", "single pdf in dir"
    close = difflib.get_close_matches(base, pdfs, n=2, cutoff=0.5)
    if len(close) == 1 or (len(close) == 2 and
                           difflib.SequenceMatcher(None, base, close[0]).ratio() -
                           difflib.SequenceMatcher(None, base, close[1]).ratio() > 0.08):
        return "rename", f"{d}/{close[0]}", f"fuzzy {close[0][:40]!r}"
    return "ambiguous", None, f"{len(pdfs)} pdfs, no clear match: {pdfs[:3]}"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试）")
    args = ap.parse_args()

    from jindai.storage import storage

    log.info("[index] scanning disk ...")
    id_index, dir_files = build_disk_index()
    log.info("[index] id dirs=%d, dirs-with-pdf=%d", len(id_index), len(dir_files))

    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT id, path FROM file_metadata "
        "WHERE path LIKE $1 AND lower(path) LIKE '%.pdf' ORDER BY path",
        C.ONEDRIVE_LIKE,
    )
    stale = [r for r in rows
             if not os.path.exists(storage.safe_join(r["path"]))]
    ids = [r["id"] for r in stale]
    para_cnt = {}
    if ids:
        got = await conn.fetch(
            "SELECT source_id, count(*) AS n FROM paragraph "
            "WHERE source_id = ANY($1::uuid[]) GROUP BY source_id", ids)
        para_cnt = {g["source_id"]: g["n"] for g in got}
    log.info("[stale] %d stale paths (of scanned rows)", len(stale))
    if args.limit:
        stale = stale[: args.limit]

    plan = []          # (id, new_path, kind, note)
    manual = []
    for r in stale:
        kind, target, note = resolve(r["path"], id_index, dir_files)
        if kind == "rename" and target:
            owner = await conn.fetchval(
                "SELECT id FROM file_metadata WHERE path = $1 AND id <> $2",
                target, r["id"])
            if owner:
                plan.append((r["id"], target, "dup", f"target owned by {owner}; "
                             f"my_paras={para_cnt.get(r['id'], 0)}"))
            else:
                plan.append((r["id"], target, "rename", note))
        else:
            manual.append((r["path"], kind, note))

    counts = {"stale": len(stale)}
    for kind in ("rename", "dup"):
        counts[kind] = sum(1 for p in plan if p[2] == kind)
    counts["ambiguous"] = sum(1 for _, k, _ in manual if k == "ambiguous")
    counts["gone"] = sum(1 for _, k, _ in manual if k == "gone")
    log.info("[plan] %s", json.dumps(counts, ensure_ascii=False))
    for p in plan[:8]:
        log.info("  %s %s -> %s (%s)", p[2], str(p[0])[:8], p[1], p[3])
    for m in manual[:8]:
        log.info("  %s %s (%s)", m[1], m[0], m[2])

    os.makedirs("batch_operation/reports", exist_ok=True)
    with open("batch_operation/reports/fix_stale_paths_plan.json", "w",
              encoding="utf-8") as f:
        json.dump({"counts": counts,
                   "plan": [{"id": str(a), "new_path": b, "kind": c, "note": d}
                            for a, b, c, d in plan],
                   "manual": [{"path": a, "kind": b, "note": c}
                              for a, b, c in manual]},
                  f, ensure_ascii=False, indent=1)

    if not args.apply:
        log.info("dry-run only; re-run with --apply to write")
        await conn.close()
        return 0

    # ---- apply：dup 统一“并入 owner”（owner 持有效路径；ghost 段落/嵌入迁移后删 ghost 行）
    # subset（ghost 内容已全部存在于 owner）→ 直接删 ghost 段落；否则搬运。
    EMB_BATCH = 4000
    n_upd = n_del = n_move = n_sub = n_skip = 0
    t0 = time.time()
    for i, (fid, new_path, kind, note) in enumerate(plan, 1):
        if kind == "rename":
            tag = await conn.execute(
                "UPDATE file_metadata SET path = $1 WHERE id = $2 AND path <> $1",
                new_path, fid)
            n_upd += int(tag.split()[-1])
            continue
        owner = note.split("owned by ")[1].split(";")[0]
        try:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT (SELECT count(*) FROM paragraph WHERE source_id=$1::uuid) AS gc, "
                    "(SELECT count(*) FROM paragraph WHERE source_id=$2::uuid) AS oc, "
                    "(SELECT count(*) FROM paragraph g JOIN paragraph o "
                    "  ON o.source_id=$2::uuid AND coalesce(g.content,'')=coalesce(o.content,'') "
                    " WHERE g.source_id=$1::uuid) AS mc", fid, owner)
                gc, oc, mc = row["gc"], row["oc"], row["mc"]
                gl = (await conn.fetchval(
                    "SELECT coalesce(sum(length(coalesce(content,''))),0) "
                    "FROM paragraph WHERE source_id=$1::uuid", fid)) or 0
                ol = (await conn.fetchval(
                    "SELECT coalesce(sum(length(coalesce(content,''))),0) "
                    "FROM paragraph WHERE source_id=$1::uuid", owner)) or 0
                # 决策：留行 = 文本更多的一方（2% 容差内优先 owner，路径不用改）
                if oc == 0:
                    keep, drop = fid, owner          # owner 空 → ghost 接管路径
                elif gc == 0:
                    keep, drop = owner, fid
                elif ol >= gl * 0.98:
                    keep, drop = owner, fid
                else:
                    keep, drop = fid, owner          # ghost 文本更多 → 删 owner、ghost 接管
                # 1) 数据集链接并到 keep 行
                await conn.execute(
                    "INSERT INTO file_dataset (file_id, dataset_id) "
                    "SELECT $1::uuid, fd.dataset_id FROM file_dataset fd "
                    "WHERE fd.file_id = $2::uuid ON CONFLICT DO NOTHING", keep, drop)
                await conn.execute("DELETE FROM file_dataset WHERE file_id = $1::uuid", drop)
                await conn.execute(
                    "DELETE FROM embedding_pending_queue WHERE source_id = $1::uuid", drop)
                # 2) 先按 id 批量删 drop 侧嵌入（source 分区裁剪 + pkey 定位），
                #    再删段落（剩余孤儿嵌入靠 FK 级联兜底）
                ids = [r["id"] for r in await conn.fetch(
                    "SELECT id FROM paragraph WHERE source_id = $1::uuid", drop)]
                for j in range(0, len(ids), EMB_BATCH):
                    await conn.execute(
                        "DELETE FROM text_embeddings WHERE source_id = $1::uuid "
                        "AND id = ANY($2::uuid[])", drop, ids[j:j + EMB_BATCH])
                await conn.execute(
                    "DELETE FROM paragraph WHERE source_id = $1::uuid", drop)
                # 3) 删 drop 行（守卫：仍挂着段落就放弃）；若 drop 是 owner，keep 行接管路径
                tag = await conn.execute(
                    "DELETE FROM file_metadata WHERE id = $1::uuid "
                    "AND NOT EXISTS (SELECT 1 FROM paragraph WHERE source_id = $1)", drop)
                dropped = int(tag.split()[-1])
                if dropped and drop == owner:
                    await conn.execute(
                        "UPDATE file_metadata SET path = $1 WHERE id = $2", new_path, keep)
                    n_upd += 1
                n_del += dropped
        except Exception as ex:
            n_skip += 1
            log.warning("dup merge failed %s: %s", str(fid)[:8], str(ex)[:200])
        if i % 25 == 0:
            log.info("[apply] %d/%d upd=%d move=%d subset=%d rowdel=%d fail=%d %.1fs",
                     i, len(plan), n_upd, n_move, n_sub, n_del, n_skip,
                     time.time() - t0)
    log.info("[apply] done upd=%d move=%d subset=%d rowdel=%d fail=%d in %.1f min",
             n_upd, n_move, n_sub, n_del, n_skip, (time.time() - t0) / 60)
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
