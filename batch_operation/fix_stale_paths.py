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


# 人工核定映射（自动匹配同分时，用首段内容里的卷号/译本信息人工定过靶）：
#   Davidson 2854 -> 同名目录 3541；康德 4636 首段证明是《判断力批判(注释本)》李秋零译 -> 5112；
#   熊十力 4902/4907/4908 首段【第八/第二/第三卷】 -> 全集·第8/2/3卷 (18926/18924/18925)。
MANUAL_MAP = {
    "OneDrive/Donald Davidson/Essays on Actions and Events_ Philosophical Essays (2854)/Essays on Actions and Events_ P - Donald Davidson.pdf":
        "OneDrive/Donald Davidson/Essays on Actions and Events_ Philosophical Essays (3541)",
    "OneDrive/Immanuel Kant/判断力批判 (4636)/Pan Duan Li Pi Pan - Immanuel Kant.pdf":
        "OneDrive/Immanuel Kant/判断力批判 注释本 (5112)",
    "OneDrive/熊十力/熊十力全集 (4902)/Xiong Shi Li Quan Ji - Xiong Shi Li.pdf":
        "OneDrive/熊十力/熊十力全集·第8卷 (18926)",
    "OneDrive/熊十力/熊十力全集 (4907)/Xiong Shi Li Quan Ji - Xiong Shi Li.pdf":
        "OneDrive/熊十力/熊十力全集·第2卷 (18924)",
    "OneDrive/熊十力/熊十力全集 (4908)/Xiong Shi Li Quan Ji - Xiong Shi Li.pdf":
        "OneDrive/熊十力/熊十力全集·第3卷 (18925)",
}


def norm(s):
    """归一化用于书名/作者模糊匹配：去标点空白、小写（保留 CJK）。"""
    return re.sub(r"[\W_]+", "", s or "", flags=re.UNICODE).lower()


def build_disk_index():
    """{book_id: [rel_dir...]}、{rel_dir: [pdf 文件名...]} 与 dir_meta 书名索引。"""
    from jindai.storage import storage

    root = storage.safe_join("OneDrive")
    id_index: dict[str, list[str]] = {}
    dir_files: dict[str, list[str]] = {}
    dir_meta: list[tuple] = []
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
            if pdfs:
                dtitle = ID_RE.sub("", parts[2]).strip()
                dir_meta.append((norm(dtitle), norm(parts[1]), rel))
        n += 1
        if n % 5000 == 0:
            log.info("[index] walked %d dirs, %d id dirs", n, len(id_index))
    return id_index, dir_files, dir_meta


def _pick_file(base, pdfs):
    """在命中目录里选文件：原名精确 → 唯一 PDF → 相似度最明显者。"""
    if base in pdfs:
        return base, "same filename"
    if len(pdfs) == 1:
        return pdfs[0], "single pdf in dir"
    close = difflib.get_close_matches(base, pdfs, n=2, cutoff=0.5)
    if len(close) == 1 or (len(close) == 2 and
                           difflib.SequenceMatcher(None, base, close[0]).ratio() -
                           difflib.SequenceMatcher(None, base, close[1]).ratio() > 0.08):
        return close[0], f"fuzzy {close[0][:40]!r}"
    return None, f"{len(pdfs)} pdfs, no clear match: {pdfs[:3]}"


def _resolve_title(stale, parts, row, dir_meta, dir_files):
    """id 反查失败后的兜底：按 书名(+作者) 匹配磁盘目录（换 id/作者目录改名都能救）。"""
    title = (row["title"] if row else None) or ID_RE.sub("", parts[1]).strip()
    author = parts[1]
    if row and row["authors"]:
        author = row["authors"][0]
    rt, ra = norm(title), norm(author)
    scored = []
    for nt, na, d in dir_meta:
        if rt and (rt in nt or nt in rt) and min(len(rt), len(nt)) >= 3:
            ts = 0.90 + 0.09 * min(len(rt), len(nt)) / max(len(rt), len(nt))
        else:
            r = difflib.SequenceMatcher(None, rt, nt).ratio() if rt and nt else 0.0
            if r < 0.62:
                continue
            ts = r
        ascore = difflib.SequenceMatcher(None, ra, na).ratio() if ra and na else 0.0
        pdfs_d = dir_files.get(d, [])
        if parts[2] in pdfs_d:
            fsim = 1.2  # 目录内存在与旧文件完全同名的 PDF → 强信号
        else:
            fsim = max([difflib.SequenceMatcher(None, parts[2], f).ratio()
                        for f in pdfs_d] or [0.0])
        scored.append((ts + 0.35 * ascore + 0.25 * fsim, ts, ascore, d, fsim))
    if not scored:
        return "gone", None, f"{stale.rsplit('/', 1)[-1][:50]}: no id & no title match"
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    if len(scored) > 1 and best[0] - scored[1][0] < 0.05:
        return "ambiguous", None, f"title match too close: {[(s[3], round(s[0], 2)) for s in scored[:3]]}"
    d = best[3]
    f, note = _pick_file(parts[2], dir_files[d])
    if not f:
        return "ambiguous", None, f"title-matched {d} but {note}"
    return "rename", f"{d}/{f}", f"title-match {d} (score {best[0]:.2f}; {note})"


def resolve(stale: str, id_index, dir_files, row=None, dir_meta=None):
    rel = stale[len("OneDrive/"):] if stale.startswith("OneDrive/") else stale
    parts = rel.split("/")
    if len(parts) != 3:
        return "gone", None, f"not Calibre-shaped ({len(parts)} segs)"
    if stale in MANUAL_MAP:
        d = MANUAL_MAP[stale]
        f, note = _pick_file(parts[2], dir_files.get(d, []))
        if f:
            return "rename", f"{d}/{f}", f"manual-map {d} ({note})"
        return "ambiguous", None, f"manual-map target has no matching pdf: {d}"
    m = ID_RE.search(parts[1])
    if not m:
        return "gone", None, "no (id) in title dir"
    dirs = id_index.get(m.group(1), [])
    if not dirs:
        if dir_meta:
            return _resolve_title(stale, parts, row, dir_meta, dir_files)
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
    f, note = _pick_file(parts[2], pdfs)
    if not f:
        return "ambiguous", None, note
    return "rename", f"{d}/{f}", note


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试）")
    args = ap.parse_args()

    from jindai.storage import storage

    log.info("[index] scanning disk ...")
    id_index, dir_files, dir_meta = build_disk_index()
    log.info("[index] id dirs=%d, dirs-with-pdf=%d, title-dirs=%d",
             len(id_index), len(dir_files), len(dir_meta))

    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT id, path, title, authors FROM file_metadata "
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
        kind, target, note = resolve(r["path"], id_index, dir_files,
                                     row=r, dir_meta=dir_meta)
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

    # 同一 target 被多条陈旧行命中（换 id 的多卷/多副本）：段落字符最多者做 rename，
    # 其余转 dup 指向它，apply 时逐对并入，避免 path 唯一约束冲突。
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, (_fid, np_, _kind, _note) in enumerate(plan):
        if _kind == "rename":
            groups[np_].append(idx)
    for np_, idxs in groups.items():
        if len(idxs) < 2:
            continue
        chars = {}
        for i in idxs:
            n = await conn.fetchval(
                "SELECT coalesce(sum(length(coalesce(content,''))),0) "
                "FROM paragraph WHERE source_id=$1::uuid", plan[i][0])
            chars[i] = int(n or 0)
        best = max(idxs, key=lambda i: chars[i])
        for i in idxs:
            if i == best:
                continue
            plan[i] = (plan[i][0], np_, "dup",
                       f"target owned by {plan[best][0]}; "
                       f"my_paras={para_cnt.get(plan[i][0], 0)}; group-dup")
        log.info("[group] %s -> keeper %s, %d dups",
                 np_[-70:], str(plan[best][0])[:8], len(idxs) - 1)

    counts = {"stale": len(stale)}
    for kind in ("rename", "dup"):
        counts[kind] = sum(1 for p in plan if p[2] == kind)
    counts["ambiguous"] = sum(1 for _, k, _ in manual if k == "ambiguous")
    counts["gone"] = sum(1 for _, k, _ in manual if k == "gone")
    log.info("[plan] %s", json.dumps(counts, ensure_ascii=False))
    for p in plan[:40]:
        log.info("  %s %s -> %s (%s)", p[2], str(p[0])[:8], p[1], p[3])
    for m in manual[:40]:
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
        owner = None
        try:
            async with conn.transaction():
                owner = await conn.fetchval(
                    "SELECT id FROM file_metadata WHERE path = $1 AND id <> $2",
                    new_path, fid)
                if owner is None:
                    # 目标路径当前无主（前序合并已把原 owner 删掉/或本来就没主）→ 直接接管
                    tag = await conn.execute(
                        "UPDATE file_metadata SET path = $1 WHERE id = $2 "
                        "AND NOT EXISTS (SELECT 1 FROM file_metadata WHERE path = $1)",
                        new_path, fid)
                    if int(tag.split()[-1]):
                        n_upd += 1
                    continue
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
