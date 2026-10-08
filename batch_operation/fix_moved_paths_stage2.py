"""fix_moved_paths 第二阶段：dup_conflict 去重 + truly_gone 删除（+ multi 列出）。

2026-10-08 用户指令：
  - dup_conflict 123 条：按“保留文本多的一侧”去重（2% 容差内优先目标行，免改路径；
    被删的是目标行时老行接管 new path —— 与 fix_stale_paths 的 dup 规则一致）。
  - truly_gone 977 条：如无关联 paragraph 则删除行（先删其队列行）；有段落的保留并报告。
  - multi 1 条：只列出，不自动处理。

输入 = reports/filemeta_moved_analysis.tsv（阶段一分类结果，路径按字符串回读）。
每条处理前在线复核：行还在、目标行还在、path 在 /storage 上确实无文件（gone 删除的前置守卫）。

用法（容器内 cwd /app）：
  python batch_operation/fix_moved_paths_stage2.py                # dry-run
  python batch_operation/fix_moved_paths_stage2.py --apply --limit 2
  python batch_operation/fix_moved_paths_stage2.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import os
import sys

import common as C

log = C.setup_logging("fix_moved_paths_stage2",
                      "batch_operation/reports/fix_moved_paths_stage2.log")

TSV = "batch_operation/reports/filemeta_moved_analysis.tsv"
EMB_BATCH = 4000
TOL = 0.98  # 目标行文本 >= 老行的 98% 即留目标行（免改路径）


def load_classes() -> dict[str, list[tuple[str, str]]]:
    classes: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    with open(TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            classes[row["class"]].append((row["old_path"], row["new_path"]))
    return classes


async def row_id(conn, path: str) -> str | None:
    return await conn.fetchval("SELECT id FROM file_metadata WHERE path = $1", path)


async def para_stats(conn, fid: str) -> tuple[int, int]:
    """(段落数, 总字符数)"""
    r = await conn.fetchrow(
        "SELECT count(*) AS n, coalesce(sum(length(coalesce(content,''))),0) AS \"L\" "
        "FROM paragraph WHERE source_id = $1::uuid", fid)
    return int(r["n"]), int(r["L"])


async def dup_decide(conn, old_id: str, owner_id: str) -> tuple[str, str, int, int, int, int]:
    """返回 (keep, drop, gc, oc, gl, ol) —— 决策规则与 fix_stale_paths 一致。"""
    gc, gl = await para_stats(conn, old_id)
    oc, ol = await para_stats(conn, owner_id)
    if oc == 0:
        keep, drop = old_id, owner_id        # 目标行空 → 老行接管路径
    elif gc == 0:
        keep, drop = owner_id, old_id
    elif ol >= gl * TOL:
        keep, drop = owner_id, old_id        # 2% 容差内留目标行
    else:
        keep, drop = old_id, owner_id        # 老行文本更多 → 删目标行、老行接管
    return keep, drop, gc, oc, gl, ol


async def merge_pair(conn, old_path: str, new_path: str, old_id: str, owner_id: str,
                     st: dict) -> None:
    """一个 pair 一个事务。"""
    keep, drop, gc, oc, gl, ol = await dup_decide(conn, old_id, owner_id)
    if keep == old_id:
        st["keep_old"] += 1
        st["chars_dropped"] += ol          # 删的是目标行
    else:
        st["keep_target"] += 1
        st["chars_dropped"] += gl          # 删的是老行
    async with conn.transaction():
        # 1) 数据集链接并到 keep
        await conn.execute(
            "INSERT INTO file_dataset (file_id, dataset_id) "
            "SELECT $1::uuid, fd.dataset_id FROM file_dataset fd "
            "WHERE fd.file_id = $2::uuid ON CONFLICT DO NOTHING", keep, drop)
        await conn.execute("DELETE FROM file_dataset WHERE file_id = $1::uuid", drop)
        await conn.execute(
            "DELETE FROM embedding_pending_queue WHERE source_id = $1::uuid", drop)
        # 2) 先按 id 批量删 drop 侧嵌入，再删段落（残余孤儿嵌入靠 FK 级联兜底）
        ids = [r["id"] for r in await conn.fetch(
            "SELECT id FROM paragraph WHERE source_id = $1::uuid", drop)]
        for j in range(0, len(ids), EMB_BATCH):
            await conn.execute(
                "DELETE FROM text_embeddings WHERE source_id = $1::uuid "
                "AND id = ANY($2::uuid[])", drop, ids[j:j + EMB_BATCH])
        await conn.execute("DELETE FROM paragraph WHERE source_id = $1::uuid", drop)
        st["dropped_paras"] += len(ids)
        # 3) 删 drop 行（守卫：仍挂着段落就放弃）；drop 是目标行时老行接管路径
        tag = await conn.execute(
            "DELETE FROM file_metadata WHERE id = $1::uuid "
            "AND NOT EXISTS (SELECT 1 FROM paragraph WHERE source_id = $1)", drop)
        dropped = int(tag.split()[-1])
        if dropped and drop == owner_id:
            await conn.execute(
                "UPDATE file_metadata SET path = $1 WHERE id = $2", new_path, keep)
            st["path_reassigned"] += 1
        st["rows_deleted"] += dropped
        if not dropped:
            st["row_delete_guarded"] += 1


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="每类最多处理 N 条（调试）")
    args = ap.parse_args()

    from jindai.storage import storage

    od = storage.safe_join("OneDrive")
    base = od[: -len("OneDrive")]            # /storage/ 根（与阶段一同源）

    classes = load_classes()
    log.info("[load] %s", json.dumps({k: len(v) for k, v in sorted(classes.items())},
                                     ensure_ascii=False))
    dups = classes.get("dup_conflict", [])
    gones = classes.get("truly_gone", [])
    multis = classes.get("multi", []) + classes.get("multi_id", [])
    if args.limit:
        dups, gones = dups[:args.limit], gones[:args.limit]

    conn = await C.get_conn()
    st = collections.defaultdict(int)
    plan: list[dict] = []

    # ---------- dup_conflict ----------
    for old, new in dups:
        old_id = await row_id(conn, old)
        if old_id is None:
            st["dup_old_row_missing"] += 1
            continue
        owner_id = await conn.fetchval(
            "SELECT id FROM file_metadata WHERE path = $1 AND id <> $2::uuid",
            new, old_id)
        if owner_id is None:
            # 目标路径当前无主 → 老行直接接管（uniq_path 有守卫）
            if not os.path.exists(os.path.join(base, old)):
                if args.apply:
                    async with conn.transaction():
                        tag = await conn.execute(
                            "UPDATE file_metadata SET path = $1 WHERE id = $2::uuid "
                            "AND NOT EXISTS (SELECT 1 FROM file_metadata WHERE path = $1)",
                            new, old_id)
                        if int(tag.split()[-1]):
                            st["rename_takeover"] += 1
                else:
                    st["rename_takeover"] += 1
            else:
                st["old_still_on_disk"] += 1
            continue
        if os.path.exists(os.path.join(base, old)):
            st["old_still_on_disk"] += 1
            continue
        keep, drop, gc, oc, gl, ol = await dup_decide(conn, old_id, owner_id)
        plan.append({"kind": "dup", "old": old, "new": new,
                     "keep": "old" if keep == old_id else "target",
                     "old_paras": gc, "old_chars": gl,
                     "target_paras": oc, "target_chars": ol})
        st["dup_processed"] += 1
        if args.apply:
            try:
                await merge_pair(conn, old, new, old_id, owner_id, st)
            except Exception as ex:
                st["dup_failed"] += 1
                log.warning("[dup] failed %s: %s", old[:70], str(ex)[:200])

    # ---------- truly_gone ----------
    gone_report: list[dict] = []
    for gi, (old, _) in enumerate(gones, 1):
        fid = await row_id(conn, old)
        if fid is None:
            st["gone_row_missing"] += 1
            continue
        if os.path.exists(os.path.join(base, old)):
            st["gone_now_on_disk"] += 1          # 分类后文件又出现 → 跳过
            continue
        n, L = await para_stats(conn, fid)
        if n:
            gone_report.append({"path": old, "paras": n, "chars": L,
                                "action": "kept_has_paragraphs"})
            st["gone_kept_paras"] += 1
            st["gone_kept_paras_total"] += n
            continue
        if args.apply:
            try:
                async with conn.transaction():
                    q = await conn.execute(
                        "DELETE FROM embedding_pending_queue WHERE source_id = $1::uuid", fid)
                    st["gone_queue_deleted"] += int(q.split()[-1])
                    tag = await conn.execute(
                        "DELETE FROM file_metadata WHERE id = $1::uuid "
                        "AND NOT EXISTS (SELECT 1 FROM paragraph WHERE source_id = $1)", fid)
                    dropped = int(tag.split()[-1])
                    st["gone_deleted"] += dropped
                    if not dropped:
                        st["gone_guarded"] += 1
                    gone_report.append({"path": old, "paras": 0,
                                        "action": "deleted" if dropped else "guarded"})
            except Exception as ex:
                st["gone_failed"] += 1
                log.warning("[gone] failed %s: %s", old[:70], str(ex)[:200])
        else:
            st["gone_deletable"] += 1
            gone_report.append({"path": old, "paras": 0, "action": "deletable"})
        if gi % 200 == 0:
            log.info("[gone] %d/%d deleted=%d kept_paras=%d", gi, len(gones),
                     st["gone_deleted"], st["gone_kept_paras"])

    # ---------- multi（只列） ----------
    multi_report = []
    for old, _ in multis:
        multi_report.append({"old": old, "action": "manual_review_only"})

    summary = {"mode": "apply" if args.apply else "dry-run",
               "limit": args.limit or None,
               "counters": {k: v for k, v in sorted(st.items())},
               "dup_plan": plan,
               "gone_report": gone_report,
               "multi": multi_report}
    os.makedirs("batch_operation/reports", exist_ok=True)
    out = "batch_operation/reports/fix_moved_paths_stage2_plan.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log.info("[done] mode=%s counters=%s -> %s", summary["mode"],
             json.dumps(summary["counters"], ensure_ascii=False), out)
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
