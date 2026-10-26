"""字节级重复组分侧统计与合并决策（dry-run 默认，--apply 执行无歧义组）。

输入：scratch/byte_dup_groups.json（basename+真实盘大小 完全一致的组）。
决策阶梯（2026-10-26 用户裁定 + 站规）：
  1. 一侧有正文、另一侧没有 → 留有正文侧；
  2. 两侧都有正文 → 平手/差异小：Series>Journals、拉丁作者目录>中文作者目录；
     其余 → 列入 ambiguous 交用户裁定；
  3. 两侧都无正文 → 留书目信息更全（title 非空）的一侧，再平 Series>Journals。
Zotero 侧只删 jindai 行（Zotero 本体不动）。报告落 reports/byte_dup_plan.json。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/app")

import common as C

log = C.setup_logging("byte_dup", "batch_operation/reports/byte_dup.log")
GROUPS = "/app/batch_operation/reports/byte_dup_groups.json"
OUT = "batch_operation/reports/byte_dup_plan.json"
ROOT = "/storage"
CJK = re.compile("[一-鿿]")


def side_of(path: str) -> str:
    return path.split("/")[0]


def is_cjk_author(path: str) -> bool:
    seg = path.split("/")
    return len(seg) >= 2 and seg[0] == "OneDrive" and bool(CJK.search(seg[1]))


async def row_stats(conn, path: str):
    row = await conn.fetchrow(
        "SELECT id, coalesce(btrim(title),'') t, cardinality(authors) na "
        "FROM file_metadata WHERE path = $1", path)
    if not row:
        return {"row": False}
    npara = await conn.fetchval(
        "SELECT count(*) FROM paragraph WHERE source_id = $1", row["id"])
    chars = await conn.fetchval(
        "SELECT coalesce(sum(length(content)),0) FROM paragraph "
        "WHERE source_id = $1", row["id"])
    return {"row": True, "id": str(row["id"]), "title": row["t"],
            "nauthors": int(row["na"] or 0), "npara": int(npara or 0),
            "chars": int(chars or 0)}


def decide(group_stats):
    """返回 (decision, keep_path, drop_paths, reason)。decision ∈ auto/ambiguous."""
    sides = [(g["path"], g["stat"]) for g in group_stats if g["stat"]["row"]]
    if len(sides) < 2:
        return "skip", None, [], "fewer than 2 DB rows"
    texted = [s for s in sides if s[1]["chars"] > 0]
    if len(texted) == 1:
        keep = texted[0]
        drops = [s for s in sides if s is not keep]
        return "auto", keep[0], [d[0] for d in drops], "one side has text"
    if len(texted) == 0:
        # 都无正文：title 非空优先 → Series> Journals → 非中文作者目录优先
        def key(s):
            p = s[0]
            return (bool(s[1]["title"]), side_of(p) in ("Series", "OneDrive"),
                    not is_cjk_author(p))
        sides.sort(key=key, reverse=True)
        keep = sides[0]
        return "auto", keep[0], [s[0] for s in sides[1:]], "no text anywhere; metadata+tree priority"
    # 双方都有正文
    tp = {s[0]: s[1]["chars"] for s in texted}
    tops = {side_of(p) for p in tp}
    # 2026-10-26 用户裁定扩展：跨树专题对一律留 Series（保住手工策划文件夹）；
    # 同作者双导入留编号小的（首次导入）。均为字节级一致文件，无文本损失。
    if tops == {"OneDrive", "Series"}:
        keep = next(p for p in tp if side_of(p) == "Series")
        return "auto", keep, [p for p in tp if p != keep], "user rule: keep Series over OneDrive dup"
    if tops == {"Series", "Zotero"}:
        keep = next(p for p in tp if side_of(p) == "Series")
        return "auto", keep, [p for p in tp if p != keep], "user rule: keep Series over Zotero dup (Zotero本体不动)"
    if tops == {"OneDrive"}:
        def idkey(pth):
            m = re.search(r"\((\d+)\)", pth.split("/")[2] if len(pth.split("/")) > 2 else "")
            return int(m.group(1)) if m else 10**9
        keep = min(tp, key=idkey)
        return "auto", keep, [p for p in tp if p != keep], "same-author double import: keep lowest id"
    if tops == {"Journals", "Series"}:
        keep = next(p for p in tp if side_of(p) == "Series")
        return "auto", keep, [p for p in tp if p != keep], "user rule: Series > Journals"
    cj = {is_cjk_author(p) for p in tp}
    if cj == {True, False} and all(side_of(p) == "OneDrive" for p in tp):
        keep = next(p for p in tp if not is_cjk_author(p))
        return "auto", keep, [p for p in tp if p != keep], "user rule: latin author > translated"
    lo, hi = min(tp.values()), max(tp.values())
    if hi <= lo * 1.02:
        return "ambiguous", None, sorted(tp), "both sides text, near-equal"
    keep = max(tp, key=tp.get)
    return "auto", keep, [p for p in tp if p != keep], "keep more text"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    groups = json.load(open(GROUPS, encoding="utf-8"))
    conn = await C.get_conn()
    plan, ambiguous, skipped = [], [], []
    for g in groups:
        stats = []
        for p in g["paths"]:
            st = await row_stats(conn, p)
            stats.append({"path": p, "stat": st})
        dec, keep, drops, reason = decide(stats)
        entry = {"basename": g["basename"], "size": g["size"], "decision": dec,
                 "reason": reason, "keep": keep, "drops": drops,
                 "sides": stats}
        if dec == "auto":
            plan.append(entry)
        elif dec == "ambiguous":
            ambiguous.append(entry)
        else:
            skipped.append(entry)
    stat = {"groups": len(groups), "auto": len(plan),
            "ambiguous": len(ambiguous), "skipped": len(skipped),
            "ts": time.strftime("%F %T")}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": stat, "plan": plan, "ambiguous": ambiguous,
                   "skipped": skipped}, f, ensure_ascii=False, indent=1)
    log.info("[plan] %s", json.dumps(stat, ensure_ascii=False))
    for e in ambiguous:
        log.info("[ambiguous] %s :: %s", e["basename"][:60],
                 [p[:70] for p in e["drops"] + [e["keep"]]][:0] or e["drops"])

    if args.apply and plan:
        done = failed = 0
        for e in plan:
            keep_id = next(s["stat"]["id"] for s in e["sides"]
                           if s["path"] == e["keep"])
            ok = await apply_merge(conn, keep_id, e)
            if ok:
                done += 1
            else:
                failed += 1
        log.info("[apply] merged groups ok=%d failed=%d", done, failed)
    await conn.close()
    return 0


async def apply_merge(conn, keep_id: str, entry: dict) -> bool:
    """删 drop 侧（子表→行→文件→空目录），keep 侧不动。"""
    import uuid as _uuid
    for dp in entry["drops"]:
        st = next(s["stat"] for s in entry["sides"] if s["path"] == dp)
        if not st.get("row"):
            continue
        did = _uuid.UUID(st["id"])
        try:
            async with conn.transaction():
                await conn.execute(
                    "SET LOCAL session_replication_role = replica")
                for stmt in (
                    "DELETE FROM file_dataset WHERE file_id=$1",
                    "DELETE FROM text_embeddings WHERE source_id=$1",
                    "DELETE FROM embedding_pending_queue WHERE source_id=$1",
                    "DELETE FROM page_skip_state WHERE source_id=$1",
                    "DELETE FROM paragraph WHERE source_id=$1",
                ):
                    await conn.execute(stmt, did)
                r = await conn.execute(
                    "DELETE FROM file_metadata WHERE id=$1 "
                    "  AND NOT EXISTS (SELECT 1 FROM paragraph p "
                    "                  WHERE p.source_id=file_metadata.id)", did)
            full = os.path.join(ROOT, dp)
            if os.path.isfile(full):
                os.unlink(full)
            d = os.path.dirname(full)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
            log.info("[merge] drop %s (%s)", dp[:90], r.split()[-1])
            return True
        except Exception as ex:  # noqa: BLE001
            log.warning("[merge] %s failed: %s", dp[:90], str(ex)[:150])
            return False
    return True


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
