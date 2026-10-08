"""multi（Rorty 哲学、文学和政治）处置 2026-10-09 用户指令：
  1) 老行 Authors/Rorty, Richard/哲学、文学和政治 - Richard Rorty.pdf 并入
     OneDrive/Richard Rorty/哲学、文学和政治 (9148)/... （同书，72% 逐字重合；
     目标行字符 190,919 ≥ 老行 190,527 → 留目标行，规则同 fix_moved_paths_stage2）
  2) (4833) 行书目更正：实为 Embry & Cooper 编《Philosophy, Literature, and
     Politics: Essays Honoring Ellis Sandoz》，非 Rorty 著作。
幂等；dry-run 默认，--apply 落库。
"""
from __future__ import annotations
import argparse, asyncio, sys
import common as C

log = C.setup_logging("multi_rorty_fix", "batch_operation/reports/multi_rorty_fix.log")

OLD = "Authors/Rorty, Richard/哲学、文学和政治 - Richard Rorty.pdf"
TGT = "OneDrive/Richard Rorty/哲学、文学和政治 (9148)/哲学、文学和政治 - Richard Rorty.pdf"
WRONG = "OneDrive/Richard Rorty/哲学、文学和政治 (4833)/哲学、文学和政治 - Richard Rorty.pdf"
NEW_TITLE = "Philosophy, Literature, and Politics: Essays Honoring Ellis Sandoz"
NEW_AUTH = ["Charles R. Embry", "Barry Cooper"]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    conn = await C.get_conn()
    st = {}

    old_id = await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", OLD)
    tgt_id = await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", TGT)
    st["old_row_present"] = bool(old_id)
    if old_id and tgt_id:
        r = await conn.fetchrow(
            "SELECT (SELECT count(*) FROM paragraph WHERE source_id=$1::uuid) oc, "
            "       (SELECT count(*) FROM paragraph WHERE source_id=$2::uuid) tc, "
            "       (SELECT coalesce(sum(length(coalesce(content,''))),0) FROM paragraph WHERE source_id=$1::uuid) ol, "
            "       (SELECT coalesce(sum(length(coalesce(content,''))),0) FROM paragraph WHERE source_id=$2::uuid) tl",
            old_id, tgt_id)
        st.update(old_paras=r["oc"], old_chars=r["ol"], tgt_paras=r["tc"], tgt_chars=r["tl"])
        if args.apply and r["ol"] * 0.98 <= r["tl"]:   # 目标行不输 → 留目标行
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO file_dataset (file_id, dataset_id) "
                    "SELECT $1::uuid, fd.dataset_id FROM file_dataset fd "
                    "WHERE fd.file_id=$2::uuid ON CONFLICT DO NOTHING", tgt_id, old_id)
                await conn.execute("DELETE FROM file_dataset WHERE file_id=$1::uuid", old_id)
                await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id=$1::uuid", old_id)
                ids = [x["id"] for x in await conn.fetch(
                    "SELECT id FROM paragraph WHERE source_id=$1::uuid", old_id)]
                for j in range(0, len(ids), 4000):
                    await conn.execute(
                        "DELETE FROM text_embeddings WHERE source_id=$1::uuid AND id=ANY($2::uuid[])",
                        old_id, ids[j:j + 4000])
                await conn.execute("DELETE FROM paragraph WHERE source_id=$1::uuid", old_id)
                tag = await conn.execute(
                    "DELETE FROM file_metadata WHERE id=$1::uuid "
                    "AND NOT EXISTS (SELECT 1 FROM paragraph WHERE source_id=$1)", old_id)
                st["old_row_deleted"] = int(tag.split()[-1])
                st["old_paras_dropped"] = len(ids)
        elif args.apply:
            log.warning("target chars < 98%% of old (%d vs %d) — NOT merging", r["tl"], r["ol"])
            st["merge_skipped_target_too_short"] = True

    # (4833) 书目更正
    w_id = await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", WRONG)
    st["wrong_row_present"] = bool(w_id)
    if w_id:
        before = await conn.fetchrow(
            "SELECT title, authors::text[] a FROM file_metadata WHERE id=$1::uuid", w_id)
        st["before"] = {"title": before["title"], "authors": before["a"]}
        if args.apply:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE file_metadata SET title=$1, authors=$2::text[] WHERE id=$3::uuid",
                    NEW_TITLE, NEW_AUTH, w_id)
            after = await conn.fetchrow(
                "SELECT title, authors::text[] a FROM file_metadata WHERE id=$1::uuid", w_id)
            st["after"] = {"title": after["title"], "authors": after["a"]}
    log.info("[%s] %s", "apply" if args.apply else "dry-run",
             __import__("json").dumps(st, ensure_ascii=False))
    await conn.close()
    return 0


sys.exit(asyncio.run(main()))
