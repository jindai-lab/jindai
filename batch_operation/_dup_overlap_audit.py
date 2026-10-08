"""只读：对 filemeta_moved_analysis.tsv 的 dup_conflict 123 对算内容重合度（去重损失核算）。"""
import asyncio, csv, json, sys
import common as C

log = C.setup_logging("dup_overlap", "batch_operation/reports/dup_overlap.log")

async def main():
    conn = await C.get_conn()
    pairs = [(r["old_path"], r["new_path"]) for r in csv.DictReader(
        open("batch_operation/reports/filemeta_moved_analysis.tsv", encoding="utf-8"),
        delimiter="\t") if r["class"] == "dup_conflict"]
    out = []
    for i, (old, new) in enumerate(pairs, 1):
        old_id = await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", old)
        new_id = await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", new)
        if not old_id or not new_id or old_id == new_id:
            out.append({"old": old, "skip": "row missing/identical"})
            continue
        r = await conn.fetchrow("""
            SELECT
              (SELECT coalesce(sum(length(coalesce(content,''))),0) FROM paragraph
                WHERE source_id=$1::uuid
                  AND NOT EXISTS (SELECT 1 FROM paragraph o
                    WHERE o.source_id=$2::uuid AND o.content = paragraph.content)) AS old_uniq,
              (SELECT coalesce(sum(length(coalesce(content,''))),0) FROM paragraph
                WHERE source_id=$2::uuid
                  AND NOT EXISTS (SELECT 1 FROM paragraph g
                    WHERE g.source_id=$1::uuid AND g.content = paragraph.content)) AS new_uniq,
              (SELECT coalesce(sum(length(coalesce(content,''))),0) FROM paragraph
                WHERE source_id=$1::uuid) AS old_total,
              (SELECT coalesce(sum(length(coalesce(content,''))),0) FROM paragraph
                WHERE source_id=$2::uuid) AS new_total""", old_id, new_id)
        out.append({"old": old, "new": new,
                    "old_uniq": r["old_uniq"], "new_uniq": r["new_uniq"],
                    "old_total": r["old_total"], "new_total": r["new_total"]})
        if i % 30 == 0:
            log.info("[overlap] %d/%d", i, len(pairs))
    json.dump(out, open("batch_operation/reports/dup_overlap.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    tu = sum(o.get("old_uniq") or 0 for o in out)
    tn = sum(o.get("new_uniq") or 0 for o in out)
    log.info("[done] pairs=%d old_unique_chars=%d new_unique_chars=%d",
             len(out), tu, tn)
    await conn.close()
    return 0

sys.exit(asyncio.run(main()))
