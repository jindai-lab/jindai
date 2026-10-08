
import csv, asyncio, common as C
async def main():
    conn = await C.get_conn()
    rows = list(csv.DictReader(open("batch_operation/reports/filemeta_moved_analysis.tsv", encoding="utf-8"), delimiter="\t"))
    dups = [r for r in rows if r["class"]=="dup_conflict"]
    old_present = tgt_present = 0
    missing_tgt = []
    for r in dups:
        if await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", r["old_path"]): old_present += 1
        if await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", r["new_path"]): tgt_present += 1
        else: missing_tgt.append(r["new_path"])
    print("dup old still present:", old_present, "| target present:", tgt_present, "/", len(dups))
    for p in missing_tgt[:10]: print("MISSING TARGET:", p)
    # keep-old pairs: target path should now be owned by the (former) old row with its paras
    st = await conn.fetch("""
        SELECT f.path, (SELECT count(*) FROM paragraph p WHERE p.source_id=f.id) n
        FROM file_metadata f
        WHERE f.path LIKE 'OneDrive/中村正直/%' OR f.path LIKE 'OneDrive/郑元者/%'
           OR f.path LIKE 'OneDrive/钱学森/科学的艺术与艺术的科学%'""")
    for r in st: print("KEEP_OLD check:", r["path"], r["n"])
    await conn.close()
asyncio.run(main())
