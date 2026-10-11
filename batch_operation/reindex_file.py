"""把某文件的段落/嵌入清空，交给导入器的页集续传用新文件重读该页。

用法：python reindex_file.py <storage相对路径> [--apply]
"""
import argparse, asyncio, csv, datetime, sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    conn = await C.get_conn()
    try:
        row = await conn.fetchrow("SELECT id, path FROM file_metadata WHERE path=$1", a.path)
        if not row:
            print("找不到行"); return
        fid = row["id"]
        ps = await conn.fetch(
            "SELECT id, source_page, outline, content FROM paragraph WHERE source_id=$1 ORDER BY source_page, id", fid)
        emb = await conn.fetchval("SELECT count(*) FROM text_embeddings WHERE source_id=$1", fid)
        q = await conn.fetchval("SELECT count(*) FROM embedding_pending_queue WHERE source_id=$1", fid)
        sk = await conn.fetch("SELECT source_page, reason FROM page_skip_state WHERE source_id=$1", fid)
        print(f"段落 {len(ps)} | 嵌入 {emb} | 队列 {q} | 跳页记录 {sk}")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"batch_operation/reports/paragraph_backup_{stamp}.tsv"
        with open(bak, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "source_page", "outline", "content"])
            for r in ps:
                w.writerow([r["id"], r["source_page"], r["outline"] or "", r["content"] or ""])
        print(f"段落已备份 → {bak}")
        if not a.apply:
            print("[dry-run] 未删除"); return
        async with conn.transaction():
            await conn.execute("DELETE FROM text_embeddings WHERE source_id=$1", fid)
            await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id=$1", fid)
            await conn.execute("DELETE FROM page_skip_state WHERE source_id=$1", fid)
            n = await conn.execute("DELETE FROM paragraph WHERE source_id=$1", fid)
        left = await conn.fetchval("SELECT count(*) FROM paragraph WHERE source_id=$1", fid)
        print(f"[apply] {n}; 剩余段落 {left}（该页已成 gap，导入器下一轮读新文件重导）")
    finally:
        await conn.close()

asyncio.run(main())
