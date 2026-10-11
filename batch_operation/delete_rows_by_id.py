"""按 id 列表删除 file_metadata 记录及其关联行（先备份整行、复验文件不在盘）
列表列需含 id（制表符分隔，含表头）。
用法：delete_rows_by_id.py --list f.tsv --apply
"""
import argparse, asyncio, csv, datetime, os, sys, collections

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

STORE = "/storage"
CHILD = [("text_embeddings", "source_id"), ("paragraph", "source_id"), ("embedding_pending_queue", "source_id"),
         ("page_skip_state", "source_id"), ("file_dataset", "file_id")]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    ids, meta = [], {}
    for row in csv.DictReader(open(a.list, encoding="utf-8"), delimiter="\t"):
        rid = (row.get("id") or "").strip()
        if rid:
            ids.append(rid)
            meta[rid] = row
    ids = list(dict.fromkeys(ids))
    print(f"待删 {len(ids)} 条")
    conn = await C.get_conn()
    try:
        rows = await conn.fetch("SELECT * FROM file_metadata WHERE id = ANY($1::uuid[])", ids)
        print(f"  命中 {len(rows)}/{len(ids)}")
        # 复验：文件必须仍不在盘
        live = [r["path"] for r in rows if os.path.exists(os.path.join(STORE, r["path"][7:]) if r["path"].startswith("../app/") else os.path.join(STORE, r["path"]))]
        if live:
            print(f"  !! {len(live)} 条文件其实存在，剔除不删")
            live = set(live)
            rows = [r for r in rows if r["path"] not in live]
        ids = [r["id"] for r in rows]
        cnt = collections.Counter()
        for t, col in CHILD:
            cnt[t] = await conn.fetchval(f"SELECT count(*) FROM {t} WHERE {col} = ANY($1::uuid[])", ids)
        print("  关联行:", dict(cnt))
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"batch_operation/reports/deleted_rows_{stamp}.tsv"
        cols = list(rows[0].keys()) if rows else []
        with open(bak, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(cols + ["删除理由", "对照路径"])
            for r in rows:
                m = meta.get(str(r["id"]), {})
                w.writerow([str(r[c]) for c in cols] +
                           [m.get("理由") or m.get("规则") or "", m.get("target_path") or m.get("existing_path") or ""])
        print(f"  整行备份 → {bak}")
        if not a.apply:
            print("[dry-run] 未删除")
            return
        async with conn.transaction():
            for t, col in CHILD:
                await conn.execute(f"DELETE FROM {t} WHERE {col} = ANY($1::uuid[])", ids)
            await conn.execute("DELETE FROM text_embeddings WHERE source_id = ANY($1::uuid[])", ids)
            n = await conn.execute("DELETE FROM file_metadata WHERE id = ANY($1::uuid[])", ids)
        print(f"[apply] {n}")
    finally:
        await conn.close()

asyncio.run(main())
