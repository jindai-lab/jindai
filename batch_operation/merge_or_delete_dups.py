"""重复记录处理（回避 text_embeddings ↔ paragraph 的循环外键，不改挂子表）：
  A 目标行已有正文 → 旧行是纯重复：删旧行子表 + 旧行
  B 目标行无正文   → 旧行是唯一全文：删掉空壳目标行，把旧行 path 改到目标路径（正文保留）
列表列：id / old_path / target_path / 标题 / 作者 / 理由
用法：merge_or_delete_dups.py --list f.tsv [--apply]
"""
import argparse, asyncio, csv, datetime, os, sys
import asyncpg, time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

STORE = "/storage"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    pairs = []
    for r in csv.DictReader(open(a.list, encoding="utf-8"), delimiter="\t"):
        if (r.get("id") or "").strip() and (r.get("target_path") or "").strip():
            pairs.append((r["id"].strip(), r["target_path"].strip(), r.get("理由", "")))
    conn = await C.get_conn()
    try:
        info = {str(r["id"]): r for r in await conn.fetch(
            "SELECT id, path, coalesce(title,'') AS title, coalesce(array_to_string(authors,'|'),'') AS au "
            "FROM file_metadata WHERE id = ANY($1::uuid[])", [p[0] for p in pairs])}
        tmap = {r["path"]: str(r["id"]) for r in await conn.fetch(
            "SELECT id, path FROM file_metadata WHERE path = ANY($1::text[])", list({p[1] for p in pairs}))}
        print(f"待处理 {len(pairs)} 对 | 旧行命中 {len(info)} | 目标命中 {len(tmap)}")

        A, B, skip = [], [], []
        for pid, tp, why in pairs:
            if pid not in info or tp not in tmap:
                skip.append((pid, tp, "缺旧行或目标行"))
                continue
            if os.path.exists(os.path.join(STORE, info[pid]["path"])):
                skip.append((pid, tp, "旧文件其实在盘"))
                continue
            np_ = await conn.fetchval("SELECT count(*) FROM paragraph WHERE source_id = $1::uuid", pid)
            tn_ = await conn.fetchval("SELECT count(*) FROM paragraph WHERE source_id = $1::uuid", tmap[tp])
            (A if tn_ > 0 or np_ == 0 else B).append((pid, tmap[tp], why, np_, tn_))
        print(f"  A 删纯重复旧行 {len(A)} | B 保留旧行(改路径到目标、删空壳目标行) {len(B)} | 跳过 {len(skip)}")
        for s in skip[:5]:
            print("    跳过:", s[1][:64], "|", s[2])

        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"batch_operation/reports/deleted_rows_{stamp}.tsv"
        with open(bak, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "path", "title", "authors", "target", "处理", "旧行段落数", "目标段落数"])
            for pid, tid, why, np_, tn_ in A:
                w.writerow([pid, info[pid]["path"], info[pid]["title"], info[pid]["au"], tid, "A 删旧行", np_, tn_])
            for pid, tid, why, np_, tn_ in B:
                w.writerow([pid, info[pid]["path"], info[pid]["title"], info[pid]["au"], tid, "B 保留旧行改路径", np_, tn_])
        print(f"  备份 → {bak}")
        if not a.apply:
            print("[dry-run] 未执行")
            return

        async def run_with_retry(fn, tag):
            for attempt in range(6):
                try:
                    await fn()
                    return True
                except (asyncpg.exceptions.DeadlockDetectedError,
                        asyncpg.exceptions.SerializationError,
                        asyncpg.exceptions.LockNotAvailableError) as e:
                    print(f"    死锁重试 {tag} #{attempt+1}: {type(e).__name__}", flush=True)
                    await asyncio.sleep(4 + attempt * 4)
                except Exception as e:
                    print(f"    !! 失败 {tag}: {type(e).__name__} {str(e)[:90]}", flush=True)
                    return False
            return False

        async def del_a(pid):
            async with conn.transaction():
                await conn.execute("DELETE FROM text_embeddings WHERE source_id = $1::uuid", pid)
                await conn.execute("DELETE FROM paragraph WHERE source_id = $1::uuid", pid)
                await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id = $1::uuid", pid)
                await conn.execute("DELETE FROM file_dataset WHERE file_id = $1::uuid", pid)
                await conn.execute("DELETE FROM file_metadata WHERE id = $1::uuid", pid)

        for pid, tid, why, np_, tn_ in A:
            await run_with_retry(lambda pid=pid: del_a(pid), f"A:{str(pid)[:8]}")

        async def del_b(pid, tid):
            async with conn.transaction():
                tr = await conn.fetchrow(
                    "SELECT path, coalesce(title,'') AS title FROM file_metadata WHERE id = $1::uuid", tid)
                await conn.execute("DELETE FROM text_embeddings WHERE source_id = $1::uuid", tid)
                await conn.execute("DELETE FROM paragraph WHERE source_id = $1::uuid", tid)
                await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id = $1::uuid", tid)
                await conn.execute("DELETE FROM file_dataset WHERE file_id = $1::uuid", tid)
                await conn.execute("DELETE FROM file_metadata WHERE id = $1::uuid", tid)
                await conn.execute(
                    "UPDATE file_metadata SET title = COALESCE(NULLIF(title,''), $2), path = $3 WHERE id = $1::uuid",
                    pid, tr["title"], tr["path"])

        for pid, tid, why, np_, tn_ in B:
            await run_with_retry(lambda pid=pid, tid=tid: del_b(pid, tid), f"B:{str(pid)[:8]}")
        print(f"[apply] A 删 {len(A)} 条，B 保留并改路径 {len(B)} 条")
    finally:
        await conn.close()

asyncio.run(main())
