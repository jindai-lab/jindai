"""删除指向不存在文件的 file_metadata 记录（含子表 paragraph / 向量 / 待办队列 / 数据集关联 / 跳页状态）

--apply 才写库；先备份整行、并复验文件仍不存在。
"""
import argparse, asyncio, csv, datetime, os, sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

STORE = "/storage"
PATHS = [
    "Zotero/毛 and 王_2023_应用元宇宙促进城市空间规划变革研究.pdf",
    "Zotero/郭 and 申_2022_国家安全视阈下语言问题新表征语言恐怖主义的缘起、识别与应对.pdf",
    "Zotero/朱 and 风_2019_国家社科基金社会学类项目研究趋势与成果评价分析(1991—2016).pdf",
    "Zotero/复 and 周_2015_空间可沟通性“微游上海”的城市意象及其生产.pdf",
    "Zotero/王 et al_2023_消解与重构：数字媒介时代青年亚文化的“去污名化”分析——以二次元为例.pdf",
    "Zotero/王_2021_组织因素对重大工程项目绩效影响研究.pdf",
    "Zotero/赵_陈_2023_网络社会视角下的城市意象理论辨析与规划应用.pdf",
    "Zotero/黎_2023_二次元文化、数字交往与民族共同体意识的建构.pdf",
]
CHILD = [("paragraph", "source_id"), ("embedding_pending_queue", "source_id"),
         ("page_skip_state", "source_id"), ("file_dataset", "file_id")]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT * FROM file_metadata WHERE path = ANY($1::text[])", PATHS)
        print(f"命中 {len(rows)}/{len(PATHS)} 行")
        miss = [p for p in PATHS if p not in {r['path'] for r in rows}]
        for m in miss:
            print("  未命中:", m[:70])
        ids = [r["id"] for r in rows]
        # 复验：文件必须仍不存在，否则不删
        bad = [r["path"] for r in rows if os.path.exists(os.path.join(STORE, r["path"]))]
        if bad:
            print("!! 以下文件其实存在，拒绝删除:", bad)
            return
        # 子表计数
        for t, col in CHILD:
            if t == "paragraph":
                n = await conn.fetchval(
                    f"SELECT count(*) FROM paragraph WHERE {col} = ANY($1::uuid[])", ids)
            else:
                n = await conn.fetchval(
                    f"SELECT count(*) FROM {t} WHERE {col} = ANY($1::uuid[])", ids)
            print(f"  子表 {t}: {n} 行")
        emb = await conn.fetchval(
            "SELECT count(*) FROM text_embeddings WHERE source_id = ANY($1::uuid[])", ids)
        print(f"  子表 text_embeddings: {emb} 行（ON DELETE CASCADE）")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"batch_operation/reports/file_metadata_deleted_{stamp}.tsv"
        cols = list(rows[0].keys()) if rows else []
        with open(bak, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(cols)
            for r in rows:
                w.writerow([str(r[c]) for c in cols])
        print(f"  备份整行 → {bak}")
        if not a.apply:
            print("[dry-run] 未删除")
            return
        async with conn.transaction():
            for t, col in CHILD:
                await conn.execute(
                    f"DELETE FROM {t} WHERE {col} = ANY($1::uuid[])", ids)
            await conn.execute(
                "DELETE FROM text_embeddings WHERE source_id = ANY($1::uuid[])", ids)
            n = await conn.execute(
                "DELETE FROM file_metadata WHERE id = ANY($1::uuid[])", ids)
        print(f"[apply] {n}")
    finally:
        await conn.close()

asyncio.run(main())
