"""file_metadata 清理 + 新增 import_state 列（2026-10-20 用户指令）。

两件事：
  ① `dataset_id` 已死（2026-10-10 起成员只走 file_dataset；941 个非空值全指向
     「书库--未知」，且这些行都有正常 file_dataset 绑定）→ 全部置 NULL
     （列保留：models.py 仍映射它，删列要动 ORM + 重建镜像，另议）。
  ② 新增 `file_metadata.import_state text`，把「导入状态」冗余成一列
     （语义 = file_import_state 盖章 + 段落存在性 + 盘上存在性的合成）：

       complete   file_import_state 盖章：页集全覆盖
       partial    盖章：有段落但页集未全覆盖（下轮重试）
       imported   有段落但无盖章（2026-10-16 盖章机制上线前的旧导入）
       pending    无段落、无完整章、文件在盘（= 导入队列真正要消化的量）
       missing    无段落、文件不在盘（以前混在 pending 里，是 pending 的地板来源）
       excluded   Tool Books/（站规不导入）

     只对 PDF 行取值，其余（webpage/newspaper 等）留 NULL。

维护：record_import_state（import_onedrive_pdfs.py）盖章时同步 UPDATE 本列；
     磁盘扫描 upsert 新登记的行写 pending。本脚本是全量重算（幂等），
     磁盘侧变化大时重跑一次即可。Dry-run 默认；--apply 写入。
"""
import asyncio, json, os, sys, logging, argparse, time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
logging.disable(logging.CRITICAL)
import common as C

REPORTS = "batch_operation/reports"
INVENTORY = REPORTS + "/disk_inventory.json"
CHUNK = 5000


def load_disk_paths():
    d = json.load(open(INVENTORY))
    age_h = (time.time() - d["ts"]) / 3600
    if age_h > 6:
        print("WARN: disk_inventory %.1f h 旧（>6h），missing 判定可能偏松" % age_h)
    return set(d["sizes"].keys()), age_h


async def main(apply=False):
    conn = await C.get_conn()

    # ---- ① dataset_id 死值清理（守卫：非空行必须都有 file_dataset 绑定）----
    n_null = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE dataset_id IS NOT NULL")
    orphan = await conn.fetchval("""SELECT count(*) FROM file_metadata fm
        WHERE fm.dataset_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM file_dataset fd WHERE fd.file_id = fm.id)""")
    print("dataset_id 非空: %d | 其中无 file_dataset 绑定(孤儿): %d" % (n_null, orphan))
    if orphan:
        print("ABORT: 有孤儿行，先人工看")
        return

    # ---- ② 建列 + 索引 ----
    if apply:
        bak = open(REPORTS + "/filemeta_dataset_id_backup.tsv", "w")
        for r in await conn.fetch(
                "SELECT id, path, dataset_id::text AS d FROM file_metadata WHERE dataset_id IS NOT NULL"):
            bak.write("%s\t%s\t%s\n" % (r["id"], r["path"], r["d"]))
        bak.close()
        tag = await conn.execute("UPDATE file_metadata SET dataset_id = NULL WHERE dataset_id IS NOT NULL")
        print("dataset_id 清空:", tag)
        await conn.execute("ALTER TABLE file_metadata ADD COLUMN IF NOT EXISTS import_state text")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_metadata_import_state "
            "ON file_metadata (import_state) WHERE import_state IS NOT NULL")
        print("列/索引就绪")
    else:
        has_col = await conn.fetchval(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name='file_metadata' AND column_name='import_state'")
        print("import_state 列已存在" if has_col else "import_state 列未建（--apply 时建）")

    # ---- ③ 全量重算 ----
    on_disk, age_h = load_disk_paths()
    print("disk paths: %d (%.1fh 旧)" % (len(on_disk), age_h))

    counts, changed = {}, 0
    cursor = None
    est_total = await conn.fetchval(
        "SELECT count(*) FROM file_metadata WHERE lower(path) LIKE '%.pdf'")
    done = 0
    t_start = time.time()
    while True:
        if cursor:
            rows = await conn.fetch(
                "SELECT id, path FROM file_metadata "
                "WHERE lower(path) LIKE '%.pdf' AND id > $1::uuid ORDER BY id LIMIT " + str(CHUNK), cursor)
        else:
            rows = await conn.fetch(
                "SELECT id, path FROM file_metadata "
                "WHERE lower(path) LIKE '%.pdf' ORDER BY id LIMIT " + str(CHUNK))
        if not rows:
            break
        cursor = rows[-1]["id"]
        ids = [r["id"] for r in rows]
        done += len(ids)
        print("  [%d/%d] %.1fs %s…" % (done, est_total, time.time() - t_start,
                                        rows[0]["path"][:46]), flush=True)

        stamps = {r["source_id"]: r["status"] for r in await conn.fetch(
            "SELECT source_id, status FROM file_import_state WHERE source_id = ANY($1::uuid[])", ids)}
        with_para = {r["source_id"] for r in await conn.fetch(
            "SELECT source_id FROM paragraph WHERE source_id = ANY($1::uuid[]) GROUP BY source_id", ids)}

        states = []
        for r in rows:
            p = r["path"]
            if p.startswith("Tool Books/"):
                st = "excluded"
            elif stamps.get(r["id"]) == "complete":
                st = "complete"
            elif stamps.get(r["id"]) == "partial":
                st = "partial"
            elif r["id"] in with_para:
                st = "imported"
            elif p not in on_disk:
                st = "missing"
            else:
                st = "pending"
            states.append(st)
            counts[st] = counts.get(st, 0) + 1

        if apply:
            n = await conn.execute("""
                UPDATE file_metadata fm SET import_state = v.st
                FROM unnest($1::uuid[], $2::text[]) AS v(id, st)
                WHERE fm.id = v.id AND fm.import_state IS DISTINCT FROM v.st""",
                ids, states)
            changed += int(n.split()[-1])

    print("PDF 行分类:", json.dumps(counts, ensure_ascii=False, sort_keys=True))
    if apply:
        print("实际 UPDATE 行数:", changed)
        for st in ("pending", "missing", "complete", "partial"):
            n = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE import_state=$1", st)
            print("  读回 %s = %d" % (st, n))
    else:
        print("DRY-RUN (no --apply)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(apply=a.apply))
