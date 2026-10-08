"""把 file_metadata 里 `X.epub#zip/<部件>` 形态的行合并成真实 epub 文件行。

2026-10-08 用户指令：epub#zip 格式的路径合并为实际的 epub 文件路径，
zip/ 之后的部件路径转移到 paragraph.outline 字段。

做法（每个 epub 一个事务，dry-run 默认，--apply 落库）：
  1. 目标行 = `#zip/` 前的真实 epub 路径；已有裸 epub 行则复用，否则从首个部件行
     复制书目字段新建（extension='epub'）。
  2. 部件行的段落**备份→删除→原 id 重插**到目标行：paragraph 按 source_id 分区、
     且受 fk_text_embeddings_id（ON UPDATE NO ACTION）限制，`UPDATE source_id`
     会直接报错，只能搬行；outline 置为 zip/ 后的部件路径。
  3. 嵌入同样备份→重插（同 id、source 改指目标行），**不触发重嵌入**；
     重插段落的 insert 触发器产生的队列行按 id 清掉（嵌入已在，无需重算）。
  4. 段落清空后的部件行：迁 file_dataset、清残留队列行、删行（守卫：仍挂段落即放弃）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import common as C

log = C.setup_logging("merge_epub_zip", "batch_operation/reports/merge_epub_zip.log")

GROUP_SQL = """
    SELECT split_part(path, '#zip/', 1) AS epub, count(*) AS comps,
           sum((SELECT count(*) FROM paragraph p WHERE p.source_id = f.id)) AS paras
    FROM file_metadata f
    WHERE path LIKE '%#zip%'
    GROUP BY 1 ORDER BY 1
"""
COMP_SQL = "SELECT id, path FROM file_metadata WHERE path LIKE $1 || '#zip/%' ORDER BY path"

TARGET_COLS = ("dataset_id, item_type, title, authors, date, language, isbn, "
               "publisher, tags, size_bytes")
PARA_COLS = ("id, source_page, author, pdate, outline, content, pagenum, lang, "
             "extdata, keywords, source_id")


async def ensure_target(conn, epub: str, first_comp):
    row = await conn.fetchval("SELECT id FROM file_metadata WHERE path = $1", epub)
    if row:
        return row, False
    low = epub.lower()
    ext = "epub" if low.endswith(".epub") else ("zip" if low.endswith(".zip") else "")
    await conn.execute(
        f"INSERT INTO file_metadata (path, extension, {TARGET_COLS}) "
        f"SELECT $1, $3, {TARGET_COLS} FROM file_metadata WHERE id = $2 "
        "ON CONFLICT (path) DO NOTHING",
        epub, first_comp, ext)
    return await conn.fetchval("SELECT id FROM file_metadata WHERE path = $1", epub), True


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--epub", help="只处理指定前缀的 epub")
    args = ap.parse_args()

    from jindai.storage import storage

    conn = await C.get_conn()
    groups = await conn.fetch(GROUP_SQL)
    if args.epub:
        groups = [g for g in groups if g["epub"] == args.epub]
    log.info("[groups] %d epub group(s), %d component(s), %d paragraph(s)",
             len(groups), sum(g["comps"] for g in groups),
             sum(g["paras"] or 0 for g in groups))

    plan = []
    for g in groups:
        epub = g["epub"]
        n_comp = g["comps"]
        para = int(g["paras"] or 0)
        tgt = await conn.fetchval("SELECT id FROM file_metadata WHERE path = $1", epub)
        exists = os.path.exists(storage.safe_join(epub))
        plan.append({"epub": epub, "comps": n_comp, "paras": para,
                     "target": str(tgt) if tgt else None, "disk": exists})
        log.info("  %s | comps=%d paras=%d target=%s disk=%s",
                 epub[-90:], n_comp, para, "yes" if tgt else "CREATE", exists)

    if not args.apply:
        log.info("dry-run only; re-run with --apply to write")
        await conn.close()
        return 0

    await conn.execute("CREATE TEMP TABLE IF NOT EXISTS para_bak AS "
                       "SELECT * FROM paragraph WHERE false")
    await conn.execute("CREATE TEMP TABLE IF NOT EXISTS emb_bak AS "
                       "SELECT id, source_id, chunk_id, embedding "
                       "FROM text_embeddings WHERE false")

    tot_para = tot_comp = tot_tgt_new = 0
    for g in groups:
        epub = g["epub"]
        comps = await conn.fetch(COMP_SQL, epub)
        try:
            async with conn.transaction():
                target, created = await ensure_target(conn, epub, comps[0]["id"])
                if not target:
                    raise RuntimeError("no target row")
                tot_tgt_new += 1 if created else 0
                n_para = n_comp = 0
                for comp in comps:
                    zipsub = comp["path"].split("#zip/", 1)[1]
                    await conn.execute(
                        "INSERT INTO para_bak " + PARA_COLS + " "
                        "SELECT id, source_page, author, pdate, $3, content, pagenum, "
                        "       lang, extdata, keywords, source_id "
                        "FROM paragraph WHERE source_id = $1 AND $2 = $2",
                        comp["id"], target, zipsub)
                    n_para += await conn.fetchval(
                        "SELECT count(*) FROM para_bak")
                    await conn.execute(
                        "INSERT INTO emb_bak SELECT id, source_id, chunk_id, embedding "
                        "FROM text_embeddings WHERE source_id = $1", comp["id"])
                    await conn.execute(
                        "DELETE FROM paragraph WHERE source_id = $1", comp["id"])
                    await conn.execute(
                        "INSERT INTO paragraph " + PARA_COLS + " "
                        "SELECT id, source_page, author, pdate, outline, content, "
                        "       pagenum, lang, extdata, keywords, $2 FROM para_bak",
                        comp["id"], target)
                    await conn.execute(
                        "INSERT INTO text_embeddings " 
                        "SELECT id, $2, chunk_id, embedding FROM emb_bak "
                        "ON CONFLICT DO NOTHING", comp["id"], target)
                    # 清掉重插段落自动入队的行（嵌入已随迁，无需重算）
                    await conn.execute(
                        "DELETE FROM embedding_pending_queue "
                        "WHERE id IN (SELECT id FROM para_bak)")
                    await conn.execute("TRUNCATE para_bak, emb_bak")
                    # 部件行收尾
                    await conn.execute(
                        "INSERT INTO file_dataset (file_id, dataset_id) "
                        "SELECT $2, fd.dataset_id FROM file_dataset fd "
                        "WHERE fd.file_id = $1 ON CONFLICT DO NOTHING",
                        comp["id"], target)
                    await conn.execute("DELETE FROM file_dataset WHERE file_id = $1",
                                       comp["id"])
                    await conn.execute(
                        "DELETE FROM embedding_pending_queue WHERE source_id = $1",
                        comp["id"])
                    tag = await conn.execute(
                        "DELETE FROM file_metadata WHERE id = $1 "
                        "AND NOT EXISTS (SELECT 1 FROM paragraph "
                        "                 WHERE source_id = $1)", comp["id"])
                    if int(tag.split()[-1]):
                        n_comp += 1
                tot_para += n_para
                tot_comp += n_comp
            log.info("[merge] %s comps=%d paras=%d -> %s",
                     epub[-70:], n_comp, n_para, str(target)[:8])
        except Exception as ex:
            log.warning("[merge] FAILED %s: %s", epub[-70:], str(ex)[:300])

    summary = {"groups": len(groups), "components_deleted": tot_comp,
               "paragraphs_moved": tot_para, "targets_created": tot_tgt_new}
    os.makedirs("batch_operation/reports", exist_ok=True)
    with open("batch_operation/reports/merge_epub_zip.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "plan": plan}, f, ensure_ascii=False, indent=1)
    log.info("[done] %s", json.dumps(summary, ensure_ascii=False))
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
