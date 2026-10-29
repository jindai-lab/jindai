"""人民日报按日期合并（2026-10-09 用户指令）。

把 item_type='newspaperArticle'、path 形如 `1983-04-24.第5版() 今日谈`（根级无目录）
的行按日期合并成一天一行：

  path        → https://virtual/renmin-ribao/YYYY-MM-DD
  title       → 人民日报 YYYY-MM-DD      （原本全空；不填会被 fill_nontitle 回填成 URL）
  publication → 人民日报
  date        → YYYY-MM-DD
  extdata     → [{"path": 原路径, "page": 第N版, "column": 栏目, "title": 标题}, ...]

段落搬运走「备份→删→原 id 重插」配方（paragraph 按 source_id 分区，**不能 UPDATE source_id**）：
- 重插时 outline / pagenum / paragraph.extdata 原样保留——版面/栏目/标题本来就在段落里
  （pagenum=`第6版(国际)`、outline=标题、extdata.column=栏目），合并后仍能按段落还原到文章；
- 队列由触发器自动处理：DELETE 触发器清旧行、INSERT 触发器按新 source_id 重建；
- text_embeddings.source_id=文件 id，本批 0 条嵌入，无需迁移；
- file_dataset 全在同一个数据集（报刊--人民日报），目标行本就是成员 → 源行成员关系直接删。

用法:
  merge_rmrbb_by_date.py               # dry-run：建/刷新 rm_map 并报计划，不动正文
  merge_rmrbb_by_date.py --apply       # 按日期分批合并（可中断续跑）
  merge_rmrbb_by_date.py --apply --limit 5   # 只合 5 个日期（调试）
  merge_rmrbb_by_date.py --verify      # 合并后核对读回
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

import common as C

log = C.setup_logging("rmrb_merge", "batch_operation/reports/rmrb_merge.log")
OUT = "batch_operation/reports/rmrb_merge.json"
VIRTUAL = "https://virtual/renmin-ribao/"

# ---- 解析（SQL 侧做，344k 行不来回搬） -------------------------------------
RE_DATE = r"^\d{4}-\d{1,2}-\d{1,2}\."
BUILD_MAP = """
CREATE TABLE IF NOT EXISTS rm_map (
    source_id uuid PRIMARY KEY,
    target_id uuid NOT NULL,
    src_path  text NOT NULL,
    art_date  text NOT NULL,
    page      text NOT NULL DEFAULT '',
    column_   text NOT NULL DEFAULT '',
    title_    text NOT NULL DEFAULT ''
);
TRUNCATE rm_map;
INSERT INTO rm_map (source_id, target_id, src_path, art_date, page, column_, title_)
SELECT f.id, tgt.id, f.path, f.d,
       coalesce(s.page_, ''), coalesce(s.col_, ''), coalesce(s.ttl_, '')
FROM (
    SELECT f2.id, f2.path,
           to_char((substring(f2.path from '^(\d{4}-\d{1,2}-\d{1,2})'))::date, 'YYYY-MM-DD') AS d,
           substring(f2.path from '^\d{4}-\d{1,2}-\d{1,2}\.(.*)$') AS rest
    FROM file_metadata f2
    WHERE f2.item_type = 'newspaperArticle' AND f2.path ~ '^\d{4}-\d{1,2}-\d{1,2}\.'
) f
JOIN (SELECT DISTINCT ON (d) d, id
      FROM (SELECT to_char((substring(path from '^(\d{4}-\d{1,2}-\d{1,2})'))::date, 'YYYY-MM-DD') AS d, id
            FROM file_metadata
            WHERE item_type = 'newspaperArticle' AND path ~ '^\d{4}-\d{1,2}-\d{1,2}\.') x
      ORDER BY d, id) tgt ON tgt.d = f.d
CROSS JOIN LATERAL (
    SELECT coalesce(substring(f.rest from '^(第[0-9一二三四五六七八九十百]+版)'), '') AS page_,
           coalesce(substring(f.rest from '^第[0-9一二三四五六七八九十百]+版\(([^)]*)\)'), '') AS col_,
           coalesce(btrim(substring(f.rest from
                 '^第[0-9一二三四五六七八九十百]+版(?:\([^)]*\))?\s*(.*)$')), btrim(f.rest)) AS ttl_
) s;
ANALYZE rm_map;
"""

SQL_COUNTS = """
SELECT (SELECT count(*) FROM file_metadata WHERE item_type='newspaperArticle'),
       (SELECT count(*) FROM file_metadata WHERE item_type='newspaperArticle' AND path ~ '^\\d{4}-\\d{2}-\\d{2}'),
       (SELECT count(DISTINCT art_date) FROM rm_map),
       (SELECT count(*) FROM rm_map),
       (SELECT count(*) FROM rm_map m JOIN file_metadata t ON t.id=m.target_id WHERE t.path LIKE 'https://virtual/%')
"""

SQL_UNPARSED = """
SELECT path FROM file_metadata
WHERE item_type='newspaperArticle' AND path !~ '^\\d{4}-\\d{1,2}-\\d{1,2}\\.' LIMIT 10
"""

# 待处理目标（还没变成虚拟网址的）——天然的续跑条件
SQL_TODO_TARGETS = """
SELECT DISTINCT m.target_id::text
FROM rm_map m JOIN file_metadata t ON t.id = m.target_id
WHERE NOT (t.path LIKE 'https://virtual/%')
ORDER BY 1
"""

# 一批日期里要动的段落：备份
SQL_PB = """
CREATE TEMP TABLE pb ON COMMIT DROP AS
SELECT p.id, p.source_page, p.author, p.pdate, p.outline, p.content, p.pagenum,
       p.lang, p.extdata, p.keywords, m.target_id
FROM paragraph p JOIN rm_map m ON m.source_id = p.source_id
WHERE m.target_id = ANY($1::uuid[]);
"""
SQL_PB_DEL = """
DELETE FROM paragraph p USING rm_map m
WHERE m.source_id = p.source_id AND m.target_id = ANY($1::uuid[])
"""
SQL_PB_INS = """
INSERT INTO paragraph (id, source_page, author, pdate, outline, content, pagenum,
                       lang, extdata, keywords, source_id)
SELECT id, source_page, author, pdate, outline, content, pagenum,
       lang, extdata, keywords, target_id
FROM pb
"""
SQL_QUEUE_INS = """
INSERT INTO embedding_pending_queue (id, source_id, created_at)
SELECT id, target_id, now() FROM pb
ON CONFLICT (id, source_id) DO NOTHING
"""
SQL_DS_DEL = "DELETE FROM file_dataset fd USING rm_map m WHERE fd.file_id = m.source_id AND m.target_id = ANY($1::uuid[])"
SQL_FM_DEL = """
DELETE FROM file_metadata f USING rm_map m
WHERE f.id = m.source_id AND m.target_id = ANY($1::uuid[])
  AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
"""
SQL_TGT_UPD = """
UPDATE file_metadata t
SET path = $2 || m.art_date,
    title = '人民日报 ' || m.art_date,
    publication = '人民日报',
    date = (m.art_date || ' 00:00:00')::timestamp,
    extdata = sub.extdata
FROM (SELECT target_id,
             jsonb_agg(jsonb_build_object('path', src_path, 'page', page,
                                           'column', column_, 'title', title_)
                       ORDER BY src_path) AS extdata
      FROM rm_map WHERE target_id = ANY($1::uuid[]) GROUP BY target_id) sub
JOIN (SELECT DISTINCT target_id, art_date FROM rm_map WHERE target_id = ANY($1::uuid[])) m
  ON m.target_id = sub.target_id
WHERE t.id = sub.target_id
"""

VERIFY = """
SELECT (SELECT count(*) FROM file_metadata WHERE item_type='newspaperArticle'
         AND path !~ '^https://virtual/'),
       (SELECT count(*) FROM file_metadata WHERE path LIKE 'https://virtual/renmin-ribao/%'),
       (SELECT count(*) FROM file_metadata WHERE path LIKE 'https://virtual/%'
         AND (coalesce(title,'')='' OR date IS NULL OR coalesce(publication,'')='')),
       (SELECT count(DISTINCT m.target_id) FROM rm_map m JOIN file_metadata t ON t.id=m.target_id
         WHERE NOT (t.path LIKE 'https://virtual/%')),
       (SELECT count(*) FROM paragraph p JOIN file_metadata f ON f.id=p.source_id
         WHERE f.item_type='newspaperArticle' AND f.path !~ '^https://virtual/'),
       (SELECT count(*) FROM paragraph p JOIN file_metadata f ON f.id=p.source_id
         WHERE f.item_type='newspaperArticle'),
       (SELECT count(*) FROM file_metadata f JOIN rm_map m ON m.source_id=f.id)
"""


async def build_map(conn):
    async with conn.transaction():
        await conn.execute(BUILD_MAP)
    n, d, dates, mapped, done = (await conn.fetchrow(SQL_COUNTS))
    log.info("[map] rows=%d 可解析=%d 日期数=%d 映射=%d 已完成目标=%d", n, d, dates, mapped, done)
    unparsed = await conn.fetch(SQL_UNPARSED)
    if unparsed:
        log.warning("[map] 非日期开头、跳过 %d+ 行，样本: %s", len(unparsed),
                    [u["path"] for u in unparsed])
    return {"rows": n, "parseable": d, "dates": dates, "mapped": mapped,
            "targets_done": done, "unparsed_sample": [u["path"] for u in unparsed]}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只处理 N 个日期")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    conn = await C.get_conn()
    # 关并行（本机并行哈希会撞 shared memory）+ 提高 work_mem
    await conn.execute("SET max_parallel_workers_per_gather=0")
    await conn.execute("SET work_mem='256MB'")
    await conn.execute("SET jit=off")

    if args.verify:
        row = await conn.fetchrow(VERIFY)
        keys = ["未合并的人民日报行(应0)", "虚拟网址行(应20913)", "字段缺失(应0)",
                "待合并日期(应0)", "仍挂在未合并行的段落(应0)",
                "人民日报段落总数(应1374841)", "仍存在的源行(应0)"]
        res = {k: int(v) for k, v in zip(keys, row)}
        log.info("[verify] %s", json.dumps(res, ensure_ascii=False))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        await conn.close()
        return 0

    plan = await build_map(conn)

    if not args.apply:
        # 日期分布统计
        row = await conn.fetchrow(
            "SELECT count(*), count(DISTINCT art_date), coalesce(min(art_date),''), "
            "coalesce(max(art_date),''), "
            "(SELECT count(*) FROM rm_map WHERE page=''), (SELECT count(*) FROM rm_map WHERE title_='') "
            "FROM rm_map")
        plan.update({"mapped_rows": int(row[0]), "distinct_dates": int(row[1]),
                     "first_date": row[2], "last_date": row[3],
                     "no_page": int(row[4]), "no_title": int(row[5])})
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=1)
        log.info("[dry-run] %s", json.dumps(plan, ensure_ascii=False))
        print(json.dumps(plan, ensure_ascii=False, indent=1))
        await conn.close()
        return 0

    # ---------------- apply ----------------
    # paragraph 32 个哈希分区共 ~110GB：哈希连接会整表扫，循环里必须逼成索引嵌套循环。
    # 只能在 build_map 之后开（map 重建是等值连接+排序，关哈希会退化成逐行全扫）。
    await conn.execute("SET enable_hashjoin=off")
    await conn.execute("SET enable_mergejoin=off")
    todo = [r["target_id"] for r in await conn.fetch(SQL_TODO_TARGETS)]
    if args.limit:
        todo = todo[: args.limit]
    log.info("[apply] 待合并日期 %d", len(todo))
    BATCH = 200
    done_dates = 0
    moved_paras = 0
    deleted_rows = 0
    try:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            t0 = time.monotonic()
            async with conn.transaction():
                await conn.execute(SQL_PB, batch)
                t_pb = time.monotonic()
                para_n = int(await conn.fetchval("SELECT count(*) FROM pb") or 0)
                tag_del = await conn.execute(SQL_PB_DEL, batch)   # 触发器按 (id, 旧source) 清队列
                t_del = time.monotonic()
                # 逐行 plpgsql 触发器建队列太慢（实测 16.7k 行 187s）：关触发器重插，
                # 再用一条批量 INSERT 从 pb 重建队列（新 (id, target) 对）。
                await conn.execute("SET session_replication_role = replica")
                await conn.execute(SQL_PB_INS)
                await conn.execute(SQL_QUEUE_INS)
                t_ins = time.monotonic()
                await conn.execute(SQL_DS_DEL, batch)
                # file_metadata 删除的 RI 检查要探 text_embeddings 的 32 个分区（每行 ~50ms）；
                # 已核验这些源行在 text_embeddings / page_skip_state 里 0 行、file_dataset 本批已先删。
                fm_n = await conn.execute(SQL_FM_DEL, batch)
                await conn.execute(SQL_TGT_UPD, batch, VIRTUAL)
                await conn.execute("SET session_replication_role = origin")   # 事务内 SET，回滚也自动复原
                t_fm = time.monotonic()
            timing = (t_pb - t0, t_del - t_pb, t_ins - t_del, t_fm - t_ins,
                      time.monotonic() - t_fm)
            moved_paras += para_n
            deleted_rows += int(fm_n.split()[-1]) if fm_n and fm_n.split()[-1].isdigit() else 0
            done_dates += len(batch)
            left = await conn.fetchval(
                "SELECT count(DISTINCT target_id) FROM rm_map m JOIN file_metadata t "
                "ON t.id=m.target_id WHERE NOT (t.path LIKE 'https://virtual/%')")
            log.info("[progress] 日期 %d/%d；本批段落 %d、删源行 %s；累计搬 %d 删 %d；剩余目标 %s；"
                     "耗时 pb备份 %.1fs/删 %.1fs/插 %.1fs/文件 %.1fs/更新 %.1fs",
                     done_dates, len(todo), para_n, fm_n.split()[-1] if fm_n else '?',
                     moved_paras, deleted_rows, left, *timing)
    except Exception:
        log.exception("[apply] 中断（事务已回滚本批）；重跑同命令可续")
        raise

    plan.update({"applied": True, "dates_done": done_dates, "moved_paras": moved_paras,
                  "deleted_source_rows": deleted_rows})
    if not args.limit:          # --limit 是调试跑，跳过 4 分钟的核对查询
        row = await conn.fetchrow(VERIFY)
        keys = ["unmerged_rows", "virtual_rows", "missing_fields", "targets_pending",
                "paras_on_unmerged", "rmrb_paras_total", "sources_left"]
        plan["verify"] = {k: int(v) for k, v in zip(keys, row)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    log.info("[done] %s", json.dumps(plan, ensure_ascii=False))
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    await conn.close()
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
