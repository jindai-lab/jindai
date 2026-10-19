
import asyncio, json, sys
sys.path.insert(0, "/app")
import common as C
async def go():
    m = json.load(open("/app/batch_operation/reports/name_final_map.json"))
    conn = await C.get_conn()
    try:
        n = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE authors && $1::text[]", list(m))
        print("② 映射残留行:", n)
        g = await conn.fetchval("""
          WITH t AS (SELECT regexp_replace(regexp_replace(lower(d.name), '^书库--', ''), '[^0-9a-zÀ-ÿ\u0400-\u04FF\u4e00-\u9fff]+', '', 'g') AS k
                     FROM dataset d WHERE d.name LIKE '书库--%%')
          SELECT count(*) FROM (SELECT k FROM t GROUP BY k HAVING count(*)>1) x""")
        print("③ 书库数据集异写组:", g)
        tot = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE authors IS NOT NULL")
        cjk = await conn.fetchval("""
          SELECT count(*) FROM file_metadata, LATERAL unnest(authors) a
          WHERE authors IS NOT NULL AND a ~ '^[\u4e00-\u9fff·、，]+$'
            AND a NOT IN ('未知','佚名')""")
        print("④ 总行:", tot, " 纯CJK元素(含日韩/华裔/中国作者，预期保留):", cjk)
    finally:
        await conn.close()
asyncio.run(go())
