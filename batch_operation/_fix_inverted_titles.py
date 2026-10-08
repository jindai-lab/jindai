
"""修正 `作者 - 书名` 型文件名被反解的行：authors[1] 是书名（>60 字符的“作者”）→ 互换。幂等。"""
import asyncio, sys
import common as C
log = C.setup_logging("fix_inverted_titles", "batch_operation/reports/fix_inverted_titles.log")

async def main():
    conn = await C.get_conn()
    rows = await conn.fetch("""
        SELECT path, title, authors[1] a FROM file_metadata
        WHERE lower(path) LIKE '%.pdf' AND cardinality(authors)=1
          AND length(authors[1]) > 60 AND coalesce(btrim(title),'') <> ''""")
    log.info("[plan] %d inverted rows", len(rows))
    n = 0
    for r in rows:
        async with conn.transaction():
            tag = await conn.execute("""
                UPDATE file_metadata SET title=$2, authors=$3::text[]
                WHERE path=$1 AND cardinality(authors)=1 AND length(authors[1])>60""",
                r["path"], r["a"], [r["title"]])
            n += int(tag.split()[-1])
    left = await conn.fetchval("""
        SELECT count(*) FROM file_metadata WHERE lower(path) LIKE '%.pdf'
          AND cardinality(authors)=1 AND length(authors[1])>60""")
    log.info("[done] swapped=%d remaining=%d", n, left)
    for r in rows[:5]:
        log.info("[sample] %s | %s", (r["a"] or "")[:70], (r["title"] or "")[:50])
    await conn.close()
asyncio.run(main())
