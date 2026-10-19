import asyncio, time, common as C

# EXACT watcher SQL (no .pdf filter!) so it is comparable with the log line
WATCHER_SQL = """
    SELECT (SELECT count(*)::bigint FROM file_metadata f
              WHERE f.path LIKE $1
                AND NOT EXISTS (SELECT 1 FROM paragraph p
                                WHERE p.source_id = f.id)) AS files_left,
           (SELECT count(*)::bigint FROM paragraph) AS paras
"""

async def snap(conn, tag):
    t0 = time.time()
    row = await conn.fetchrow(WATCHER_SQL, C.ONEDRIVE_LIKE, timeout=120)
    print(f"[{time.strftime('%H:%M:%S')}] {tag} files_left(watcher-exact)={row['files_left']} "
          f"paras={int(row['paras']):,} query={time.time()-t0:.1f}s", flush=True)
    return row["files_left"], row["paras"]

async def m():
    conn = await C.get_conn()
    a = await snap(conn, "A")
    await asyncio.sleep(75)
    b = await snap(conn, "B")
    print(f"DELTA 75s: files_done={a[0]-b[0]} paras_added={int(b[1]-a[1]):,} "
          f"({(b[1]-a[1])/75:.1f} rows/s)", flush=True)
    await conn.close()

asyncio.run(m())
