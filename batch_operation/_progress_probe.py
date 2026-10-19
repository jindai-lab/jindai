import asyncio, time, common as C

Q_MISS = """
    SELECT count(*) FROM file_metadata f
    WHERE f.path LIKE $1 AND lower(f.path) LIKE '%.pdf'
      AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
"""

async def snap(conn):
    tup = await conn.fetchval("SELECT tup_inserted FROM pg_stat_database WHERE datname = current_database()")
    ins = await conn.fetchval("SELECT n_tup_ins FROM pg_stat_user_tables WHERE relname = 'paragraph'")
    live = await conn.fetchval("SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = 'paragraph'")
    miss = await conn.fetchval(Q_MISS, C.ONEDRIVE_LIKE)          # exact watcher query
    miss2 = await conn.fetchval(Q_MISS, "%OneDrive/%")            # broader pattern
    return tup, ins, live, miss, miss2

async def m():
    conn = await C.get_conn()
    print("ONEDRIVE_LIKE =", C.ONEDRIVE_LIKE, flush=True)
    a = await snap(conn)
    print(f"[{time.strftime('%H:%M:%S')}] db_tup_inserted={a[0]:,} para_n_tup_ins={a[1]:,} "
          f"para_n_live_tup~{a[2]:,} miss(watcher)= {a[3]} miss(broad)={a[4]}", flush=True)
    await asyncio.sleep(90)
    b = await snap(conn)
    print(f"[{time.strftime('%H:%M:%S')}] db_tup_inserted={b[0]:,} para_n_tup_ins={b[1]:,} "
          f"para_n_live_tup~{b[2]:,} miss(watcher)= {b[3]} miss(broad)={b[4]}", flush=True)
    print(f"DELTA 90s: db_tuples_inserted={b[0]-a[0]:,} paragraph_rows_inserted={b[1]-a[1]:,} "
          f"files_done={a[3]-b[3]} (watcher), {a[4]-b[4]} (broad)", flush=True)
    await conn.close()

asyncio.run(m())
