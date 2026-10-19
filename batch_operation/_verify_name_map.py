
import asyncio, json, sys
sys.path.insert(0, "/app")
import common as C
async def go():
    keys = json.load(open("/app/batch_operation/reports/name_final_map.json"))
    conn = await C.get_conn()
    try:
        n = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE authors && $1::text[]", list(keys))
        print("residual_rows:", n)
    finally:
        await conn.close()
asyncio.run(go())
