"""Add only_files whitelist to the CalibreDataSource stage of a stored task."""
import asyncio, json, sys, common as C

TASK = "导入 OneDrive PDF"
WL = "/app/batch_operation/only_files_lt600_nondict.txt"

async def main():
    conn = await C.get_conn()
    pipe = await conn.fetchval("SELECT pipeline FROM task_dbo WHERE name = $1", TASK)
    if isinstance(pipe, str):
        pipe = json.loads(pipe)
    node = next((n for n in pipe if "CalibreDataSource" in n), None)
    if not node:
        print("CalibreDataSource stage not found"); return 2
    params = node["CalibreDataSource"]
    print("BEFORE:", json.dumps(params, ensure_ascii=False))
    if "--apply" in sys.argv:
        params["only_files"] = WL
        await conn.execute("UPDATE task_dbo SET pipeline = $2::jsonb WHERE name = $1",
                           TASK, json.dumps(pipe, ensure_ascii=False))
        back = await conn.fetchval("SELECT pipeline FROM task_dbo WHERE name = $1", TASK)
        if isinstance(back, str): back = json.loads(back)
        n2 = next(n for n in back if "CalibreDataSource" in n)
        print("AFTER read-back:", json.dumps(n2["CalibreDataSource"], ensure_ascii=False))
    else:
        print("(dry run, pass --apply to write)")
    await conn.close()
    return 0

sys.exit(asyncio.run(main()))
