"""file_import_state 计数（queue.sh 看板用）。输出一行：complete=N partial=M。"""
import asyncio
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")

import common as C  # noqa: E402


async def main() -> None:
    conn = await C.get_conn()
    try:
        r = await conn.fetchrow(
            "SELECT count(*) FILTER (WHERE status='complete') c, "
            "       count(*) FILTER (WHERE status='partial')  p "
            "FROM file_import_state")
        print(f"complete={r['c']} partial={r['p']}")
    finally:
        await conn.close()


asyncio.run(main())
