
import asyncio, os, sys, shutil
sys.path.insert(0, "/app")
import common as C
ST = "/storage"
SRC_DIR = "OneDrive/Phil. Hubbard"
DST_DIR = "OneDrive/Phil Hubbard"
async def go():
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path FROM file_metadata WHERE path LIKE $1 || '/%'",
            "OneDrive/Phil. Hubbard")
        print("rows:", [(str(r["id"]), r["path"]) for r in rows])
        for r in rows:
            old_rel = r["path"]
            new_rel = old_rel.replace("OneDrive/Phil. Hubbard/", "OneDrive/Phil Hubbard/", 1)
            src, dst = os.path.join(ST, old_rel), os.path.join(ST, new_rel)
            book_old = os.path.dirname(src); book_new = os.path.dirname(dst)
            print("bookdir:", book_old, "->", book_new, "src exists:", os.path.isdir(book_old), "dst exists:", os.path.isdir(book_new))
            if os.path.isdir(book_old) and not os.path.exists(book_new):
                shutil.move(book_old, book_new)
            elif os.path.exists(book_new):
                # 目标书目录已存在（同书两行）→ 只在盘上缺文件时搬文件
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
                print("  dst bookdir existed")
            # DB: 若目标 path 已有行 → 冲突处理（此处应无）
            dup = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE path=$1", new_rel)
            if dup:
                print("  DUP at target! skip repoint:", new_rel); continue
            await conn.execute("UPDATE file_metadata SET path=$2 WHERE id=$1::uuid", r["id"], new_rel)
            # 数据集重挂：书库--Phil. Hubbard → 书库--Phil Hubbard
            await conn.execute(
                "INSERT INTO file_dataset (file_id, dataset_id) "
                "SELECT fd.file_id, d2.id FROM file_dataset fd "
                "JOIN dataset d1 ON d1.id=fd.dataset_id AND d1.name='书库--Phil. Hubbard' "
                "JOIN dataset d2 ON d2.name='书库--Phil Hubbard' "
                "WHERE fd.file_id=$1::uuid ON CONFLICT DO NOTHING", r["id"])
            await conn.execute(
                "DELETE FROM file_dataset WHERE file_id=$1::uuid AND dataset_id="
                "(SELECT id FROM dataset WHERE name='书库--Phil. Hubbard')", r["id"])
        # 删空壳数据集
        n = await conn.fetchval(
            "SELECT count(*) FROM file_dataset fd JOIN dataset d ON d.id=fd.dataset_id "
            "WHERE d.name='书库--Phil. Hubbard'")
        print("残留成员:", n)
        if n == 0:
            ref = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE dataset_id="
                "(SELECT id FROM dataset WHERE name='书库--Phil. Hubbard')")
            al = await conn.fetchval("SELECT count(*) FROM dataset WHERE alias_of="
                "(SELECT id FROM dataset WHERE name='书库--Phil. Hubbard')")
            print("引用检查 fmeta:", ref, "alias:", al)
            if not ref and not al:
                await conn.execute("DELETE FROM dataset WHERE name='书库--Phil. Hubbard'")
                print("已删 书库--Phil. Hubbard")
        # 空目录清理
        if os.path.isdir(os.path.join(ST, SRC_DIR)) and not os.listdir(os.path.join(ST, SRC_DIR)):
            os.rmdir(os.path.join(ST, SRC_DIR)); print("删空目录", SRC_DIR)
    finally:
        await conn.close()
asyncio.run(go())
