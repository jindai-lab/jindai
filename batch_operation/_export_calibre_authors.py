"""宿主侧导出 Calibre 著者表 → reports/calibre_authors.json（author_evidence_scan 的证据 A）。

CIFS 上的 SQLite 在容器内会 `locking protocol`/PermissionError，宿主读正常。
只读打开，绝不写 metadata.db。
"""
import json
import os
import shutil
import sqlite3
import tempfile

DB = "/misc/Public/OneDrive/Books/metadata.db"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "reports/calibre_authors.json")

# CIFS 上带 WAL 的 SQLite 直接开也会 `locking protocol` —— 拷到本地盘再读（只读用途）
workdir = tempfile.mkdtemp(prefix="calibre_ro_", dir=os.environ.get("TMPDIR"))
local = os.path.join(workdir, "metadata.db")
# 只拷主体：copy2 的 copystat（chmod/utime）在 CIFS 上会抛 PermissionError，
# 而 -wal 是 0 字节、-shm 不需要 —— 拷主体即可得到一致的只读快照
shutil.copyfile(DB, local)

con = sqlite3.connect(local, timeout=60)
out: dict[str, list[str]] = {}
for bid, name in con.execute(
        "SELECT l.book, a.name FROM books_authors_link l "
        "JOIN authors a ON a.id=l.author"):
    out.setdefault(str(bid), []).append(name)
n_books = con.execute("SELECT count(*) FROM books").fetchone()[0]
con.close()
shutil.rmtree(workdir, ignore_errors=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
print(f"books={n_books} with_authors={len(out)} -> {OUT}")
