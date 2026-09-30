#!/usr/bin/env python3
"""Verify the author-bucket fix end to end (run on host, no shell quoting).

1. new-code stage logs ('... OCR N/M textless pages ...')
2. fresh paragraphs (embedding_pending_queue, last 6 min) — author column set?
3. Kurt Gödel 282-file bucket reality check
4. import log tail
"""
import subprocess
import time


def sh(*cmd: str, timeout: int = 90) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def psql(sql: str) -> str:
    return sh("docker", "exec", "jindai-pgvector", "psql", "-U", "myuser",
              "-d", "jindai", "-tAc", sql, timeout=60)


FRESH_SQL = """
    SELECT coalesce(nullif(p.author,''), '<NULL>') AS author, count(*)
    FROM embedding_pending_queue q
    JOIN paragraph p ON p.id = q.id
    WHERE q.created_at > now() - interval '6 min'
    GROUP BY 1 ORDER BY 2 DESC LIMIT 12
"""

print("== new-code stage logs ==")
logs = sh("docker", "logs", "--since", "20m", "jindai", timeout=60)
hits = [ln for ln in logs.splitlines()
        if "textless pages" in ln or "importing " in ln]
print("\n".join(hits[-6:]) or "(stage logs not on container stdout)")

print("== fresh paragraph authors (poll up to 2.5 min) ==")
fresh = ""
for i in range(4):
    fresh = psql(FRESH_SQL).strip()
    if fresh:
        print(f"(round {i + 1})")
        print(fresh)
        break
    time.sleep(40)
else:
    print("STILL EMPTY — import likely mid-prescan on first (scanned) file")

print("== Kurt Gödel folder reality check ==")
print(psql("SELECT count(*) FROM file_metadata "
           "WHERE path LIKE 'OneDrive/Kurt Gödel/%' "
           "AND lower(path) LIKE '%.pdf'").strip())
print(psql("SELECT left(path, 105) FROM file_metadata "
           "WHERE path LIKE 'OneDrive/Kurt Gödel/%' LIMIT 4").strip())

print("== import log tail ==")
with open("/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
          "import_onedrive.log", encoding="utf-8", errors="replace") as fh:
    print("".join(fh.readlines()[-2:]).rstrip())
