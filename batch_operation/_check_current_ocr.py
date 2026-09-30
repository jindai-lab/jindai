#!/usr/bin/env python3
"""Spot-check live OCR quality: find import pid's current file via py-spy,
then print its newest paragraphs (content head) from the DB."""
import re
import subprocess
import sys

PID = sys.argv[1] if len(sys.argv) > 1 else "2633"


def sh(*cmd: str, timeout: int = 60) -> str:
    return subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout).stdout


dump = sh("docker", "exec", "jindai", "sh", "-c",
          f"/app/.venv/bin/py-spy dump -p {PID} --locals 2>/dev/null")
m = re.search(r"filepath\s*=\s*['\"]([^'\"]+)['\"]", dump)
if not m:
    print("filepath not found in py-spy dump; frames:")
    print("\n".join(l for l in dump.splitlines() if "filepath" in l)[:500])
    sys.exit(1)
fp = m.group(1)
print("current file:", fp)

fp_sql = fp.replace("'", "''")
sql = (
    "SELECT p.source_page, left(regexp_replace(p.content, chr(10), ' ', 'g'), 140) "
    "FROM paragraph p JOIN file_metadata f ON p.source_id = f.id "
    f"WHERE f.path = '{fp_sql}' ORDER BY p.source_page DESC LIMIT 5"
)
out = sh("docker", "exec", "jindai-pgvector", "psql", "-U", "myuser",
         "-d", "jindai", "-tAc", sql, timeout=90)
print("newest paragraphs (page, content head):")
print(out.strip() or "(none yet)")
