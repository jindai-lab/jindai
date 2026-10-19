"""一次性：对待导入 PDF 抽样判文本层占比（按顶层目录分桶）。"""
import asyncio
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/app")
import fitz  # noqa: E402
import common as C  # noqa: E402


def has_text(path: str, sample: int = 6) -> bool | None:
    try:
        doc = fitz.open(os.path.join("/storage", path))
    except Exception:
        return None
    try:
        n = doc.page_count
        if n == 0:
            return None
        idxs = sorted(set([0, n // 2, n - 1] + [int(n * i / sample) for i in range(sample)]))
        good = sum(1 for i in idxs
                   if i < n and len((doc[i].get_text() or "").strip()) >= 30)
        return good >= max(1, int(len(idxs) * 0.2))
    except Exception:
        return None
    finally:
        doc.close()


async def main():
    conn = await C.get_conn()
    rows = await conn.fetch(
        """
        SELECT split_part(f.path,'/',1) AS top, f.path FROM file_metadata f
        WHERE lower(f.path) LIKE '%.pdf'
          AND f.path NOT LIKE 'Tool Books/%' AND f.path NOT LIKE '.trash/%'
          AND f.path NOT LIKE '../%'
          AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
        """
    )
    await conn.close()
    by_top = {}
    for r in rows:
        by_top.setdefault(r["top"], []).append(r["path"])
    print("pending by top:", {k: len(v) for k, v in
                              sorted(by_top.items(), key=lambda kv: -len(kv[1]))})
    random.seed(42)
    for top, paths in sorted(by_top.items(), key=lambda kv: -len(kv[1])):
        n = min(30, len(paths))
        sample = random.sample(paths, n)
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(has_text, sample))
        ok = [r for r in res if r is not None]
        txt = sum(1 for r in ok if r)
        print(f"{top}: sampled={n} valid={len(ok)} with_text={txt} "
              f"scans={len(ok)-txt} (text share {txt/max(1,len(ok)):.0%})")


asyncio.run(main())
