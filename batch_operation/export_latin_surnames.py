"""导出「只有姓」的拉丁作者行（单 token），供 PDF 查证全名。

输出 reports/latin_surname_rows.tsv：id \t path \t 原 authors(竖线分隔) \t 姓(逗号分隔)
"""
import asyncio, csv, re, sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
import common as C  # noqa: E402

TOKEN = re.compile(r"^[A-Za-z][A-Za-z'\-]+$")


async def main():
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, authors FROM file_metadata "
            "WHERE EXISTS (SELECT 1 FROM unnest(authors) x WHERE x ~ '^[A-Za-z][A-Za-z''\\-]+$') "
            "ORDER BY path")
        n = 0
        with open("/app/batch_operation/reports/latin_surname_rows.tsv", "w",
                  encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "path", "authors", "姓"])
            for r in rows:
                surs = [x for x in r["authors"] if TOKEN.match(x)]
                if not surs:
                    continue
                w.writerow([r["id"], r["path"], "|".join(r["authors"]), ",".join(surs)])
                n += 1
        print(f"导出 {n} 文件（含单 token 拉丁作者）")
    finally:
        await conn.close()

asyncio.run(main())
