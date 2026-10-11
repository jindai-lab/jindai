"""authors_stray3_fix.py — 收尾 3 条仍带垃圾 authors 的行（2026-10-11）。

这三条躲过了 SCOPE_SQL（或被文件名回填成同样的垃圾值），逐条给证据：

1. Authors/Zizek, Slavoj/齐泽克 - 享受你的症状：好莱坞内外的拉康.pdf
   现状 title='齐泽克' / authors=['享受你的症状：好莱坞内外的拉康']（题名↔作者颠倒）。
   → title=题名，authors=['Slavoj Žižek']（库内既有 8 条 Zizek 条目一律写 Slavoj Žižek）。
2. Journals/…/4280_嘉定錢大昕全集（拾） - 2傳記資料.pdf
   现状 title='4280_嘉定錢大昕全集（拾）' / authors=['2傳記資料']（卷次说明当作者）。
   → title=文件名主干（与同目录 8 条兄弟行一致），authors=['錢大昕']。
3. OneDrive/51人/51摊 (8047)/51摊 - 51人.pdf
   现状 authors=['51人'] —— 那是策划编辑团队（=目录名），不是作者。
   → 作者据做書《杭州市集上卖得最好的80种书和文创》著录：曹飞乐、张婷、Erika、二斤马。
   （PDF 首页只有目录，p8 每篇标 'Author: Zhang Ting'，可与著录互证。）

dry-run 默认；--apply 写库（先落 backup，UPDATE 带 title 旧值护栏）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

log = C.setup_logging(
    "authors_stray3_fix", "batch_operation/reports/authors_stray3_fix.log")

FIXES: dict[str, dict] = {
    "Authors/Zizek, Slavoj/齐泽克 - 享受你的症状：好莱坞内外的拉康.pdf": {
        "title": "享受你的症状：好莱坞内外的拉康",
        "authors": ["Slavoj Žižek"],
        "why": "题名↔作者颠倒；作者按目录名 Zizek, Slavoj 且与库内 8 条既有写法一致",
    },
    "Journals/历代史料笔记丛刊/嘉定錢大昕全集全十冊/4280_嘉定錢大昕全集（拾） - 2傳記資料.pdf": {
        "title": "4280_嘉定錢大昕全集（拾） - 2傳記資料",
        "authors": ["錢大昕"],
        "why": "title 改成文件名主干（同目录兄弟行口径）；'2傳記資料' 是卷次说明不是作者",
    },
    "OneDrive/51人/51摊 (8047)/51摊 - 51人.pdf": {
        "title": "51摊",
        "authors": ["曹飞乐", "张婷", "Erika", "二斤马"],
        "why": "做書推文著录：作者 曹飞乐/张婷/Erika/二斤马，策划编辑 51人（目录名）",
    },
}


async def run(apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, coalesce(title,'') AS title, authors "
            "FROM file_metadata WHERE path = ANY($1::text[])",
            list(FIXES))
        plan, missing = [], [p for p in FIXES if p not in {r["path"] for r in rows}]
        for r in rows:
            f = FIXES[r["path"]]
            plan.append({"id": str(r["id"]), "path": r["path"],
                         "old_title": r["title"], "new_title": f["title"],
                         "old_authors": list(r["authors"] or []),
                         "new_authors": f["authors"], "why": f["why"],
                         "change": (r["title"] != f["title"]
                                    or list(r["authors"] or []) != f["authors"])})
        todo = [p for p in plan if p["change"]]
        summary = {"applied": apply, "matched": len(plan), "missing": missing,
                   "to_update": len(todo)}
        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/authors_stray3_plan.json", "w",
                  encoding="utf-8") as fh:
            json.dump({"summary": summary, "plans": plan}, fh,
                      ensure_ascii=False, indent=1)
        if apply and todo:
            day = dt.date.today().strftime("%Y%m%d")
            bak = f"batch_operation/reports/authors_stray3_backup_{day}.tsv"
            with open(bak, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow(["id", "path", "title", "authors"])
                for p in todo:
                    w.writerow([p["id"], p["path"], p["old_title"],
                                "{" + ",".join(p["old_authors"]) + "}"])
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2::text[], title = $3 "
                "WHERE id = $1 AND coalesce(title,'') = $4",
                [(p["id"], p["new_authors"], p["new_title"], p["old_title"])
                 for p in todo])
            summary["backup"] = bak
            log.info("[apply] updated=%d backup=%s", len(todo), bak)
        # 读回
        back = await conn.fetch(
            "SELECT path, coalesce(title,'') AS title, authors "
            "FROM file_metadata WHERE path = ANY($1::text[])", list(FIXES))
        summary["readback"] = {r["path"]: {"title": r["title"],
                                           "authors": list(r["authors"] or [])}
                               for r in back}
        log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
        return summary
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(apply=args.apply)),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
