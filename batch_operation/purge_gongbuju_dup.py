"""删除 工部局董事会会议录 的 Journals 副本（用户裁定 2026-10-26：丛刊/会议录保留
Series 原件，删 Journals 副本）。

守卫：每行须满足「同名同大小的 Series 孪生行存在 且 该孪生文件在盘」才删；
DB 行（28 条，全部 0 段落——Series 侧才是导入过的正主）与 Journals 文件一并删，
文件删除走 CIFS（NAS @Recycle 兜底）。备份 TSV 落 reports/。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, "/app")

import common as C

log = C.setup_logging("purge_gongbuju",
                      "batch_operation/reports/purge_gongbuju.log")
BACKUP = "batch_operation/reports/purge_gongbuju_20261026.tsv"
ROOT = "/storage"
JPREFIX = "Journals/工部局董事会会议录/"


async def main() -> int:
    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT j.id, j.path, j.size_bytes, s.path AS twin "
        "FROM file_metadata j JOIN file_metadata s "
        "  ON lower(split_part(s.path,'/',-1)) = lower(split_part(j.path,'/',-1)) "
        " AND s.size_bytes = j.size_bytes "
        " AND s.path LIKE 'Series/工部局董事会会议录/%' "
        "WHERE j.path LIKE $1", JPREFIX + "%")
    log.info("[plan] %d Journals rows with Series twins", len(rows))

    # 守卫：孪生文件在盘 + 本行 0 段落 + 盘上源文件在
    ok, bad = [], []
    for r in rows:
        twin_ok = os.path.isfile(os.path.join(ROOT, r["twin"]))
        src_ok = os.path.isfile(os.path.join(ROOT, r["path"]))
        para = await conn.fetchval(
            "SELECT count(*) FROM paragraph WHERE source_id = $1", r["id"])
        if twin_ok and src_ok and para == 0:
            ok.append(r)
        else:
            bad.append({"path": r["path"], "twin_ok": twin_ok,
                        "src_ok": src_ok, "para": para})
    log.info("[plan] guards ok=%d bad=%d %s", len(ok), len(bad), bad[:3])
    if bad:
        log.warning("[guard] %d rows failed guards — aborting (nothing deleted)",
                    len(bad))
        return 1

    ids = [r["id"] for r in ok]
    with open(BACKUP, "w", encoding="utf-8") as f:
        for r in ok:
            f.write(f"{r['id']}\t{r['path']}\t{r['twin']}\n")
    log.info("[apply] backup -> %s", BACKUP)

    async with conn.transaction():
        await conn.execute(
            "DELETE FROM file_dataset WHERE file_id = ANY($1::uuid[])", ids)
        await conn.execute(
            "DELETE FROM text_embeddings WHERE source_id = ANY($1::uuid[])", ids)
        await conn.execute(
            "DELETE FROM embedding_pending_queue WHERE source_id = ANY($1::uuid[])",
            ids)
        await conn.execute(
            "DELETE FROM page_skip_state WHERE source_id = ANY($1::uuid[])", ids)
        r = await conn.execute(
            "DELETE FROM paragraph WHERE source_id = ANY($1::uuid[])", ids)
        log.info("[apply] paragraphs deleted: %s", r)
        r2 = await conn.execute(
            "DELETE FROM file_metadata WHERE id = ANY($1::uuid[]) "
            "  AND NOT EXISTS (SELECT 1 FROM paragraph p "
            "                  WHERE p.source_id = file_metadata.id)", ids)
        log.info("[apply] file_metadata deleted: %s", r2)
    await conn.close()

    # 文件删除（孪生已验证在盘，源文件逐个 unlink）
    removed = 0
    for row in ok:
        full = os.path.join(ROOT, row["path"])
        try:
            os.unlink(full)
            removed += 1
        except OSError as ex:
            log.warning("[rm] %s failed: %s", row["path"], ex)
    # 空目录收尾
    d = os.path.join(ROOT, "Journals/工部局董事会会议录")
    try:
        if not os.listdir(d):
            os.rmdir(d)
            log.info("[rm] removed empty dir %s", d)
    except OSError:
        pass
    log.info("[done] rows=%d files_removed=%d ts=%s", len(ok), removed,
             time.strftime("%F %T"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
