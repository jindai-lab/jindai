"""定向单测：用 batch_import_fast 的真实链路导入 N 个待导入的 OneDrive 书。

验证点：段落落库 / embedding_pending_queue 入队 / 数据集规则挂载 / terms。
  docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/_test_fast_one.py [N]
"""
from __future__ import annotations

import asyncio
import copy
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")

import common as C  # noqa: E402
import batch_import_fast as B  # noqa: E402

log = C.setup_logging("test_fast_one", None)


async def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    from jindai.pipeline import Pipeline
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins

    dbo = await TaskDBO.get("导入 OneDrive PDF")
    assert dbo, "task not found"
    _init_plugins()

    BatchSave = B._make_saver_class(50)
    Pipeline.ctx["BatchSaveParagraph"] = BatchSave

    ds_params = None
    chain = []
    for st in dbo.pipeline:
        name = next(iter(st))
        if "PDFDataSource" in name:
            ds_params = dict(st[name])
            continue
        if "CalibreDataSource" in name:
            continue
        chain.append(B.substitute_save(st, 50))
    assert ds_params is not None
    ds_params.pop("content", None)

    conn = await C.get_conn()
    rows = await conn.fetch(
        """
        SELECT f.path FROM file_metadata f
        WHERE lower(f.path) LIKE '%.pdf'
          AND f.path LIKE 'OneDrive/%'
          AND f.path NOT LIKE 'Tool Books/%'
          AND f.path NOT LIKE '.trash/%'
          AND NOT EXISTS (SELECT 1 FROM paragraph p WHERE p.source_id = f.id)
          AND NOT EXISTS (SELECT 1 FROM page_skip_state s WHERE s.source_id = f.id)
        ORDER BY f.path LIMIT $1
        """,
        n,
    )
    paths = [r["path"] for r in rows]
    await conn.close()
    print("TEST PATHS:", *paths, sep="\n  ")
    if not paths:
        print("no eligible pending files")
        return 1

    pipeline = Pipeline(copy.deepcopy(chain), log=lambda *x: print("chain|", *x))
    pipeline.gctx = {}
    head = pipeline.stages[0]
    saver = B.find_saver(pipeline.stages)
    assert saver is not None, "saver not found"

    logfn = B.make_logfn("/app/batch_operation/reports/import_onedrive.json.log")
    t0 = asyncio.get_event_loop().time()
    for p in paths:
        try:
            await B.import_one(p, head, ds_params, logfn)
        except Exception:
            import traceback
            traceback.print_exc()
        await saver.flush()
    elapsed = asyncio.get_event_loop().time() - t0
    logfn.close()

    print(f"SAVED={saver.saved} DROPPED={saver.failed} ELAPSED={elapsed:.1f}s")

    # ---- read-back verification ----
    conn = await C.get_conn()
    try:
        for p in paths:
            row = await conn.fetchrow(
                """
                SELECT f.id,
                       (SELECT count(*) FROM paragraph x WHERE x.source_id = f.id) AS paras,
                       (SELECT count(*) FROM embedding_pending_queue q WHERE q.source_id = f.id) AS queued,
                       (SELECT string_agg(d.name, ',') FROM file_dataset fd
                          JOIN dataset d ON d.id = fd.dataset_id
                         WHERE fd.file_id = f.id) AS datasets
                FROM file_metadata f WHERE f.path = $1
                """,
                p,
            )
            print(f"READBACK {p}\n  paras={row['paras']} queued={row['queued']} "
                  f"datasets={row['datasets']}")
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
