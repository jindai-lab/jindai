"""Read or patch a parameter of a stage inside a stored task pipeline.

The pipeline column of ``task_dbo`` is JSONB: a list of one-key dicts like
``{"PDFDataSource": {...}}``. This tool locates the named stage, optionally
sets key=value pairs (value parsed as JSON when possible, else kept as
string), and writes the pipeline back atomically.

  # inspect
  python batch_operation/task_param.py --task "导入 OneDrive PDF" --stage PDFDataSource

  # set params
  python batch_operation/task_param.py --task "导入 OneDrive PDF" \\
      --stage PDFDataSource --set ocr_fallback=true

Multiple --set flags are allowed; run without --set for a dry preview.
"""
from __future__ import annotations

import argparse
import json
import sys

import common as C

log = C.setup_logging("task_param", "batch_operation/reports/task_param.log")

SELECT_SQL = "SELECT pipeline FROM task_dbo WHERE name = $1"
UPDATE_SQL = "UPDATE task_dbo SET pipeline = $2::jsonb WHERE name = $1"


def parse_value(raw: str):
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="exact task name")
    ap.add_argument("--stage", required=True, help="stage key, e.g. PDFDataSource")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                   help="parameter to set (repeatable)")
    args = ap.parse_args()

    conn = await C.get_conn()
    try:
        pipeline = await conn.fetchval(SELECT_SQL, args.task)
        if pipeline is None:
            log.error("task %r not found", args.task)
            return 2
        if isinstance(pipeline, str):  # asyncpg may return raw JSONB text
            pipeline = json.loads(pipeline)
        node = next((n for n in pipeline if args.stage in n), None)
        if node is None:
            log.error("stage %r not in pipeline (stages: %s)", args.stage,
                      [next(iter(n)) for n in pipeline])
            return 3
        params = node[args.stage]
        log.info("task=%r stage=%s current params: %s",
                 args.task, args.stage, json.dumps(params, ensure_ascii=False))
        if not args.set:
            return 0
        for kv in args.set:
            if "=" not in kv:
                log.error("bad --set %r (want K=V)", kv)
                return 4
            k, v = kv.split("=", 1)
            old = params.get(k, "<unset>")
            params[k] = parse_value(v)
            log.info("set %s: %r -> %r", k, old, params[k])
        await conn.execute(UPDATE_SQL, args.task, json.dumps(pipeline, ensure_ascii=False))
        check = await conn.fetchval(SELECT_SQL, args.task)
        log.info("written; read-back %s stage params: %s",
                 args.stage,
                 json.dumps(next(n for n in check if args.stage in n)[args.stage],
                            ensure_ascii=False))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
