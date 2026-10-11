"""定点重导单个 PDF：克隆「导入 OneDrive PDF」管线、去掉 CalibreDataSource、
把字面路径喂给 PDFDataSource（Calibre 路由看不到非书库目录的文件）。

用法：python reimport_one.py "Zotero/xxx.pdf"
"""
import argparse
import asyncio
import copy
import logging
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/app")
log = logging.getLogger("reimport_one")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="storage 相对路径")
    ap.add_argument("--task", default="导入 OneDrive PDF")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
    from jindai import Task
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins

    dbo = await TaskDBO.get(args.task)
    if not dbo:
        log.error("task %r not found", args.task)
        return 2
    _init_plugins()

    pipeline = copy.deepcopy(dbo.pipeline)
    # 2026-10-08 起存储的任务里已无 CalibreDataSource 阶段，直接给 PDFDataSource 喂字面路径
    new_pipeline, fed = [], None
    for stage in pipeline:
        name = next(iter(stage))
        if "CalibreDataSource" in name:
            fed = args.path
            continue
        if "PDFDataSource" in name and fed is None:
            stage["PDFDataSource"]["content"] = args.path
        elif fed is not None and "PDFDataSource" in name:
            stage["PDFDataSource"]["content"] = fed
            fed = None
        new_pipeline.append(stage)

    clone = SimpleNamespace(
        name=dbo.name + " (single file)",
        pipeline=new_pipeline,
        concurrent=dbo.concurrent,
        resume_next=True,
        params=getattr(dbo, "params", {}),
    )
    task = Task.from_dbo(clone, verbose=False, use_tqdm=False,
                         log=lambda *x: log.info(" ".join(str(i) for i in x)))
    t0 = time.time()
    await task.execute_async()
    log.info("finished in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
