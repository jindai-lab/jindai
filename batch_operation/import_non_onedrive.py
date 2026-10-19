"""One-off: import 3 non-OneDrive PDFs via PDFDataSource directly.

Albums/ and Authors/ are plain directories (not Calibre libraries), so the
Calibre-task route can't see them. This run clones the "导入 OneDrive PDF"
pipeline but drops the CalibreDataSource stage and feeds its downstream
content with literal storage-relative paths.
"""
import argparse
import copy
import logging
import sys
import time

sys.path.insert(0, "/app")
log = logging.getLogger("import_non_onedrive")

PATHS = [
    "Albums/点石斋画报全集.pdf",
    "Authors/李飞飞/The Worlds I See.pdf",
    "Authors/DeSoto, Ilee/The Art of Algorithms_ Exploring AI in Creative "
    "Industries -- Ilee DeSoto -- 2024 -- a2163cef1b48d640b7cc08f81194a387 "
    "-- Anna’s Archive.pdf",
]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="导入 OneDrive PDF")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")

    from jindai import Task
    from jindai.models import TaskDBO
    from jindai.__main__ import _init_plugins

    dbo = await TaskDBO.get(args.task)
    if not dbo:
        log.error("task %r not found", args.task)
        return 2
    _init_plugins()

    pipeline = copy.deepcopy(dbo.pipeline)
    # Drop the CalibreDataSource stage and feed its downstream content with
    # our literal paths (content flows: CalibreDataSource -> PDFDataSource).
    new_pipeline = []
    fed_content = None
    for stage in pipeline:
        name = next(iter(stage))
        if "CalibreDataSource" in name:
            fed_content = "\n".join(PATHS)
            continue
        if fed_content is not None and "PDFDataSource" in name:
            stage["PDFDataSource"]["content"] = fed_content
            fed_content = None
        new_pipeline.append(stage)
    if fed_content is not None:
        log.error("no PDFDataSource stage found after CalibreDataSource")
        return 2

    from types import SimpleNamespace
    clone = SimpleNamespace(
        name=dbo.name + " (non-onedrive 3)",
        pipeline=new_pipeline,
        concurrent=dbo.concurrent,
        resume_next=getattr(dbo, "resume_next", False),
        params=getattr(dbo, "params", {}),
    )
    task = Task.from_dbo(clone, verbose=False, use_tqdm=False,
                         log=lambda *x: log.info(" ".join(str(i) for i in x)))
    t0 = time.time()
    await task.execute_async()
    log.info("finished in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main()))
