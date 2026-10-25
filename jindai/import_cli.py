"""PDF 导入引擎 CLI（主库版）：`python -m jindai.import_cli [args]`。

batch_operation/batch_import_direct.py 已退化为调用本模块的薄壳（进程名保持
不变，监控 grep / auto_ingest 拉起路径无需改动）。参数与 2026-10-18 引擎全兼容：

    python -m jindai.import_cli --selftest          # 只构造链，不碰库
    python -m jindai.import_cli --dry-run --no-scan  # 只报清单统计
    python -m jindai.import_cli --limit 8 --no-scan -w 4
    python -m jindai.import_cli                      # 全量（含磁盘登记）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

# 容器内直跑时保证 /app 在 sys.path（python -m 方式天然满足，薄壳脚本不满足）
if "/app" not in sys.path:
    sys.path.insert(0, "/app")


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("jindai.import")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    # 旧看门狗把 stdout 重定向进 import_onedrive.log；手工跑时也直接打 stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    from jindai.import_lists import REPORTS
    try:
        REPORTS.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(REPORTS / "import_direct.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    return logger


async def main() -> int:
    log = _setup_logging()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress-secs", type=int, default=600)
    ap.add_argument("--stats-cache-secs", type=float, default=3600)
    ap.add_argument("--scan-cache-secs", type=float, default=3600)
    ap.add_argument("--progress-count-secs", type=int, default=1800)
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--selftest", action="store_true",
                    help="只构造 stage 链并打印，不连库不写数据")
    ap.add_argument("--report",
                    default="batch_operation/reports/import_direct.json")
    args = ap.parse_args()

    from jindai import import_engine as E

    if args.selftest:
        _cls, pipeline, saver = E.build_chain(args.batch)
        names = [f"{type(s).__name__}" for s in pipeline.stages]
        print("chain:", " -> ".join(names))
        print("saver:", type(saver).__name__, "batch=", saver.batch)
        print("ds_params:", json.dumps(E.PDF_PARAMS, ensure_ascii=False))
        return 0

    keep, list_meta = await E.prepare_list(
        no_scan=args.no_scan, only=args.only,
        scan_cache_secs=args.scan_cache_secs,
        stats_cache_secs=args.stats_cache_secs,
    )
    log.info("pre: import list=%s", json.dumps(list_meta, ensure_ascii=False))

    if args.limit > 0:
        keep = keep[:args.limit]
        log.info("--limit %d: processing first %d files", args.limit, len(keep))

    if args.dry_run:
        print(json.dumps({"list": list_meta, "limit": len(keep)},
                         ensure_ascii=False, indent=2))
        return 0

    if not E.acquire_lock():
        log.error("another importer is running (lock %s)", E.LOCK_FILE)
        return 3

    try:
        summary = await E.run_round(
            keep, workers=args.workers, batch=args.batch,
            store_state=not args.no_state,
            progress_secs=args.progress_secs,
            progress_count_secs=args.progress_count_secs,
        )
        summary["list"] = list_meta
    finally:
        E.release_lock()

    from jindai.import_lists import REPORTS
    report_path = args.report
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    except OSError:
        # 相对路径按 cwd；容器内 cwd=/app 正常。写不了就落 REPORTS。
        with open(REPORTS / "import_direct.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    log.info("summary: %s", json.dumps(summary, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
