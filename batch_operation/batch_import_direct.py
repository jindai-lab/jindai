#!/usr/bin/env python
"""批量 PDF 导入 CLI 薄壳（2026-10-25 起实现迁至主库 jindai/import_cli.py）。

本脚本只做参数透传，行为与 2026-10-18 引擎完全一致（参数同名同义）：
- 清单/磁盘登记/工具书与磁盘缺失过滤/毒文件跳过/暂缓清单；
- stage 链按插件模块就地构造（不读 Task/TaskDBO）；
- 文件级并发 + 页集续传 + 盖章 complete/partial。

保留本入口的原因：
1. 进程名含 batch_import_direct —— gpu_cpu_monitor.py / auto_ingest_scoped.py
   的 ps grep 无需改动；
2. 看门狗 import_watchdog_fast.sh / 手工调用路径不变。

常驻自动导入由 jindai/import_worker.py（web app 内 supervisor）经本壳拉起。
"""
import sys

sys.path.insert(0, "/app")

if __name__ == "__main__":
    import asyncio
    from jindai.import_cli import main
    sys.exit(asyncio.run(main()))
