#!/usr/bin/env python3
"""建立存储根的文件名索引：basename -> [绝对路径...]。供路径修复时跨目录定位。"""
import os, json, sys, time

ROOT = "/misc/Public/jindai-storage"
OUT = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/disk_index.jsonl"
SKIP = {".git", ".trash", "__pycache__"}
# 与文献无关的巨型镜像，跳过以加速
PRUNE = {"Albums/vndb", "Albums/gutenberg", "Albums/www.starringthecomputer.com", "Albums/ISFDB-backup-MySQL-55-2023-10-14"}

n = 0
t0 = time.time()
with open(OUT, "w", encoding="utf-8") as f:
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        rel_dir = os.path.relpath(dirpath, ROOT)
        dirnames[:] = [d for d in dirnames if d not in SKIP and
                       os.path.join(rel_dir, d).lstrip("./") not in PRUNE]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, ROOT)
            f.write(json.dumps({"b": fn, "p": rel}, ensure_ascii=False) + "\n")
            n += 1
        if n and n % 20000 < len(filenames):
            print(f"  已索引 {n} 个文件 {round(time.time()-t0)}s", flush=True)
print(f"完成：索引 {n} 个文件，耗时 {round(time.time()-t0)}s", flush=True)
