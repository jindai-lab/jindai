#!/usr/bin/env python3
"""全库盘点：file_metadata 里 path 指向本地但文件不在盘的行，输出 TSV 供修路径。"""
import os, subprocess, csv, sys, time

STORE = "/misc/Public/jindai-storage"
OUT = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/disk_missing_audit.tsv"
FIELDS = ["path", "file", "title", "authors", "item_type", "绝对路径", "父目录存在", "父目录", "同目录同名文件"]


def norm(p):
    return p[7:] if p.startswith("../app/") else p


def q(sql):
    return subprocess.run(["docker", "exec", "jindai-pgvector", "psql", "-U", "myuser", "-d", "jindai",
                           "-A", "-t", "-F", "\t", "-c", sql], capture_output=True, text=True).stdout


def main():
    t0 = time.time()
    n = web = local = miss = 0
    w = csv.writer(open(OUT, "w", encoding="utf-8", newline=""), delimiter="\t")
    w.writerow(FIELDS)
    chunk = 20000
    off = 0
    while True:
        rows = [l.split("\t") for l in
                q(f"SELECT path, coalesce(title,''), coalesce(array_to_string(authors,'|'),''), coalesce(item_type,'') "
                  f"FROM file_metadata ORDER BY path LIMIT {chunk} OFFSET {off}").splitlines() if "\t" in l]
        if not rows:
            break
        for r in rows:
            path, title, au, it = (r + ["", "", "", ""])[:4]
            n += 1
            if path.startswith(("http://", "https://")):
                web += 1
                continue
            local += 1
            full = os.path.join(STORE, norm(path))
            if os.path.exists(full):
                continue
            miss += 1
            d = os.path.dirname(full)
            dOK = os.path.isdir(d)
            twin = ""
            if dOK:
                base = os.path.basename(full)
                try:
                    same = [f for f in os.listdir(d) if f == base]
                    twin = "有" if same else ""
                except Exception:
                    twin = "读目录失败"
            w.writerow([path, os.path.basename(path), title, au, it, full, "是" if dOK else "否", d, twin])
        off += chunk
        print(f"  已查 {n} 行（本地 {local}，缺失 {miss}，网址 {web}）{round(time.time()-t0)}s", flush=True)
    print(f"完成：总 {n} | 网址 {web} | 本地文件 {local} | 不在盘 {miss} | {round(time.time()-t0)}s", flush=True)


main()
