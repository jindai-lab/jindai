#!/usr/bin/env bash
# 导入队列看板 —— 一条命令看「正在扫哪个 / 排到哪 / 还剩多少」。
# 用法：bash batch_operation/queue.sh [日志里回看的MB数，默认4]
# 日志与 reports/ 是宿主机仓库目录（容器 /app 直接挂载），所以宿主机可直读。
set -uo pipefail
cd "$(dirname "$0")"

LOG=reports/import_onedrive.log
MB=${1:-4}
LOGSZ=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
FROM=$(( LOGSZ > MB * 1048576 ? LOGSZ - MB * 1048576 : 0 ))
tail -c +$((FROM + 1)) "$LOG" > /tmp/.queue_tail.$$ 2>/dev/null || { echo "无日志"; exit 1; }
T=/tmp/.queue_tail.$$
trap 'rm -f "$T"' EXIT

echo "===== 轮次 ====="
grep -a "\[watchdog\]" "$LOG" | tail -1

echo "===== 队列位置（总 70543，按目录优先级排序；已导入的会被跳过）====="
grep -a " importing " "$T" | tail -1 | cut -c1-190
PROG=$(grep -a "\[progress\]" "$T" | tail -1)
[ -z "$PROG" ] && PROG=$(grep -a "\[progress\]" "$LOG" | tail -1)
echo "$PROG"

echo "===== 4 个 worker 正在处理的文件 ====="
for f in reports/import_fast_inflight/w*; do
  [ -e "$f" ] && echo "  $(basename "$f"): $(cat "$f")"
done

echo "===== 还剩多少没导（统计缓存，TTL 1 小时）====="
[ -f reports/import_stats_cache.json ] && \
  echo "  未导入文件=$(sed -n 's/.*onedrive_pdfs_without_paragraph": *\([0-9]*\).*/\1/p' reports/import_stats_cache.json)  段落总数=$(sed -n 's/.*paragraph_total": *\([0-9]*\).*/\1/p' reports/import_stats_cache.json)"
echo "  已盖章 $(docker exec jindai /app/.venv/bin/python -u /app/batch_operation/import_state_stats.py 2>/dev/null | tail -1 || echo '(查不到)')  ← complete 的下一轮不再进队列"

echo "===== 最近 OCR（跳过纯文本 PDF，只列扫描件）====="
grep -a "textless pages" "$T" | tail -3 | cut -c1-165

echo "===== 副本与异常 ====="
echo "  请求 8080=$(grep -ac '8080/ocr' "$T")  8083=$(grep -ac '8083/ocr' "$T")  回退=$(grep -ac 'image fallback' "$T")  segfault=$(grep -ac 'Segmentation fault' "$T")  （计数只覆盖最近 ${MB}MB 日志）"
echo "  拉黑文件=$(grep -vc '^#' reports/import_skip.txt) 个 → reports/import_skip.txt"
