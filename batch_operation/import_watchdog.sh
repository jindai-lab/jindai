#!/bin/sh
# 导入自愈看门狗 v2（2026-10-10）
#   v1 缺陷：文件级 FileDataError 只让 task 静默结束(exit=0)，看门狗据此"clean exit"
#            退出 —— 表现为"导入又停了"，后面 6 万多个文件根本没轮到。
#   v2 规则：只看**本轮新增**日志（start 行之后）
#     1) 非 0 退出(段错误/abort) → 拉黑本轮最后一条 importing 的文件，重启
#     2) 0 退出但本轮出现 "Failed to open file '...'" → 拉黑该坏文件，重启
#     3) 本轮 imported_files > 0（有进展）→ 继续下一轮
#     4) 既无新错误又零进展 → 判定跑完，正常退出
#   黑名单：batch_operation/reports/import_skip.txt（每行一个 path，# 注释）
cd /app || exit 1
LOG=batch_operation/reports/import_onedrive.log
SKIP=batch_operation/reports/import_skip.txt
[ -f "$SKIP" ] || : > "$SKIP"

for i in $(seq 1 60); do
  echo "[watchdog] round $i start $(date +%T)" >> "$LOG"
  start=$(wc -l < "$LOG")
  .venv/bin/python batch_operation/import_onedrive_pdfs.py --progress-secs 120 >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog] round $i exit=$rc" >> "$LOG"
  tail_from() { awk -v s="$start" 'NR>s' "$LOG"; }

  target=""
  if [ "$rc" -ne 0 ]; then
    target=$(tail_from | grep -o 'importing .*' | tail -1 | sed 's/^importing //')
    echo "[watchdog] round $i crashed rc=$rc last=$target" >> "$LOG"
  fi
  if [ -z "$target" ]; then
    target=$(tail_from | grep -o "Failed to open file '[^']*'" | tail -1 \
             | sed -e "s/^Failed to open file '//" -e "s/'$//" -e 's#^/storage/##')
  fi

  added=""
  if [ -n "$target" ]; then
    if grep -qxF "$target" "$SKIP" 2>/dev/null; then
      echo "[watchdog] round $i error-file already skipped: $target" >> "$LOG"
    else
      echo "$target" >> "$SKIP"
      echo "[watchdog] SKIP $target" >> "$LOG"
      added=1
    fi
  fi
  if [ -n "$added" ]; then sleep 3; continue; fi

  imp=$(tail_from | grep -o '"imported_files": [0-9]*' | tail -1 | grep -o '[0-9]*')
  imp=${imp:-0}
  echo "[watchdog] round $i imported=$imp rc=$rc" >> "$LOG"
  if [ "$imp" -gt 0 ]; then sleep 3; continue; fi
  echo "[watchdog] no new error and no progress -> done $(date +%T)" >> "$LOG"
  exit 0
done
echo "[watchdog] rounds exhausted" >> "$LOG"
