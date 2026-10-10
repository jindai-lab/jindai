#!/bin/sh
# 崩溃自愈看门狗：进程若非 0 退出（段错误 139），把日志里最后一条 importing 路径
# 追加进 import_skip.txt，然后重启，最多 60 轮。
cd /app || exit 1
LOG=batch_operation/reports/import_onedrive.log
SKIP=batch_operation/reports/import_skip.txt
for i in $(seq 1 60); do
  echo "[watchdog] round $i start $(date +%T)" >> "$LOG"
  .venv/bin/python batch_operation/import_onedrive_pdfs.py --progress-secs 120 >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog] round $i exit=$rc" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "[watchdog] clean exit" >> "$LOG"; exit 0; }
  # 退出码 139/134 = 段错误/abort：找最后一条 importing
  last=$(grep -o 'importing .*' "$LOG" | tail -1 | sed 's/^importing //')
  if [ -n "$last" ] && ! grep -qxF "$last" "$SKIP" 2>/dev/null; then
    echo "$last" >> "$SKIP"
    echo "[watchdog] SKIP $last" >> "$LOG"
  fi
  sleep 5
done
echo "[watchdog] gave up" >> "$LOG"
