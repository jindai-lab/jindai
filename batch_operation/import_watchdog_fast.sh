#!/bin/sh
# 导入看门狗（2026-10-18）：驱动 batch_import_direct.py
#   —— 该引擎不读 Task/TaskDBO，stage 链按 plugins 模块就地构造。
#   旧引擎 batch_import_fast.py 同日退役（保留文件，看门狗不再调用）。
#
# 与 legacy import_watchdog.sh 的区别 —— 崩溃归因改为「在飞集合取交集」：
#   文件级并发下，单轮 crash 时在飞的 N 个文件里只有 1 个是毒文件，
#   按"最后一条 importing 行"拉黑会误伤。于是：
#     crash 轮读 reports/import_direct_inflight/w* 得到本轮在飞集合，
#     与**上一连续 crash 轮**的在飞集合取交集，交集里的文件才拉黑
#     （连续两轮崩在同一文件上 = 毒文件）。
#   规则：
#     1) 非 0 退出 → 记录在飞集合，交集拉黑，重启
#     2) 0 退出但本轮 imported_files > 0 → 再跑一轮
#     3) 0 退出且无进展 → 跑完，正常退出
#   黑名单：batch_operation/reports/import_skip.txt（每行一个 path，# 注释）
cd /app || exit 1
LOG=batch_operation/reports/import_onedrive.log
SKIP=batch_operation/reports/import_skip.txt
INFL=batch_operation/reports/import_direct_inflight
ENGINE=batch_operation/batch_import_direct.py
[ -f "$SKIP" ] || : > "$SKIP"
mkdir -p "$INFL"
rm -f "$INFL"/w* "$INFL"/.cur "$INFL"/.prev "$INFL"/.inter 2>/dev/null

prev_file=""
export LC_ALL=C
for i in $(seq 1 60); do
  echo "[watchdog] round $i start $(date +%T)" >> "$LOG"
  start=$(wc -l < "$LOG")
  .venv/bin/python "$ENGINE" --progress-secs 600 >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog] round $i exit=$rc" >> "$LOG"
  tail_from() { awk -v s="$start" 'NR>s' "$LOG"; }

  if [ "$rc" -ne 0 ]; then
    cat "$INFL"/w* 2>/dev/null | sort -u > "$INFL/.cur"
    if [ -s "$INFL/.cur" ] && [ -n "$prev_file" ] && [ -s "$prev_file" ]; then
      comm -12 "$INFL/.cur" "$prev_file" > "$INFL/.inter"
      while IFS= read -r tgt; do
        [ -n "$tgt" ] || continue
        if grep -qxF "$tgt" "$SKIP" 2>/dev/null; then
          echo "[watchdog] round $i already skipped: $tgt" >> "$LOG"
        else
          echo "$tgt" >> "$SKIP"
          echo "[watchdog] round $i SKIP (crashed twice in a row): $tgt" >> "$LOG"
        fi
      done < "$INFL/.inter"
    else
      echo "[watchdog] round $i crash, no previous crash set -> no blacklist yet" >> "$LOG"
    fi
    cp "$INFL/.cur" "$INFL/.prev" 2>/dev/null || : > "$INFL/.prev"
    prev_file="$INFL/.prev"
    sleep 3
    continue
  fi

  # 干净退出：重置 crash 链，再看有没有进展
  prev_file=""
  imp=$(tail_from | grep -o '"imported_files": [0-9]*' | tail -1 | grep -o '[0-9]*$')
  imp=${imp:-0}
  echo "[watchdog] round $i imported=$imp rc=$rc" >> "$LOG"
  if [ "$imp" -gt 0 ]; then
    sleep 3
    continue
  fi
  echo "[watchdog] no new error and no progress -> done $(date +%T)" >> "$LOG"
  exit 0
done
echo "[watchdog] rounds exhausted" >> "$LOG"
