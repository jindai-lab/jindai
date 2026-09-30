#!/usr/bin/env bash
# embed_missing 看门狗：每 10 分钟采样 text_embeddings 计数。
# - 进程消失 30 分钟 → 认为正常跑完/人工停止，看门狗退出。
# - 进程在、计数 60 分钟纹丝不动 → 判定死等，杀掉重启（最多 3 次，然后放弃并记日志）。
# 重启的 embed 自带 300s HTTP 超时与按缺失重扫，重启安全幂等。
set -u
LOG=/home/zhuth/jindai-lab/jindai/batch_operation/reports/embed_watchdog.log
OUT=/home/zhuth/jindai-lab/jindai/batch_operation/reports/embed_missing.out
LOCK=/tmp/embed_watchdog.lock
exec 9>"$LOCK"; flock -n 9 || { echo "$(date '+%F %T') another watchdog running" >>"$LOG"; exit 0; }

emb()  { docker exec jindai-pgvector psql -U myuser -d jindai -tAc \
           "SELECT count(*) FROM text_embeddings" 2>/dev/null | tr -d ' \r'; }
alive(){ docker exec jindai sh -c 'ps -eo args | grep -c "[e]mbed_missing.py --scope"' 2>/dev/null | tr -d ' \r'; }

last_n=$(emb); last_t=$(date +%s); absent=0; restarts=0
echo "$(date '+%F %T') watchdog start, emb=$last_n" >>"$LOG"
while true; do
  sleep 600
  now=$(date +%s); n=$(emb); a=$(alive)
  if [ -z "$n" ] || [ -z "$a" ]; then echo "$(date '+%F %T') probe failed (emb='$n' alive='$a')" >>"$LOG"; continue; fi
  if [ "$a" = "0" ]; then
    absent=$((absent+1))
    if [ "$absent" -ge 3 ]; then
      echo "$(date '+%F %T') embed gone 30min — assuming finished; watchdog exits" >>"$LOG"; exit 0
    fi
    continue
  fi
  absent=0
  if [ "$n" != "$last_n" ]; then last_n=$n; last_t=$now; continue; fi
  if [ $((now - last_t)) -ge 3600 ]; then
    restarts=$((restarts+1))
    if [ "$restarts" -gt 3 ]; then
      echo "$(date '+%F %T') STALLED 3x (last emb=$n) — giving up, NEED MANUAL FIX" >>"$LOG"; exit 1
    fi
    echo "$(date '+%F %T') STALL (emb=$n static $((now-last_t))s) — restarting embed #$restarts" >>"$LOG"
    docker exec jindai sh -c 'pkill -f "embed_missing.py --scope" || true'
    sleep 3
    nohup docker exec -w /app jindai /app/.venv/bin/python \
      /app/batch_operation/embed_missing.py --scope all --inflight 16 --batch 256 \
      >>"$OUT" 2>&1 &
    last_n=$(emb); last_t=$(date +%s)
  fi
done
