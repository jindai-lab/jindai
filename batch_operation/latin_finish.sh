#!/bin/bash
# 收尾：等 latin_fullname_vision.py 跑完 → 生成计划 → 带护栏落库 → 打印摘要
# 幂等：可重复运行（计划脚本重跑覆盖、落库有旧值护栏）
set -uo pipefail
cd /home/zhuth/jindai-lab/jindai/batch_operation

echo "[1/4] 等待采图进程结束…"
while pgrep -f latin_fullname_vision.py >/dev/null; do sleep 60; done
echo "      进程已结束"

echo "[2/4] 生成计划"
python3 latin_fullname_plan.py

echo "[3/4] 落库（--force：全名不以前姓开头属正常）"
docker cp reports/latin_fullname_plan.tsv jindai:/app/batch_operation/reports/ >/dev/null
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/apply_onechar.py \
    --plan /app/batch_operation/reports/latin_fullname_plan.tsv --apply --force 2>&1 \
    | grep -E "计划行|apply" || echo "      （无计划行）"

echo "[4/4] 复验"
docker exec jindai-pgvector psql -U myuser -d jindai -A -t -c \
  "SELECT '单 token 拉丁作者残留: '||count(*)||' 元素 / '||count(DISTINCT id)||' 文件' \
   FROM file_metadata, LATERAL unnest(authors) a WHERE a ~ '^[A-Za-z][A-Za-z''-]+\$';"
echo "完成"
