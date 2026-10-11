#!/bin/bash
# 收尾：等采图进程结束 → 补跑限流失败的 → 生成计划 → 带护栏落库 → 打印摘要
# 幂等：可重复运行（计划脚本重跑覆盖、落库有旧值护栏）
set -uo pipefail
cd /home/zhuth/jindai-lab/jindai/batch_operation
J=reports/latin_vision.jsonl

echo "[1/5] 等待采图进程结束…"
while pgrep -f latin_fullname_vision.py >/dev/null; do sleep 60; done
echo "      进程已结束：$(wc -l < "$J") 条记录"

echo "[2/5] 剔除失败记录并补跑"
python3 - <<'EOF'
import json
J = "reports/latin_vision.jsonl"
keep = []
bad = 0
for l in open(J, encoding="utf-8"):
    r = json.loads(l)
    if (r.get("raw") or "").strip() and '"authors"' in r["raw"]:
        keep.append(l)
    else:
        bad += 1
open(J, "w", encoding="utf-8").writelines(keep)
print(f"      保留 {len(keep)} 条，剔除待补 {bad} 条")
EOF
if [ "$(wc -l < "$J")" -lt 1051 ]; then
  python3 latin_fullname_vision.py 4 2>&1 | tail -2
fi

echo "[3/5] 生成计划"
python3 latin_fullname_plan.py

echo "[4/5] 落库（--force：全名不以前姓开头属正常）"
docker cp reports/latin_fullname_plan.tsv jindai:/app/batch_operation/reports/ >/dev/null
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/apply_onechar.py \
    --plan /app/batch_operation/reports/latin_fullname_plan.tsv --apply --force 2>&1 \
    | grep -E "计划行|apply" || echo "      （无计划行）"

echo "[5/5] 复验"
docker exec jindai-pgvector psql -U myuser -d jindai -A -t -c \
  "SELECT '单 token 拉丁作者残留: '||count(*)||' 元素 / '||count(DISTINCT id)||' 文件' \
   FROM file_metadata, LATERAL unnest(authors) a WHERE a ~ '^[A-Za-z][A-Za-z''-]+\$';"
echo "完成"
