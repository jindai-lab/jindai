"""兜底补挂：清理中失去全部数据集绑定的文件（2026-10-10）。

数据源 = 09:00 全量绑定备份 reports/dataset_bindings_backup_20261010.tsv 中
属于「被删数据集」（外国文学/外国哲学/期刊/现当代非文学家/特辑去重）的路径，
与库中现状比对，取**当前 0 绑定**的补挂：

  Authors/OneDrive/Supplementary Books（有作者段）→ 书库--<作者>
  其余 → ensure_dataset 口径：父目录名匹配已有 `X` 或 `*--X`，否则新建 X

stdin 每行一个 path；dry-run 默认，--apply 落库。
  python batch_operation/_repair_fallback.py < lost_paths.txt
  python batch_operation/_repair_fallback.py --apply < lost_paths.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import common as C

log = C.setup_logging(
    "repair_fallback", "batch_operation/reports/repair_fallback.log"
)

AUTHOR_TOPS = ("Authors", "OneDrive", "Supplementary Books")


def target_name(path: str, known: dict) -> str | None:
    seg = path.split("/")
    if len(seg) >= 3 and seg[0] in AUTHOR_TOPS:
        return f"书库--{seg[1]}"
    if len(seg) >= 2:
        parent = seg[-2]
        if parent in known:
            return parent
        for n in known:
            if n.endswith("--" + parent):
                return n
        return parent
    return None


async def run(apply: bool = False) -> dict:
    paths = [l.strip() for l in sys.stdin if l.strip()]
    conn = await C.get_conn()
    try:
        known = {r["name"]: r["id"] for r in await conn.fetch(
            "SELECT id, name FROM dataset")}
        zero, ok = [], 0
        for i in range(0, len(paths), 30000):
            chunk = paths[i:i + 30000]
            for r in await conn.fetch("""
                SELECT f.path, count(fd.dataset_id) AS n
                FROM file_metadata f
                LEFT JOIN file_dataset fd ON fd.file_id = f.id
                WHERE f.path = ANY($1)
                GROUP BY f.path, f.id HAVING count(fd.dataset_id) = 0""",
                chunk):
                zero.append(r["path"])
        plan = []
        for path in zero:
            name = target_name(path, known)
            plan.append({"path": path, "dataset": name})
            if apply and name:
                if name not in known:
                    rid = await conn.fetchval(
                        "INSERT INTO dataset (name) VALUES ($1) "
                        "ON CONFLICT (name) DO NOTHING RETURNING id", name)
                    known[name] = rid or (await conn.fetchval(
                        "SELECT id FROM dataset WHERE name = $1", name))
                await conn.execute(
                    "INSERT INTO file_dataset (file_id, dataset_id) "
                    "SELECT f.id, $2 FROM file_metadata f WHERE f.path = $1 "
                    "ON CONFLICT DO NOTHING", path, known[name])
                ok += 1
        if apply:
            still = 0
            for i in range(0, len(paths), 30000):
                for r in await conn.fetch("""
                    SELECT count(*) AS c FROM (
                      SELECT f.id FROM file_metadata f
                      LEFT JOIN file_dataset fd ON fd.file_id=f.id
                      WHERE f.path = ANY($1) GROUP BY f.id
                      HAVING count(fd.dataset_id) =0) t""", paths[i:i + 30000]):
                    still += r["c"]
        else:
            still = len(zero)
        summary = {"input_paths": len(paths), "zero_now": len(zero),
                   "bound": ok, "still_zero": still,
                   "unresolvable": [p for p in plan if not p["dataset"]],
                   "by_top": {}}
        for p in zero:
            t = p.split("/")[0]
            summary["by_top"][t] = summary["by_top"].get(t, 0) + 1
        summary["targets"] = {}
        for p in plan:
            if p["dataset"]:
                summary["targets"][p["dataset"]] = \
                    summary["targets"].get(p["dataset"], 0) + 1
        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/repair_fallback.json", "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        log.info("[summary] %s", json.dumps(
            {k: v for k, v in summary.items() if k != "targets"},
            ensure_ascii=False))
        return summary
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    import asyncio
    asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    main()
