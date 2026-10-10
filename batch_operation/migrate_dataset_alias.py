"""dataset 表新增「别名」栏位 alias_of，并把 古代/近代/现当代 的二级数据集
登记为 书库 下同名数据集的别名（2026-10-10 用户指令）。

语义（用户原话）：选中「现当代」= 选中其下所有二级数据集，而它们分别等同于
查询书库下的对应数据集 —— 即 `现当代--郭沫若` 是 `书库--郭沫若` 的**别名**，
查询时双向展开（别名 ↔ 规范）。

  ALTER TABLE dataset ADD COLUMN alias_of uuid REFERENCES dataset(id) ON DELETE SET NULL
  UPDATE dataset SET alias_of = 书库--<同名>.id   -- 古代--/近代--/现当代--

dry-run 默认；--apply 落库（幂等，可重复执行）。
列改动用 raw SQL（asyncpg），不依赖 ORM 模型，可在代码上线前先跑。
"""
from __future__ import annotations

import argparse
import json
import os

import common as C

log = C.setup_logging(
    "migrate_dataset_alias", "batch_operation/reports/migrate_dataset_alias.log"
)

ERA_PREFIX = ("古代--", "近代--", "现当代--")
ADD_COL_SQL = """
    ALTER TABLE dataset ADD COLUMN IF NOT EXISTS alias_of uuid
        REFERENCES dataset(id) ON DELETE SET NULL
"""
ADD_IDX_SQL = """
    CREATE INDEX IF NOT EXISTS dataset_alias_of_idx ON dataset (alias_of)
"""


def suffix_of(name: str) -> str:
    """'近代--汉文小说--韩国' -> '汉文小说--韩国'（只切第一个 --）。"""
    return name.split("--", 1)[1] if "--" in name else ""


async def run(apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        has_col = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_name='dataset' AND column_name='alias_of')")
        rows = await conn.fetch("SELECT id, name FROM dataset")
        by_name = {r["name"]: r["id"] for r in rows}

        plan, missing_canon = [], []
        for r in rows:
            if not r["name"].startswith(ERA_PREFIX):
                continue
            canon_name = "书库--" + suffix_of(r["name"])
            if canon_name not in by_name:
                missing_canon.append({"alias": r["name"], "canonical": canon_name})
                continue
            plan.append({"alias": r["name"], "canonical": canon_name,
                         "alias_id": str(r["id"]),
                         "canonical_id": str(by_name[canon_name])})

        if apply:
            await conn.execute(ADD_COL_SQL)
            await conn.execute(ADD_IDX_SQL)
            if missing_canon:      # 规范数据集不存在则补建，别名必须有落点
                await conn.executemany(
                    "INSERT INTO dataset (name) VALUES ($1) "
                    "ON CONFLICT (name) DO NOTHING",
                    [(m["canonical"],) for m in missing_canon])
                for r in await conn.fetch("SELECT id, name FROM dataset"):
                    by_name.setdefault(r["name"], r["id"])
                for m in missing_canon:
                    plan.append({"alias": m["alias"], "canonical": m["canonical"],
                                 "alias_id": None, "canonical_id": None})
            n = await conn.execute(
                "UPDATE dataset d SET alias_of = c.id FROM dataset c "
                "WHERE c.name = '书库--' || substr(d.name, strpos(d.name,'--')+2) "
                "  AND (d.name LIKE '古代--%' OR d.name LIKE '近代--%' "
                "       OR d.name LIKE '现当代--%') "
                "  AND d.alias_of IS DISTINCT FROM c.id")
            log.info("[apply] %s", n)

        # 读回核验
        check = {}
        if has_col or apply:
            check = {
                "alias_total": await conn.fetchval(
                    "SELECT count(*) FROM dataset WHERE alias_of IS NOT NULL"),
                "era_with_alias": await conn.fetchval(
                    "SELECT count(*) FROM dataset WHERE "
                    "(name LIKE '古代--%' OR name LIKE '近代--%' "
                    " OR name LIKE '现当代--%') AND alias_of IS NOT NULL"),
                "era_total": await conn.fetchval(
                    "SELECT count(*) FROM dataset WHERE name LIKE '古代--%' "
                    "OR name LIKE '近代--%' OR name LIKE '现当代--%'"),
                "bad_target": await conn.fetchval(
                    "SELECT count(*) FROM dataset d JOIN dataset c "
                    "ON c.id = d.alias_of WHERE c.name NOT LIKE '书库--%'"),
                "chain": await conn.fetchval(
                    "SELECT count(*) FROM dataset WHERE alias_of IS NOT NULL "
                    "AND name LIKE '书库--%'"),
                "sample": [r["name"] for r in await conn.fetch(
                    "SELECT d.name, c.name AS canon FROM dataset d "
                    "JOIN dataset c ON c.id=d.alias_of WHERE d.alias_of IS NOT NULL "
                    "ORDER BY d.name LIMIT 5")],
            }
        summary = {"applied": apply, "column_existed": has_col,
                   "plan": len(plan), "missing_canonical_created": missing_canon,
                   "check": check, "plan_sample": plan[:5]}
        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/migrate_dataset_alias.json", "w",
                  encoding="utf-8") as f:
            json.dump({"summary": summary, "plan": plan}, f,
                      ensure_ascii=False, indent=1)
        log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
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
