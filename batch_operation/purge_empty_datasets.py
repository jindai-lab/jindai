"""清理空数据集（2026-10-20 用户指令：空 dataset 删除，别名除外）。

判空 = 两条绑定路径都为 0：`file_dataset` 0 且 `file_metadata.dataset_id` 0。
保留（不删）三类：
  ① 是别名（alias_of 非空）；
  ② 被别名指向（删它会让别名 ON DELETE SET NULL 变孤儿）；
  ③ 名字出现在 task_dbo.pipeline 里（任务引用，如「其他」）。

Dry-run 默认；--apply 删除。报表 reports/empty_datasets_cleanup.json。
"""
import asyncio, json, sys, logging, argparse

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
logging.disable(logging.CRITICAL)
import common as C


async def main(apply=False):
    conn = await C.get_conn()
    ds = await conn.fetch("SELECT id, name, alias_of FROM dataset ORDER BY name")
    print("datasets total:", len(ds))
    pipes = [r["p"] for r in await conn.fetch("SELECT pipeline::text AS p FROM task_dbo")]

    todo, kept = [], []
    for d in ds:
        fd = await conn.fetchval("SELECT count(*) FROM file_dataset WHERE dataset_id=$1::uuid", d["id"])
        md = await conn.fetchval("SELECT count(*) FROM file_metadata WHERE dataset_id=$1::uuid", d["id"])
        if fd or md:
            continue
        has_alias = await conn.fetchval("SELECT count(*) FROM dataset WHERE alias_of=$1::uuid", d["id"])
        reasons = []
        if d["alias_of"] is not None:
            reasons.append("是别名")
        if has_alias:
            reasons.append("被%d个别名指向" % has_alias)
        if any(d["name"] and d["name"] in p for p in pipes):
            reasons.append("任务引用")
        (kept.append((d["name"], reasons)) if reasons else todo.append((str(d["id"]), d["name"])))

    print("EMPTY total: %d | to delete: %d | kept: %d"
          % (len(todo) + len(kept), len(todo), len(kept)))
    for n, r in kept:
        print("  KEEP(空):", n, "|", ";".join(r))
    for _, n in todo:
        print("  DEL:", n)

    json.dump({"deleted": [x[1] for x in todo],
               "kept": [{"name": n, "why": r} for n, r in kept]},
              open("/app/batch_operation/reports/empty_datasets_cleanup.json", "w"),
              ensure_ascii=False, indent=1)

    if apply and todo:
        tag = await conn.execute(
            "DELETE FROM dataset WHERE id = ANY($1::uuid[]) AND name = ANY($2::text[])",
            [x[0] for x in todo], [x[1] for x in todo])
        print("DELETED:", tag)
        left = await conn.fetchval("SELECT count(*) FROM dataset")
        print("datasets remaining:", left)
    elif not apply:
        print("DRY-RUN (no --apply)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(apply=a.apply))
