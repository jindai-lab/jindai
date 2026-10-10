"""按顶层文件夹规则重建 file_dataset 数据集绑定（2026-10-10 用户指令）。

规则（定稿，含用户两次更正：Series 沿用「特辑--」前缀；OneDrive/Supplementary
不保留「书库」基础绑定——查询书库时已含全部下级数据集）：

  Zotero/*                       -> Zotero
  Journals/<二级目录>/...         -> 报刊--<二级目录>
  Series/<二级目录>/...           -> 特辑--<二级目录>
  OneDrive/<作者>/<书名 (id)>/f   -> 书库--<作者>（3 层 <作者>/f 同规则）
  Supplementary Books/<作者>/f    -> 书库--<作者>
  Supplementary Books/<f>         -> 书库--<authors[0]>（目前 authors 全空 → 不可解析）

语义 = **替换**：目标文件不在允许集合内的既有绑定一律删除，缺失的补上；
不可解析的文件（缺作者/刊名目录且 authors 为空）整只跳过、只报告，不改动。

  python batch_operation/backfill_folder_datasets.py            # dry-run
  python batch_operation/backfill_folder_datasets.py --apply
  python batch_operation/backfill_folder_datasets.py --apply --limit 50

改前先备份：reports/dataset_bindings_backup_20261010.tsv（path \t dataset name）。
"""
from __future__ import annotations

import argparse
import json
import os

import common as C

log = C.setup_logging(
    "backfill_folder_datasets", "batch_operation/reports/backfill_folder_datasets.log"
)

SCOPE_TOPS = ("Zotero", "Journals", "Series", "OneDrive", "Supplementary Books")
SCOPE_SQL = """
    SELECT f.id, f.path, coalesce(f.authors, '{}') AS authors
    FROM file_metadata f
    WHERE lower(f.path) LIKE '%.pdf'
      AND f.path NOT LIKE '../%'
      AND f.path NOT LIKE '.trash/%'
      AND split_part(f.path, '/', 1) = ANY($1)
"""
BIND_SQL = """
    SELECT fd.file_id, d.name
    FROM file_dataset fd
    JOIN dataset d ON d.id = fd.dataset_id
    WHERE fd.file_id = ANY($1)
"""


def desired_datasets(path: str, authors: list) -> list[str] | None:
    """按规则算出该文件应属的数据集名；不可解析返回 None（跳过、不改动）。"""
    seg = path.split("/")
    top = seg[0]
    authors = [a for a in (authors or []) if a]
    if top == "Zotero":
        return ["Zotero"]
    if top == "Journals":
        return [f"报刊--{seg[1]}"] if len(seg) >= 3 else None
    if top == "Series":
        return [f"特辑--{seg[1]}"] if len(seg) >= 3 else None
    if top == "OneDrive":
        # 4 层 <作者>/<书名 (id)>/f 与 3 层 <作者>/f 的第 2 段都是作者目录
        if len(seg) >= 3:
            return [f"书库--{seg[1]}"]
        return [f"书库--{authors[0]}"] if authors else None
    if top == "Supplementary Books":
        if len(seg) == 3:
            return [f"书库--{seg[1]}"]
        return [f"书库--{authors[0]}"] if authors else None
    return None


async def run(limit: int = 0, apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(SCOPE_SQL, list(SCOPE_TOPS))
        if limit:
            rows = rows[:limit]
        files = {r["id"]: r for r in rows}
        bindings: dict = {}
        ids = list(files)
        for i in range(0, len(ids), 30000):
            for r in await conn.fetch(BIND_SQL, ids[i:i + 30000]):
                bindings.setdefault(r["file_id"], set()).add(r["name"])

        plan_add, plan_del, unresolved, skip_ok = [], [], [], 0
        want_names: set[str] = set()
        for fid, r in files.items():
            want = desired_datasets(r["path"], list(r["authors"]))
            if not want:
                unresolved.append({"path": r["path"],
                                   "bindings": sorted(bindings.get(fid, ()))})
                continue
            want_names.update(want)
            have = bindings.get(fid, set())
            for n in want:
                if n not in have:
                    plan_add.append((fid, n))
            for n in have - set(want):
                plan_del.append((fid, n))
            if not (set(want) ^ have):
                skip_ok += 1

        # 目标数据集行：先建行、再取 id
        if want_names and apply:
            await conn.executemany(
                "INSERT INTO dataset (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                [(n,) for n in sorted(want_names)],
            )
        ids_by_name = {}
        if want_names:
            for r in await conn.fetch(
                "SELECT id, name FROM dataset WHERE name = ANY($1)", sorted(want_names)
            ):
                ids_by_name[r["name"]] = r["id"]

        applied_add = applied_del = 0
        if apply and (plan_add or plan_del):
            if plan_add:
                await conn.executemany(
                    "INSERT INTO file_dataset (file_id, dataset_id) VALUES ($1, $2) "
                    "ON CONFLICT DO NOTHING",
                    [(fid, ids_by_name[n]) for fid, n in plan_add],
                )
                applied_add = len(plan_add)
            if plan_del:
                # 需要 dataset_id：用 name→id 反查（plan_del 里的 name 一定已存在）
                del_names = sorted({n for _, n in plan_del})
                del_ids = {
                    r["name"]: r["id"]
                    for r in await conn.fetch(
                        "SELECT id, name FROM dataset WHERE name = ANY($1)", del_names
                    )
                }
                await conn.executemany(
                    "DELETE FROM file_dataset WHERE file_id = $1 AND dataset_id = $2",
                    [(fid, del_ids[n]) for fid, n in plan_del],
                )
                applied_del = len(plan_del)

        # 读回核验：重新统计合规情况
        after = await conn.fetch(BIND_SQL, ids)
        after_map: dict = {}
        for r in after:
            after_map.setdefault(r["file_id"], set()).add(r["name"])
        mismatch = 0
        for fid, r in files.items():
            want = desired_datasets(r["path"], list(r["authors"]))
            if not want:
                continue
            if after_map.get(fid, set()) != set(want):
                mismatch += 1

        summary = {
            "applied": apply,
            "scope_files": len(files),
            "already_compliant": skip_ok,
            "to_add": len(plan_add),
            "to_delete": len(plan_del),
            "added": applied_add,
            "deleted": applied_del,
            "unresolved_skipped": len(unresolved),
            "post_mismatch": mismatch,
            "datasets_to_create": sorted(
                n for n in want_names
            ) if not ids_by_name else sorted(
                n for n in want_names if n not in ids_by_name
            ),
            "unresolved_sample": unresolved[:40],
        }
        # datasets_to_create 语义：apply 前是"将新建"，apply 后应为空
        if apply:
            summary["datasets_created"] = summary.pop("datasets_to_create")
        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/backfill_folder_datasets.json",
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info("[summary] %s", json.dumps(
            {k: v for k, v in summary.items() if k != "unresolved_sample"},
            ensure_ascii=False))
        for u in unresolved[:15]:
            log.info("[unresolved] %s (has: %s)", u["path"], u["bindings"])
        return summary
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import asyncio
    asyncio.run(run(limit=args.limit, apply=args.apply))


if __name__ == "__main__":
    main()
