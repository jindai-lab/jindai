"""数据集分类清理（2026-10-10 用户指令）。dry-run 默认，--apply 落库。

指令五条：
  A foreign   整体删除「外国文学」「外国哲学」及其所有二级数据集
  B journal   「期刊」归入「报刊」（期刊--X → 报刊--X）
  C era       「古代」「近代」「现当代」的二级名称也是作者 → 等价挂到
              「书库--<同名>」（确保存在并把成员一并挂上，保留原时代分类行）
  D xiandai   「现当代」中删除非文学家的人物（NON_LITERARY 名单）
  E teji      「特辑」已有的二级名称，从其他分类下删除（排除由文件夹派生的
              书库/报刊/Zotero/Zeno——删了导入器会按目录规则重建）
  F fallback  因删除而失去全部数据集的文件补挂：Authors/OneDrive/Supplementary
              等有作者段的 → 书库--<作者>；否则沿用 ensure_dataset 的父目录匹配

删行前先把成员挂到兜底数据集（F 在 C 之后算），任何文件都不会变成 0 数据集；
计划与被删数据集的成员文件全量写 reports/cleanup_datasets_plan.json 以便回滚。

  python batch_operation/cleanup_datasets.py            # dry-run
  python batch_operation/cleanup_datasets.py --apply
"""
from __future__ import annotations

import argparse
import json
import os

import common as C

log = C.setup_logging(
    "cleanup_datasets", "batch_operation/reports/cleanup_datasets.log"
)

# 「现当代」中判定为**非文学家**的人物（创作者/诗人/散文家/小说家/文学翻译家保留）
NON_LITERARY_XIANDAI = [
    "丁文江", "余英时", "傅斯年", "傅衣凌", "冯友兰", "冯契", "吕思勉", "周扬",
    "唐君毅", "宋任穷", "巫鸿", "张一兵", "徐复观", "朱光潜", "李大钊", "李泽厚",
    "李长之", "林沄", "梁思成", "段义孚", "汤一介", "汪民安", "熊十力", "竺可桢",
    "胡乔木", "胡适", "蒋孔阳", "蔡元培", "许煜", "贺麟", "辛丰年", "郑元者",
    "金观涛", "钟敬文", "钱学森", "钱玄同", "钱穆", "陈嘉映", "陈寅恪", "陈望衡",
    "陈望道", "陈独秀", "陈鹤琴", "顾维钧", "顾颉刚", "马一浮", "高亨",
]

FOREIGN_GROUPS = ("外国文学", "外国哲学")
ERA_GROUPS = ("古代", "近代", "现当代")
# E 动作只在这些「非文件夹派生」的分类下删同名二级；书库/报刊/Zotero/Zeno 由目录规则产生，跳过
NON_FOLDER_CATEGORIES = ("古代", "近代", "现当代", "外国文学", "外国哲学",
                         "期刊", "马克思主义", "其他", "")


def suffix(name: str) -> str:
    return name.split("--", 1)[1] if "--" in name else ""


async def run(apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        ds = {r["name"]: r["id"] for r in await conn.fetch(
            "SELECT id, name FROM dataset")}
        members: dict[str, list] = {}

        async def get_members(name: str) -> list:
            if name not in members:
                members[name] = [r["file_id"] for r in await conn.fetch(
                    "SELECT file_id FROM file_dataset WHERE dataset_id = $1",
                    ds[name])]
            return members[name]

        plan: dict = {"applied": apply, "actions": {}}
        deleted: dict[str, list] = {}          # 数据集名 -> 成员文件 id（回滚用）
        deleted_names: list[str] = []

        def mark(name: str) -> None:
            deleted_names.append(name)

        # ---------- C：时代分类 → 书库 同名等价（先做，保住成员） ----------
        era_mirrored = []
        for name in list(ds):
            if not name.startswith(tuple(f"{g}--" for g in ERA_GROUPS)):
                continue
            target = "书库--" + suffix(name)
            ids = await get_members(name)
            if target not in ds:
                if apply:
                    row = await conn.fetchval(
                        "INSERT INTO dataset (name) VALUES ($1) "
                        "ON CONFLICT (name) DO NOTHING RETURNING id", target)
                    ds[target] = row or (await conn.fetchval(
                        "SELECT id FROM dataset WHERE name = $1", target))
                else:
                    ds.setdefault(target, None)
            if ids and apply and ds.get(target):
                await conn.executemany(
                    "INSERT INTO file_dataset (file_id, dataset_id) VALUES ($1,$2) "
                    "ON CONFLICT DO NOTHING",
                    [(fid, ds[target]) for fid in ids])
            era_mirrored.append({"from": name, "to": target, "files": len(ids)})
        plan["actions"]["C_era_mirror"] = {
            "datasets": len(era_mirrored),
            "files_mirrored": sum(x["files"] for x in era_mirrored),
            "sample": era_mirrored[:5],
        }

        # ---------- D：现当代删非文学家 ----------
        d_targets = [f"现当代--{n}" for n in NON_LITERARY_XIANDAI
                     if f"现当代--{n}" in ds]
        d_files = 0
        for name in d_targets:
            ids = await get_members(name)
            d_files += len(ids)
            deleted[name] = ids
            mark(name)
        if apply and d_targets:
            await conn.executemany(
                "DELETE FROM file_dataset WHERE dataset_id = $1",
                [(ds[n],) for n in d_targets])
            await conn.executemany(
                "DELETE FROM dataset WHERE id = $1", [(ds[n],) for n in d_targets])
        for n in d_targets:
            ds.pop(n, None)
        plan["actions"]["D_xiandai_nonliterary"] = {
            "deleted": len(d_targets), "files_affected": d_files,
            "names": sorted(d_targets),
            "kept_nonliterate_not_found": sorted(
                set(f"现当代--{n}" for n in NON_LITERARY_XIANDAI) - set(d_targets)),
        }

        # ---------- A：外国文学 /外国哲学 整体删除 ----------
        a_targets = [n for n in ds
                     if n in FOREIGN_GROUPS or
                     any(n.startswith(f"{g}--") for g in FOREIGN_GROUPS)]
        a_files = 0
        for name in a_targets:
            ids = await get_members(name)
            a_files += len(ids)
            deleted[name] = ids
            mark(name)
        if apply and a_targets:
            await conn.executemany(
                "DELETE FROM file_dataset WHERE dataset_id = $1",
                [(ds[n],) for n in a_targets])
            await conn.executemany(
                "DELETE FROM dataset WHERE id = $1", [(ds[n],) for n in a_targets])
        for n in a_targets:
            ds.pop(n, None)
        plan["actions"]["A_foreign_deleted"] = {
            "datasets": len(a_targets), "file_bindings": a_files,
            "group_rows": sorted(n for n in a_targets if "--" not in n)}

        # ---------- B：期刊 → 报刊 ----------
        b_plan = []
        for name in [n for n in ds if n.startswith("期刊")]:
            target = "报刊--" + suffix(name) if "--" in name else "报刊"
            ids = await get_members(name)
            if target not in ds and apply:
                rid = await conn.fetchval(
                    "INSERT INTO dataset (name) VALUES ($1) "
                    "ON CONFLICT (name) DO NOTHING RETURNING id", target)
                ds[target] = rid or (await conn.fetchval(
                    "SELECT id FROM dataset WHERE name = $1", target))
            if apply and ids and ds.get(target):
                await conn.executemany(
                    "INSERT INTO file_dataset (file_id, dataset_id) VALUES ($1,$2) "
                    "ON CONFLICT DO NOTHING",
                    [(fid, ds[target]) for fid in ids])
                await conn.execute(
                    "DELETE FROM file_dataset WHERE dataset_id = $1", ds[name])
            if apply:
                await conn.execute("DELETE FROM dataset WHERE id = $1", ds[name])
            ds.pop(name, None)
            mark(name)
            b_plan.append({"from": name, "to": target, "files": len(ids)})
        plan["actions"]["B_journal_to_baokan"] = {"datasets": b_plan}

        # ---------- E：特辑 已有的二级名，从其他分类删除 ----------
        teji = {suffix(n) for n in ds if n.startswith("特辑--")}
        e_targets = [n for n in ds
                     if "--" in n and suffix(n) in teji
                     and n.split("--", 1)[0] in NON_FOLDER_CATEGORIES
                     and not n.startswith("特辑--")]
        e_files = 0
        e_todo = []          # 本轮真正要删的（A/D 已删的不重复处理）
        for name in e_targets:
            if name in deleted:
                continue
            ids = await get_members(name)
            e_files += len(ids)
            deleted[name] = ids
            mark(name)
            e_todo.append(name)
        if apply:
            for n in e_todo:
                await conn.execute(
                    "DELETE FROM file_dataset WHERE dataset_id = $1", ds[n])
            if e_todo:
                await conn.executemany(
                    "DELETE FROM dataset WHERE id = $1", [(ds[n],) for n in e_todo])
        for n in e_todo:
            ds.pop(n, None)
        plan["actions"]["E_teji_dedupe"] = {
            "deleted": len(e_todo), "files": e_files,
            "names": sorted(e_todo),
            "skipped_folder_derived": sorted(
                n for n in ds if "--" in n and suffix(n) in teji
                and n.split("--", 1)[0] in ("书库", "报刊", "Zotero", "Zeno")),
        }

        # ---------- F：兜底——0 数据集的文件 ----------
        gone = [n for n in deleted_names]
        need_ids: set = set()
        for n in gone:
            for fid in deleted.get(n, []):
                need_ids.add(fid)
        fallback = []
        if need_ids:
            for i in range(0, len(need_ids), 30000):
                chunk = sorted(need_ids)[i:i + 30000]
                for r in await conn.fetch("""
                    SELECT f.id, f.path, count(fd.dataset_id) AS n
                    FROM file_metadata f
                    LEFT JOIN file_dataset fd ON fd.file_id = f.id
                    WHERE f.id = ANY($1) GROUP BY f.id, f.path
                    HAVING count(fd.dataset_id) = 0""", chunk):
                    fallback.append((r["id"], r["path"]))
        fb_plan = []
        for fid, path in fallback:
            seg = path.split("/")
            name = None
            if len(seg) >= 3 and seg[0] in ("Authors", "OneDrive",
                                             "Supplementary Books"):
                name = f"书库--{seg[1]}"
            elif len(seg) >= 2:
                parent = seg[-2]
                match = [n for n in ds if n == parent or n.endswith("--" + parent)]
                name = match[0] if match else parent
            if not name:
                fb_plan.append({"path": path, "dataset": None})
                continue
            fb_plan.append({"path": path, "dataset": name})
            if apply:
                if name not in ds:
                    rid = await conn.fetchval(
                        "INSERT INTO dataset (name) VALUES ($1) "
                        "ON CONFLICT (name) DO NOTHING RETURNING id", name)
                    ds[name] = rid or (await conn.fetchval(
                        "SELECT id FROM dataset WHERE name = $1", name))
                await conn.execute(
                    "INSERT INTO file_dataset (file_id, dataset_id) VALUES ($1,$2) "
                    "ON CONFLICT DO NOTHING", fid, ds[name])
        plan["actions"]["F_fallback"] = {
            "files_needing": len(fallback),
            "unresolvable": [x for x in fb_plan if not x["dataset"]],
            "target_counts": {},
        }
        for x in fb_plan:
            if x["dataset"]:
                plan["actions"]["F_fallback"]["target_counts"][x["dataset"]] = \
                    plan["actions"]["F_fallback"]["target_counts"].get(
                        x["dataset"], 0) + 1

        # ---------- 读回核验 ----------
        if apply:
            left = await conn.fetchval(
                "SELECT count(*) FROM dataset WHERE name LIKE '外国%' "
                "OR name LIKE '期刊%'")
            zero = await conn.fetchval("""
                SELECT count(*) FROM (SELECT f.id FROM file_metadata f
                  LEFT JOIN file_dataset fd ON fd.file_id=f.id
                  GROUP BY f.id HAVING count(fd.dataset_id)=0) t""")
            plan["post"] = {"foreign_or_journal_left": left,
                            "files_with_zero_dataset": zero}
        plan["deleted_datasets"] = sorted(deleted_names)

        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/cleanup_datasets_plan.json", "w",
                  encoding="utf-8") as f:
            json.dump({"plan": plan,
                       "deleted_members": {k: [str(x) for x in v]
                                           for k, v in deleted.items()}},
                      f, ensure_ascii=False, indent=1)
        log.info("[summary] %s", json.dumps(
            {k: v for k, v in plan.items() if k != "deleted_datasets"},
            ensure_ascii=False)[:3000])
        log.info("[deleted] %d datasets", len(deleted_names))
        return plan
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
