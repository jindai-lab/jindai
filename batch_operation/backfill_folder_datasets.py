"""按顶层文件夹规则重建 file_dataset 数据集绑定（2026-10-10 用户指令）。

规则（定稿，含用户两次更正：Series 沿用「特辑--」前缀；OneDrive/Supplementary
不保留「书库」基础绑定——查询书库时已含全部下级数据集）：

  Zotero/*                       -> Zotero
  Journals/<二级目录>/...         -> 报刊--<二级目录>
  Journals/<刊名>.pdf（根级 22 个）-> 报刊--<文件名去后缀>
  Series/<二级目录>/...           -> 特辑--<二级目录>
  OneDrive/<作者>/.../f           -> 书库--<作者>（3 层/4 层同规则）
  Supplementary Books/<作者>/f    -> 书库--<作者>
  Supplementary Books/<f>（根级 42）-> 书库--<推定作者>（SUPP_ROOT_AUTHORS 表）

语义 = **替换**：目标文件不在允许集合内的既有绑定一律删除，缺失的补上；
不可解析的文件整只跳过、只报告，不改动（现应为 0 个）。

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

# Supplementary Books 根级（无作者目录）42 个 PDF 的推定作者（2026-10-10 用户指令：
# 「根据文本内容或搜索来推测作者」）。依据 = 题名页文本（pymupdf 提取）+ 文件名书目
# 信息 + 必要时的网络查证；逐条见 reports/supp_root_authors_evidence.md。
SUPP_ROOT_AUTHORS = {
    "A chinese child's day.By Anice Terhune.插画.Albertine Randall Wheelan.1910年.pdf": "Anice Terhune",
    "Chinese_Researches.pdf": "Alexander Wylie",
    "Conic_Sections.pdf": "William Whewell",
    "Illustrations of China and its people.中国与中国人影像.4册.By John Thomson.1874年.pdf": "John Thomson",
    "Memorial Wylie.pdf": "Alexander Wylie",
    "Shanghai Country Walks.pdf": "E. S. Wilkinson",
    "The_treaty_ports_of_China_and_Japan_a_gu.pdf": "N. B. Dennys",
    "hangchow itineraries.pdf": "Robert F. Fitch",
    "万寿盛典初集.版画部分.卷41至卷42.王原祁等纂修.清康熙56年内府刻本.1717年.pdf": "王原祁",
    "三国志.65卷.附考证.陈寿撰.裴松之注.清乾隆4年奉敕考证.武英殿校刊本.pdf": "陈寿",
    "人民周刊_1926年2月——1927年4月.pdf": "张太雷",
    "光绪元年致日本开拓判官照会册.直隶津海关道.清光绪元年稿本.1875年.pdf": "直隶津海关道",
    "八旬万寿盛典图说.2卷.乾隆57年.武英殿套印本.pdf": "乾隆",
    "六朝事迹编类.14卷.宋张敦颐撰.宝章阁藏版.光绪13年仿宋绍兴建康府学本.pdf": "张敦颐",
    "北京各行祖师调查记略.pdf": "崇璋",
    "古先君臣图鉴.附小传古赞.明代.pdf": "佚名",
    "吕氏春秋.漢.高誘註.pdf": "吕不韦",
    "吴郡名贤图传赞.20卷.清顾沅撰.孔继尧绘.清道光9年长洲顾氏刊本.1829年.pdf": "顾沅",
    "回忆录_罗宗洛.pdf": "罗宗洛",
    "大义觉迷录.4卷.清.雍正编.清雍正时期刊本.pdf": "雍正",
    "奇姓通.14卷.明夏树芳辑.陈继儒校.明天启四年夏氏宛委堂刊本.pdf": "夏树芳",
    "战国策.12策.闵齐伋裁注集评.乌程闵齐伋.明万历47年三色套印本.pdf": "刘向",
    "支那通史.卷1-3.那珂通世编.明治时刊.pdf": "那珂通世",
    "无双谱.清.金古良编绘.朱圭刻.清康熙33年刊.1694年.pdf": "金古良",
    "明状元图考.3卷.明顾祖训辑.汉阳叶氏平安馆藏本.明万历37年.1609年.pdf": "顾祖训",
    "最新中国历史教科书.高等小学用.姚祖义编.4册.上海商务印书馆.宣统元年版.1909年.pdf": "姚祖义",
    "浙江图书馆丛书.pdf": "浙江图书馆",
    "清异录.2卷.宋陶谷撰.清李锡龄校.惜阴轩丛书.pdf": "陶谷",
    "皇清职贡图.九卷.附诸臣恭和诗二卷.清傅恒等纂.门庆安等绘.乾隆16年编成.嘉庆10年增修.内府刊本.pdf": "傅恒",
    "诸史将略.十六卷.明刘畿纂辑.吴承恩等编.明嘉靖45年杭州毛钢刊本.pdf": "刘畿",
    "诸葛忠武侯年谱.附.木牛流马考略.八阵图考略.杨希闵编.光绪4年.pdf": "杨希闵",
    "镌重订补注历朝捷录史鉴提衡.4卷.首1卷.靖难记略.1卷.顾充辑著.李廷机重订补注.明万历建邑书林熊冲宇刻本.pdf": "顾充",
}
for _n in range(1, 11):          # 元和郡县图志 1–10 卷 → 唐·李吉甫
    SUPP_ROOT_AUTHORS[f"元和郡县图志{_n}.pdf"] = "李吉甫"


def desired_datasets(path: str, authors: list) -> list[str] | None:
    """按规则算出该文件应属的数据集名；不可解析返回 None（跳过、不改动）。"""
    seg = path.split("/")
    top = seg[0]
    authors = [a for a in (authors or []) if a]
    if top == "Zotero":
        return ["Zotero"]
    if top == "Journals":
        if len(seg) >= 3:
            return [f"报刊--{seg[1]}"]
        # 根级散刊：直接以文件名（不含后缀）作刊名（2026-10-10 用户指令）
        if len(seg) == 2:
            stem = seg[1]
            for ext in (".pdf", ".PDF"):
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
            return [f"报刊--{stem}"]
        return None
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
        # 根级 42 个：用文本/查证推定的作者表
        if len(seg) == 2 and seg[1] in SUPP_ROOT_AUTHORS:
            return [f"书库--{SUPP_ROOT_AUTHORS[seg[1]]}"]
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
