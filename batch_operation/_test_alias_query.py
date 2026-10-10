"""别名等价查询的端到端验证（一次性）。"""
import asyncio
import sys

sys.path.insert(0, "/app")

from sqlalchemy import func, select

from jindai.models import (
    Dataset,
    FileDataset,
    FileMetadata,
    Paragraph,
    QueryFilters,
    get_db_session,
)


async def files_of(names, prefix=True):
    ids = await Dataset.resolve_ids(names, prefix)
    async with get_db_session() as s:
        n = (await s.execute(
            select(func.count(func.distinct(FileDataset.file_id)))
            .where(FileDataset.dataset_id.in_(ids)))).scalar()
        rows = (await s.execute(
            select(Dataset.name).where(Dataset.id.in_(ids)))).scalars().all()
    return n, sorted(rows)


async def para_count(names):
    q = await Paragraph.build_query(QueryFilters(datasets=names))
    async with get_db_session() as s:
        return (await s.execute(
            select(func.count()).select_from(q.subquery()))).scalar()


async def main():
    # 1) 选中「现当代」（分组）= 其下全部二级 + 书库对应规范集
    n, names = await files_of(["现当代"])
    print(f"① 现当代(分组) -> {len(names)} 个数据集, {n} 个文件")
    print("   规范集样例:", [x for x in names if x.startswith("书库--")][:4])
    print("   别名样例:", [x for x in names if not x.startswith("书库--")][:4])

    # 2) 单个别名：应等同于其规范数据集
    for alias in ("现当代--鲁迅", "现当代--丁玲", "近代--王国维"):
        na, na_names = await files_of([alias], prefix=False)
        canon = "书库--" + alias.split("--", 1)[1]
        nc, nc_names = await files_of([canon], prefix=False)
        ok = set(na_names) == set(nc_names)
        print(f"② {alias}: 展开后 {len(na_names)} 个数据集 {na_names} | "
              f"文件 {na}（规范集文件 {nc}）| 与规范集等价={ok}")

    # 3) 段落检索口径（用户实际查询路径）
    for names in (["现当代--鲁迅"], ["书库--鲁迅"], ["现当代"]):
        c = await para_count(names)
        print(f"③ 段落检索 {names} -> {c} 段")

    # 4) 无别名的普通数据集不受影响
    n, names = await files_of(["Zotero"], prefix=False)
    print(f"④ Zotero(无别名) -> {len(names)} 个数据集, {n} 个文件")


asyncio.run(main())
