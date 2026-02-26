import asyncio
import datetime
import logging
import os
import tempfile
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Body, Depends
from sqlalchemy import TIMESTAMP, cast, exists, func, select, text, update, delete
from sqlalchemy.dialects.postgresql import insert

from .app import get_current_admin
from .config import config
from .storage import storage
from .models import (
    Dataset,
    EmbeddingPendingQueue,
    Paragraph,
    TaskDBO,
    Terms,
    TextEmbeddings,
    get_db_session,
    QueryFilters,
)


class MaintenanceManager:

    async def sync_sources(self, dataset: str = "", folder: str = ""):
        assert dataset or folder, "Must specify at least one condition"

        async with get_db_session() as session:
            query = select(Paragraph)
            if folder:
                query = query.filter(Paragraph.source_url.startswith(folder))
            if dataset:
                query = query.filter(
                    Paragraph.dataset.in_(
                        select(Dataset)
                        .filter(
                            (Dataset.name == dataset)
                            | Dataset.name.startswith(dataset + "--")
                        )
                        .with_only_columns(Dataset.id)
                    )
                )

            query = query.distinct(Paragraph.source_url).with_only_columns(
                Paragraph.source_url
            )
            sources : list[str] = [_ for _, in await session.execute(query)]
            datasets = []

            for source in sources:
                if '://' in source: continue
                if '#' in source: source = source[:source.find('#')]
                joined = storage.safe_join(source)
                if not os.path.exists(joined):
                    logging.info(
                        f"{source} does not exist any more. Mark related paragraphs."
                    )
                    datasets.extend(
                        (
                            await session.execute(
                                select(Paragraph)
                                .filter(Paragraph.source_url == source)
                                .distinct(Paragraph.dataset)
                                .with_only_columns(Paragraph.dataset)
                            )
                        )
                        .scalars()
                        .all()
                    )
                    await session.execute(
                        update(Paragraph)
                        .filter(Paragraph.source_url == source)
                        .values(
                            extdata=Paragraph.extdata.op("||")(
                                func.json_build_object("offline", True)
                            )
                        )
                    )

            stmt = delete(Dataset).filter(
                ~exists().where(Paragraph.dataset == Dataset.id),
                Dataset.id.in_(list(set(datasets))),
            )
            await session.execute(stmt)

    async def sync_terms(self):
        async with get_db_session() as session:
            unnested_query = (
                select(text("unnest(keywords)").label("word"))
                .select_from(text("paragraph"))
                .subquery()
            )

            # Select unique words that aren't null or empty
            distinct_words_stmt = (
                select(unnested_query.c.word)
                .distinct()
                .where(unnested_query.c.word != None)
                .where(unnested_query.c.word != "")
            )

            result = await session.execute(distinct_words_stmt)
            unique_words = result.scalars().all()

            if not unique_words:
                logging.info("No keywords found to sync.")
                return

            # Perform a Bulk Upsert into the Terms table
            # We use on_conflict_do_nothing to ignore words already in the Terms table
            data = [{"term": w} for w in unique_words]

            insert_stmt = insert(Terms).values(data)
            upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["term"])

            await session.execute(upsert_stmt)
            logging.info(
                f"Sync complete. Processed {len(unique_words)} unique keywords."
            )

    async def merge_datasets(self):

        import re
        import unicodedata

        def normalize_string(s: str) -> str:
            """标准化字符串：去变音符号、去点、处理缩写"""
            if not s:
                return ""

            # 分解变音符号并去除 (例如: é -> e)
            s = unicodedata.normalize("NFD", s)
            s = "".join([c for c in s if unicodedata.category(c) != "Mn"])

            # 将 ". " 替换为空格，处理末尾的点
            s = re.sub(r"\. +", " ", s)

            s = re.sub(r"\s+", " ", s).strip().lower()

            return s

        def assess_similarity(str1: str, str2: str):
            """评估两个字符串是否相同或可能是同一人"""
            norm1 = normalize_string(str1)
            norm2 = normalize_string(str2)

            # 情况 1：完全相同（标准化后）
            if norm1.lower() == norm2.lower():
                return "MATCH", norm1

            # 情况 2：检测缩写匹配 (例如 Immanuel Kant vs I Kant)
            # 我们将字符串拆分为单词，检查是否符合“首字母 + 姓”的模式
            parts1 = norm1.split()
            parts2 = norm2.split()

            if len(parts1) == len(parts2) and len(parts1) > 1:
                is_potential = True
                for p1, p2 in zip(parts1, parts2):
                    p1l, p2l = p1.lower(), p2.lower()
                    # 如果一个是另一个的首字母，或者两者完全相同，则视为潜在匹配
                    if not (
                        p1l == p2l
                        or (len(p1) == 1 and p2l.startswith(p1l))
                        or (len(p2) == 1 and p1l.startswith(p2l))
                    ):
                        is_potential = False
                        break
                if is_potential:
                    return "POTENTIAL", f"'{str1}' and '{str2}' might be the same."

            return "DIFFERENT", None

        async with get_db_session() as session:
            ds = (await session.execute(select(Dataset))).scalars().all()
            calibre = [
                d for d in ds if d.name.startswith("书库--") or "--" not in d.name
            ]
            noncalibre = [d for d in ds if d not in calibre]
        for d1 in calibre:
            n1 = d1.name.split("--")[-1]
            for d2 in noncalibre:
                n2 = d2.name.split("--")[-1]
                cmp, _ = assess_similarity(n2, n1)
                if cmp == "MATCH":
                    await d1.rename_dataset(d2.name)
                    logging.info(f"[{d1.name}] merged to [{d2.name}]")
                    break

    async def update_author_from_url(self, pattern):
        async with get_db_session() as session:
            sql = text(
                """
                WITH extracted_data AS (
                    SELECT id, substring(source_url FROM :p) AS extracted_author
                    FROM paragraph
                    WHERE source_url ~* :p
                )
                UPDATE paragraph
                SET author = extracted_data.extracted_author
                FROM extracted_data
                WHERE paragraph.id = extracted_data.id
                AND (paragraph.author IS NULL OR paragraph.author = '')
            """
            )

            result = await session.execute(sql, {"p": pattern})
            logging.info(f"Successfully updated {result.rowcount} records.")

    async def update_pdate_from_url(self, dataset: str):
        async with get_db_session() as session:
            dataset_id_subquery = (
                select(Dataset.id).where(Dataset.name == dataset).scalar_subquery()
            )

            reg = r"(18|19|20)\d{2}"

            q = (
                select(
                    Paragraph.id,
                    (
                        func.regexp_matches(Paragraph.source_url, reg)[1] + "-01-01"
                    ).label("pdate_str"),
                )
                .where(Paragraph.dataset == dataset_id_subquery)
                .where(Paragraph.source_url.op("~*")(reg))
                .cte("q")
            )

            # 执行 UPDATE ... FROM
            stmt = (
                update(Paragraph)
                .where(Paragraph.id == q.c.id)
                .values(pdate=cast(q.c.pdate_str, TIMESTAMP))
            )

            await session.execute(stmt)

    async def update_text_embeddings(self, filters: Optional[QueryFilters] = None):

        if isinstance(filters, dict):
            filters = QueryFilters(**filters)

        if filters:
            stmt = await Paragraph.build_query(filters)
        else:
            stmt = select(Paragraph)

        stmt = (
            stmt.join(
                EmbeddingPendingQueue,
                (EmbeddingPendingQueue.id == Paragraph.id)
                & (EmbeddingPendingQueue.dataset == Paragraph.dataset),
            )
            .with_only_columns(Paragraph.id, Paragraph.dataset, Paragraph.content)
            .limit(10000)
        )

        added = 0

        while True:

            async with get_db_session() as session:
                results = await session.execute(stmt)
                results = results.mappings().all()

            results_len = len(results)

            if not results_len:
                return added

            new_bulk = []
            for i, p in enumerate(results, start=1):
                for chunk_id, chunk in enumerate(
                    TextEmbeddings.get_chunks(p["content"], 200, 50), start=1
                ):
                    new_bulk.append(
                        {
                            "id": str(p["id"]),
                            "dataset": str(p["dataset"]),
                            "chunk_id": chunk_id,
                            "content": chunk,
                        }
                    )
                if i % 320 == 0 or i == results_len:
                    added += await self.update_text_embeddings_do(bulk=new_bulk)
                    new_bulk.clear()

    async def update_text_embeddings_do(self, bulk: list):
        embs = []
        for p, emb in zip(
            bulk,
            await TextEmbeddings.batch_encode([p["content"] for p in bulk]),
        ):
            embs.append(
                TextEmbeddings(
                    id=p["id"],
                    dataset=p["dataset"],
                    chunk_id=p["chunk_id"],
                    embedding=emb,
                )
            )
        async with get_db_session() as session:
            session.add_all(embs)
        logging.info(f"{len(embs)} added")
        return len(embs)

    async def custom_task(self, task_id: str = "", **params):
        from .task import Task

        if task_id:
            dbo = await TaskDBO.get(task_id)
            assert dbo
            dbo.last_run = datetime.datetime.now()
            async with get_db_session() as session:
                await session.merge(dbo)
                await session.commit()
        else:
            dbo = TaskDBO(**params)
        task = Task.from_dbo(dbo)
        return await task.execute_async()

    async def ocr(
        self, input_path: str, output_path: str, lang: str, monochrome: bool = False
    ):
        from .pdfutils import convert_pdf_to_tiff_group4, merge_images_from_folder

        temps = []
        input_path = storage.safe_join(input_path)

        if os.path.isdir(input_path):
            fo = tempfile.NamedTemporaryFile("wb", delete=False)
            images = merge_images_from_folder(input_path, fo)
            fo.close()
            temps.append(fo.name)

            if not images:
                os.unlink(fo.name)
                for fn in storage.glob(os.path.join(input_path, "*.pdf")):
                    await self.ocr(
                        input_path=fn,
                        output_path=fn[:-4] + "_ocred.pdf",
                        lang=lang,
                        monochrome=monochrome,
                    )
                return
            input_path = fo.name

        if monochrome:
            fo = tempfile.NamedTemporaryFile("wb", delete=False)
            with open(input_path, "rb") as fi:
                convert_pdf_to_tiff_group4(fi, fo)
            fo.close()
            input_path = fo.name
            temps.append(fo.name)

        output_path = storage.safe_join(output_path)
        if output_path.endswith("/"):
            output_path += os.path.basename(input_path).rsplit(".", 1)[0] + "_ocred"
        if not output_path.endswith(".pdf"):
            output_path += ".pdf"

        try:
            import ocrmypdf

            await asyncio.to_thread(
                ocrmypdf.ocr,
                input_path,
                output_path,
                plugins=["ocrmypdf_paddleocr_remote"],
                language=lang,
                paddle_remote=config.paddle_remote,
                jobs=2,
                force_ocr=True,
            )
        finally:
            for f in temps:
                if os.path.exists(f):
                    os.unlink(f)
        return output_path

    async def test_task(self):
        logging.info("Test Task Started")
        await asyncio.sleep(10)
        logging.info("Test Task Ended")

    def get_router(self):

        router = APIRouter(
            prefix="/maintenance",
            tags=["Maintenance"],
            dependencies=[Depends(get_current_admin)],
        )

        @router.put("/{task_name}")
        async def call_task_in_background(
            task_name: str, background_tasks: BackgroundTasks, params: dict = Body(...)
        ):
            func = {
                "sync-pdate": self.update_pdate_from_url,
                "sync-terms": self.sync_terms,
                "text-embeddings": self.update_text_embeddings,
                "ocr": self.ocr,
                "custom": self.custom_task,
                "sync-sources": self.sync_sources,
            }.get(task_name)
            if func:
                background_tasks.add_task(func, **params)
                return {
                    "message": "Maintenance task started in background",
                    "task_name": task_name,
                    "params": params,
                }
            else:
                return {
                    "error": "Maintenance task not found",
                    "task_name": task_name,
                    "params": params,
                }

        return router


maintenance_manager = MaintenanceManager()
