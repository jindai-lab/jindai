from fastapi import APIRouter, BackgroundTasks, Body, Depends
from sqlalchemy import TIMESTAMP, cast, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .app import get_current_admin
from .models import Dataset, Paragraph, Terms, get_db_session


class MaintenanceManager:
    
    async def sync_terms(self):
        async with get_db_session() as session:
            unnested_query = select(
                text("unnest(keywords)").label("word")
            ).select_from(text("paragraph")).subquery()

            # 2. Select unique words that aren't null or empty
            distinct_words_stmt = (
                select(unnested_query.c.word)
                .distinct()
                .where(unnested_query.c.word != None)
                .where(unnested_query.c.word != "")
            )
            
            result = await session.execute(distinct_words_stmt)
            unique_words = result.scalars().all()

            if not unique_words:
                print("No keywords found to sync.")
                return

            # 3. Perform a Bulk Upsert into the Terms table
            # We use on_conflict_do_nothing to ignore words already in the Terms table
            data = [{"term": w} for w in unique_words]
            
            insert_stmt = insert(Terms).values(data)
            upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=['term'])

            await session.execute(upsert_stmt)
            print(f"Sync complete. Processed {len(unique_words)} unique keywords.")

    async def update_pdate_from_url(self, dataset, session):
        async with get_db_session() as session:
            dataset_id_subquery = (
                select(Dataset.id).where(Dataset.name == dataset).scalar_subquery()
            )

            reg = r"(18|19|20)\d{2}"

            # regexp_matches 在 PostgreSQL 中返回的是数组，所以需要索引 [1]
            # 在 SQLAlchemy 中可以使用 column_element.get_item(1)
            q = (
                select(
                    Paragraph.id,
                    (
                        func.regexp_matches(Paragraph.source_url, reg)[text("1")] + "-01-01"
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

    def get_router(self):

        router = APIRouter(
            prefix="/maintenance",
            tags=["Maintenance"],
            dependencies=[Depends(get_current_admin)],
        )

        @router.put("/{task_name}")
        async def call_task_in_background(
            task_name: str,
            background_tasks: BackgroundTasks,
            params: dict = Body(...)
        ):
            func = {
                'sync-pdate': self.update_pdate_from_url,
                'sync-terms': self.sync_terms,
            }.get(task_name)
            if func:
                background_tasks.add_task(func, **params)
                return {"message": "Maintenance task started in background", "task_name": task_name, "params": params}
            else:
                return {"error": "Maintenance task not found", "task_name": task_name, "params": params}

        return router


maintenance_manager = MaintenanceManager()
