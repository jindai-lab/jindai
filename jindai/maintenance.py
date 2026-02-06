from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import TIMESTAMP, cast, func, select, text, update
from sqlalchemy.orm import Session

from .app import get_current_admin
from .models import Dataset, Paragraph, get_db


class MaintenanceManager:

    def update_pdate_from_url(self, dataset, session):
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

        session.execute(stmt)

    def get_router(self):

        router = APIRouter(
            prefix="/maintenance",
            tags=["Maintenance"],
            dependencies=[Depends(get_current_admin)],
        )

        @router.post("/sync-pdate")
        async def sync_paragraph_date(
            dataset: str,
            background_tasks: BackgroundTasks,
            db: Session = Depends(get_db),
        ):
            """
            根据 source_url 中的年份异步更新 Paragraph 表的 pdate 字段
            """
            # 1. 立即返回 202 Accepted，避免前端超时
            background_tasks.add_task(self.update_pdate_from_url, dataset, db)

            return {"message": "Task started in background", "task_name": "sync-pdate"}

        return router


maintenance_manager = MaintenanceManager()
