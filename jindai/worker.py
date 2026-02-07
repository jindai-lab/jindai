import os
import tempfile
import asyncio
from typing import Any, Dict, List, Optional
from celery import Celery
from fastapi import APIRouter, Body
from sqlalchemy import func, select

from .app import config, storage
from .models import Dataset, Paragraph, TaskDBO, TextEmbeddings, get_db_session
from .task import Task as CustomTaskLogic


class WorkerManager:
    def __init__(self, app_name: str = "worker_manager"):
        self.celery = self._init_celery(app_name)
        # 将任务绑定到当前实例
        self._register_tasks()

    def _init_celery(self, app_name: str) -> Celery:
        """配置并初始化 Celery 实例"""
        return Celery(
            app_name,
            broker=f"{config.redis}/1",
            backend=f"{config.redis}/1",
            broker_connection_retry_on_startup=True,
            result_expires=86400,
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            enable_utc=True,
        )

    def _register_tasks(self):
        """
        在这里利用 Celery 的装饰器将方法注册为任务。
        注意：在类中使用 celery.task 需要特殊处理，或者直接定义为普通函数再包装。
        """
        self.handle_custom = self.celery.task(name="handle_custom")(
            self._handle_custom_logic
        )
        self.handle_ocr = self.celery.task(name="handle_ocr")(self._handle_ocr_logic)
        self.text_embedding = self.celery.task(name="text_embedding")(
            self._text_embedding_logic
        )

    # --- 任务逻辑实现 (Internal Logic) ---

    async def _handle_custom_logic(self, task_id: str = "", **params):
        if task_id:
            dbo = await TaskDBO.get(task_id)
        else:
            dbo = TaskDBO(**params)
        task = CustomTaskLogic.from_dbo(dbo, log=print)
        return await task.execute_async()

    async def _handle_ocr_logic(
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
                for fn in await storage.glob(os.path.join(input_path, "*.pdf")):
                    self.add_task(
                        "ocr",
                        {
                            "input_path": fn,
                            "output_path": fn[:-4] + "_ocred.pdf",
                            "lang": lang,
                            "monochrome": monochrome,
                        },
                    )
                return None
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

    async def _text_embedding_logic(self, bulk: List = None, filters: Dict = None):
        if bulk is not None:
            embs = []
            for i in bulk:
                for chunk_id, emb in enumerate(
                    await TextEmbeddings.get_embedding_chunks(i["content"], 200, 50),
                    start=1,
                ):
                    embs.append(
                        TextEmbeddings(id=i["id"], dataset=i["dataset"], chunk_id=chunk_id, embedding=emb)
                    )
            async for session in get_db_session():
                session.add_all(embs)
        else:
            if filters is None:
                filters = {}
            filters.update(embeddings=False)
            cte = (
                (await Paragraph.build_query(filters))
                .with_only_columns(Paragraph.id)
                .limit(10000)
            ).cte()
            stmt = (
                select(Paragraph)
                .join(cte, Paragraph.id == cte.c.id)
                .filter(func.length(Paragraph.content) > 10)
                .with_only_columns(Paragraph.id, Paragraph.Dataset, Paragraph.content)
            )

            async for session in get_db_session():
                results = (await session.execute(stmt)).mappings().all()

            new_bulk = []
            for i, p in enumerate(results, start=1):
                new_bulk.append({"id": str(p["id"]), "content": p["content"]})
                if i % 100 == 0 or i == len(results):
                    self.add_task("text_embedding", {"bulk": new_bulk})
                    new_bulk = []

    # --- 对外暴露的管理接口 (Public API) ---

    def add_task(self, task_type: str, params: Dict) -> str:
        """派发任务"""
        task_map = {
            "text_embedding": self.text_embedding,
            "ocr": self.handle_ocr,
            "custom": self.handle_custom,
        }
        if task_type not in task_map:
            raise ValueError(
                f"Unsupported task type. Choice from: {list(task_map.keys())}"
            )

        # 使用 .delay() 异步调用
        job = task_map[task_type].delay(**params)
        return job.id

    def task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务详情数据"""
        res = self.celery.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": res.status.lower(),
            "result": res.result if res.successful() else None,
            "error": str(res.result) if res.failed() else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """全局状态统计"""
        inspect = self.celery.control.inspect()

        # 统计运行中和队列中
        active = sum(len(ts) for ts in (inspect.active() or {}).values())
        pending = sum(len(ts) for ts in (inspect.reserved() or {}).values())

        # 扫描 Redis 统计已完成/失败
        success_count = 0
        failed_count = 0
        for key in self.celery.backend.client.scan_iter("celery-task-meta-*"):
            task = self.celery.AsyncResult(key.decode().split("-")[-1])
            if task.status == "SUCCESS":
                success_count += 1
            elif task.status == "FAILURE":
                failed_count += 1

        return {
            "processing": active,
            "pending": pending,
            "completed": success_count,
            "failed": failed_count,
        }

    def revoke(self, task_id: str):
        """取消任务"""
        self.celery.control.revoke(task_id, terminate=True)
        self.celery.backend.delete(task_id)
        return True

    def clear_tasks(self, status="") -> dict[str, str | int]:
        """清理指定状态的任务"""
        inspect = self.celery.control.inspect()
        revoked_count = 0
        if not status:
            self.celery.control.purge()

        if status in ["pending", ""]:
            # 取消所有待处理任务
            reserved = inspect.reserved() or {}
            for worker, tasks in reserved.items():
                for task in tasks:
                    self.celery.control.revoke(task["id"], terminate=True)
                    revoked_count += 1

        if status in ["processing", ""]:
            # 取消所有处理中任务
            active = inspect.active() or {}
            for worker, tasks in active.items():
                for task in tasks:
                    self.celery.control.revoke(task["id"], terminate=True)
                    revoked_count += 1

        if status in ["completed", "failed", ""]:
            # 删除已完成/失败任务的结果
            for key in self.celery.backend.client.scan_iter("celery-task-meta-*"):
                task_id = key.decode().replace("celery-task-meta-", "")
                task = self.celery.AsyncResult(task_id)
                if (
                    status == ""
                    or (status == "completed" and task.status == "SUCCESS")
                    or (status == "failed" and task.status == "FAILURE")
                ):
                    task.forget()
                    revoked_count += 1

        return {"status": "success", "deleted_count": revoked_count}

    def run_worker(self):
        """启动 Worker 进程"""
        self.celery.worker_main(["worker", "--pool=solo", "--loglevel=INFO"])

    def get_router(self):
        router = APIRouter(prefix="/worker", tags=["Worker"])
        
        @router.get("/{task_id}")
        def task_status(task_id: str = ""):
            return self.task_status(task_id)

        @router.post("/")
        def start_task(payload: dict = Body(...)):
            return self.add_task(**payload)

        @router.delete("/")
        def clear_task(status: str = ""):
            return self.clear_tasks(status)

        @router.delete("/{task_id}")
        def task_status(task_id: str = ""):
            return self.revoke(task_id)

        @router.get("/")
        def task_stats():
            return self.get_stats()
        
        return router


# 实例化
worker_manager = WorkerManager()
