"""Task worker"""

import glob
import os
import tempfile
from typing import Any
import asyncio

from celery import Celery
from sqlalchemy import func, select

from .app import config, storage
from .models import Dataset, Paragraph, TaskDBO, TextEmbeddings, get_db_session
from .task import Task


def make_celery(app_name=__name__) -> Celery:
    """Create and configure Celery instance

    :param app_name: Application name for Celery, defaults to current module
    :type app_name: str, optional
    :return: Configured Celery instance
    :rtype: Celery
    """
    # 初始化Celery
    celery = Celery(
        app_name,
        broker=config.redis + "/1",
        backend=config.redis + "/1",
        # 配置项
        broker_connection_retry_on_startup=True,
        result_expires=86400,  # 结果保留24小时
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        enable_utc=True,
    )
    return celery


celery = make_celery()
task_handlers = {}


@celery.task(name="handle_custom")
async def handle_custom(task_id="", **params):
    """Handle custom task execution

    :param task_id: Task ID from database, defaults to empty string
    :type task_id: str, optional
    :param params: Task parameters if creating new task
    :type params: dict
    :return: Task execution result
    """
    if task_id:
        dbo = await TaskDBO.get(task_id)
    else:
        dbo = TaskDBO(**params)
    task = Task.from_dbo(dbo, log=print)
    return await task.execute_async()


@celery.task(name="handle_ocr")
async def handle_ocr(input, output, lang, monochrome=False):
    """Handle OCR processing of PDF files

    :param input: Input file or directory path
    :type input: str
    :param output: Output file path
    :type output: str
    :param lang: Language code for OCR
    :type lang: str
    :param monochrome: Convert to monochrome before OCR, defaults to False
    :type monochrome: bool, optional
    :return: Output file path
    :rtype: str
    """
    from .pdfutils import convert_pdf_to_tiff_group4, merge_images_from_folder

    temps = []

    input = storage.safe_join(input)

    if os.path.isdir(input):
        fo = tempfile.NamedTemporaryFile("wb", delete=False)
        images = merge_images_from_folder(input, fo)
        fo.close()
        temps.append(fo.name)
        
        if not images:
            for fn in await storage.glob(os.path.join(input, '*.pdf')):
                add_task('ocr', {
                    'input': fn,
                    'output': fn[:-4] + '_ocred.pdf',
                    'lang': lang,
                    'monochrome': monochrome
                })
            return None
        
        input = fo.name
        print("Converted directory to pdf:", input)

    if monochrome:
        fi = open(input, "rb")
        fo = tempfile.NamedTemporaryFile("wb", delete=False)
        convert_pdf_to_tiff_group4(fi, fo)
        fi.close()
        fo.close()
        input = fo.name
        temps.append(fo.name)
        print("Converted pdf to monochromatic:", input)

    output = storage.safe_join(output)
    if output.endswith("/"):
        output += os.path.basename(input).rsplit(".", 1)[0] + "_ocred"
    if not output.endswith(".pdf"):
        output += ".pdf"

    error = None
    try:
        import ocrmypdf

        await asyncio.to_thread(ocrmypdf.ocr,
            input,
            output,
            plugins=["ocrmypdf_paddleocr_remote"],
            language=lang,
            paddle_remote=config.paddle_remote,
            jobs=2,
            force_ocr=True,
        )
    except Exception as e:
        error = e
    finally:
        for f in temps:
            os.unlink(f)

    if error:
        raise error

    return output


@celery.task(name="text_embedding")
async def text_embedding(bulk=None, filters=None) -> None:
    """Generate text embeddings for paragraphs

    :param bulk: Batch of paragraphs to process, defaults to None
    :type bulk: list, optional
    :param filters: Query filters for selecting paragraphs, defaults to None
    :type filters: dict, optional
    """
    if bulk is not None:
        embs = []
        for i in bulk:
            id = i["id"]
            content = i["content"]
            for chunk_id, emb in enumerate(
                await TextEmbeddings.get_embedding_chunks(content, 200, 50), start=1
            ):
                emb = TextEmbeddings(id=id, chunk_id=chunk_id, embedding=emb)
                embs.append(emb)

        async for session in get_db_session():
            session.add_all(embs)
    else:
        print(f"start db query")
        if filters is None:
            filters = {}
        filters.update(embeddings=False)
        cte = (
            (await Paragraph.build_query(filters)).with_only_columns(Paragraph.id).limit(10000)
        ).cte()
        stmt = (
            select(Paragraph)
            .join(cte, Paragraph.id == cte.c.id)
            .filter(func.length(Paragraph.content) > 10)
            .with_only_columns(Paragraph.id, Paragraph.content)
        )
        async for session in get_db_session():
            results = (await session.execute(stmt)).mappings().all()
        bulk = []
        len_results = len(results)
        print(f"start handling embedding for {len_results} records")
        for i, p in enumerate(results, start=1):
            bulk.append({"id": str(p["id"]), "content": p["content"]})
            if i % 100 == 0 or i == len_results:
                add_task("text_embedding", {"bulk": bulk})
                bulk = []
        if len_results == 1000:
            add_task("text_embedding", {})


# Interact with celery
def add_task(task_type, params):
    """Add a task to the Celery queue

    :param task_type: Type of task to add
    :type task_type: str
    :param params: Task parameters
    :type params: dict
    :return: Task ID
    :rtype: str
    :raises ValueError: If task_type is not supported
    """
    task_map = {
        "text_embedding": text_embedding,
        "ocr": handle_ocr,
        "custom": handle_custom,
    }

    if task_type not in task_map:
        raise ValueError("task_type must be one of: " + ", ".join(task_map.keys()))

    task = task_map[task_type].delay(**params)
    return task.id


def get_task_result(task_id) -> dict:
    """获取任务结果和状态"""
    task = celery.AsyncResult(task_id)

    result = {
        "task_id": task_id,
        "status": task.status.lower(),  # PENDING/PROCESSING/SUCCESS/FAILURE
        "result": None,
        "error": None,
    }

    if task.successful():
        result["result"] = task.result
    elif task.failed():
        result["error"] = str(task.result)  # 异常信息

    return result


def get_task_stats() -> dict[str, Any]:
    """统计任务状态（基于Celery Inspect，适合开发/测试环境）"""
    # 生产环境建议用Celery Flower监控，或自定义统计表
    inspect = celery.control.inspect()

    # 获取活跃（处理中）任务
    active = inspect.active() or {}
    processing_count = sum(len(tasks) for tasks in active.values())

    # 获取待处理任务
    reserved = inspect.reserved() or {}
    pending_count = sum(len(tasks) for tasks in reserved.values())

    # 已完成/失败任务需从Redis结果中统计（Celery默认不保存历史统计）
    completed_count = 0
    failed_count = 0
    completed_ids = []

    # 扫描Redis中的任务结果
    for key in celery.backend.client.scan_iter("celery-task-meta-*"):
        task_id = key.decode().split("-", 3)[-1]
        task = celery.AsyncResult(task_id)
        if task.status == "SUCCESS":
            completed_count += 1
            completed_ids.append(task_id)
        elif task.status == "FAILURE":
            failed_count += 1

    return {
        "pending": pending_count,
        "processing": processing_count,
        "completed": completed_count,
        "failed": failed_count,
        "completed_ids": completed_ids,
    }


def revoke_task(task_id, terminate=True) -> dict[str, str]:
    """取消任务"""
    celery.control.revoke(task_id, terminate=terminate)
    # 删除任务结果
    celery.backend.delete(task_id)
    return {"status": "success", "message": f"任务 {task_id} 已取消"}


def clear_tasks(status="") -> dict[str, str | int]:
    """清理指定状态的任务"""
    inspect = celery.control.inspect()
    revoked_count = 0

    if not status:
        celery.control.purge()

    if status in ["pending", ""]:
        # 取消所有待处理任务
        reserved = inspect.reserved() or {}
        for worker, tasks in reserved.items():
            for task in tasks:
                celery.control.revoke(task["id"], terminate=True)
                revoked_count += 1

    if status in ["processing", ""]:
        # 取消所有处理中任务
        active = inspect.active() or {}
        for worker, tasks in active.items():
            for task in tasks:
                celery.control.revoke(task["id"], terminate=True)
                revoked_count += 1

    if status in ["completed", "failed", ""]:
        # 删除已完成/失败任务的结果
        for key in celery.backend.client.scan_iter("celery-task-meta-*"):
            task_id = key.decode().replace("celery-task-meta-", "")
            task = celery.AsyncResult(task_id)
            if (
                status == ""
                or (status == "completed" and task.status == "SUCCESS")
                or (status == "failed" and task.status == "FAILURE")
            ):
                task.forget()
                revoked_count += 1

    return {"status": "success", "deleted_count": revoked_count}


def worker() -> None:
    """Start Celery worker process"""
    celery.worker_main(["worker", "--pool=solo", "--loglevel=INFO"])
