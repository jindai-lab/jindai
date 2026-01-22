"""Task worker"""

import os
import tempfile

from celery import Celery
from sqlalchemy import exists, func, select

from .app import config, storage
from .models import Dataset, Paragraph, TaskDBO, TextEmbeddings, db_session
from .task import Task


def make_celery(app_name=__name__):
    # 初始化Celery
    celery = Celery(
        app_name,
        broker=config.redis.rstrip("/") + "/0",
        backend=config.redis.rstrip("/") + "/1",
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


@celery.task(name="task_worker.handle_custom")
def handle_custom(task_id="", **params):
    if task_id:
        dbo = db_session.query(TaskDBO).get(task_id)
    else:
        dbo = TaskDBO(**params)
    task = Task.from_dbo(dbo)
    return task.execute()


@celery.task(name="task_worker.import_pdf")
def import_pdf(source, lang, dataset):
    from .pdfutils import extract_pdf_texts

    dataset = Dataset.get(dataset).id
    source = storage.safe_join(source)
    if os.path.isdir(source):
        inputs = []
        for pwd, ds, fs in os.walk(source):
            for f in fs:
                if f.endswith(".pdf"):
                    inputs.append(os.path.join(pwd, f))
    else:
        inputs = [source]
    inputs = [f for f in inputs if f.endswith(".pdf")]

    for file in inputs:
        rel_path = '/' + storage.relative_path(file)
        print(rel_path)
        max_page = (
            db_session.query(func.max(Paragraph.source_page))
            .filter(Paragraph.source_url == rel_path)
            .scalar()
        )
        try:
            for page, pagenum, text in extract_pdf_texts(file, since=max_page+1):
                p = Paragraph(
                    content=text,
                    lang=lang,
                    dataset=dataset,
                    source_url=rel_path,
                    source_page=page,
                    pagenum=pagenum,
                )
                db_session.add(p)
            db_session.commit()
        except:
            db_session.rollback()

    return inputs


@celery.task(name="task_worker.handle_ocr")
def handle_ocr(input, output, lang, monochrome=False):
    from .pdfutils import convert_pdf_to_tiff_group4, merge_images_from_folder

    temps = []

    input = storage.safe_join(input)

    if os.path.isdir(input):
        fo = tempfile.NamedTemporaryFile("wb", delete=False)
        merge_images_from_folder(input, fo)
        fo.close()
        input = fo.name
        temps.append(fo.name)
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

        ocrmypdf.ocr(
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


@celery.task(name="task_worker.handle_embedding")
def handle_embedding(id=None, content=None, bulk=None):
    if content is not None:
        bulk = [{"id": id, "content": content}]
    if bulk is not None:
        embs = []
        for i in bulk:
            id = i["id"]
            content = i["content"]
            for chunk_id, emb in enumerate(
                TextEmbeddings.get_embedding_chunks(content, 200, 50), start=1
            ):
                emb = TextEmbeddings(id=id, chunk_id=chunk_id, embedding=emb)
                embs.append(emb)

        try:
            db_session.add_all(embs)
            db_session.commit()
        except:
            db_session.rollback()
    else:
        stmt = select(Paragraph.id, Paragraph.content).where(
            ~exists()
            .where(TextEmbeddings.id == Paragraph.id, TextEmbeddings.chunk_id > 0)
            .correlate(Paragraph),
            func.length(Paragraph.content) > 10,
        )
        results = db_session.execute(stmt.limit(10000)).mappings().all()
        bulk = []
        len_results = len(results)
        for i, p in enumerate(results, start=1):
            bulk.append({"id": str(p["id"]), "content": p["content"]})
            if i % 100 == 0 or i == len_results:
                add_task("text_embedding", {"bulk": bulk})
                bulk = []
        if len_results:
            add_task("text_embedding", {})


# Interact with celery
def add_task(task_type, params):
    task_map = {
        "text_embedding": handle_embedding,
        "ocr": handle_ocr,
        "import": import_pdf,
        "custom": handle_custom,
    }

    if task_type not in task_map:
        raise ValueError("task_type must be one of: " + ", ".join(task_map.keys()))

    task = task_map[task_type].delay(**params)
    return task.id


def get_task_result(task_id):
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


def get_task_stats():
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


def revoke_task(task_id, terminate=True):
    """取消任务"""
    celery.control.revoke(task_id, terminate=terminate)
    # 删除任务结果
    celery.backend.delete(task_id)
    return {"status": "success", "message": f"任务 {task_id} 已取消"}


def clear_tasks(status=""):
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


def worker():
    celery.worker_main(["worker", "--concurrency=2"])
