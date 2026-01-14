"""Task worker"""

import json
import redis
import uuid

from sqlalchemy import exists, func, select

from .models import TaskDBO, Paragraph, TextEmbeddings, db_session
from .task import Task
from .app import config


r = redis.Redis(**config.redis)


def add_task(task_type, params):
    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "type": task_type,
        "params": params,
        "status": "pending",
    }
    # 1. 存入结果表，初始化状态
    r.hset(f"task:results:{task_id}", mapping={"status": "pending", "data": ""})
    # 2. 推入队列
    r.lpush("tasks:pending", json.dumps(task_data))
    return task_id


def clear_tasks(status=""):
    """
    清理/取消特定状态下的任务
    :param status: 'pending', 'processing', 'completed', 'failed' 或 '' (代表全部)
    """

    # 1. 如果清理 'pending' 或全部，需要清空待处理 List
    if status == "pending" or status == "":
        r.delete("tasks:pending")
        print("Pending list cleared")

    # 2. 清理 Hash 存储中的状态记录
    # 使用 scan_iter 避免在大数据量下阻塞 Redis
    cursor = 0
    pattern = "task:results:*"

    if not status:
        return r.flushall()
    else:
        deleted_count = 0
        for key in r.scan_iter(pattern):
            # 如果指定了状态，需要先检查状态
            if status != "":
                current_status = r.hget(key, "status")
                if current_status == status:
                    r.delete(key)
                    deleted_count += 1
            else:
                # 如果 status 为空字符串，删除所有任务记录
                r.delete(key)
                deleted_count += 1
        return deleted_count


def get_task_stats():
    # 1. 获取待处理数量（极其高效）
    pending_count = r.llen("tasks:pending")

    # 2. 统计处理中和已完成（需要遍历结果 Key）
    # 注意：如果任务量极大（10万+），SCAN 会比 KEYS 更安全，避免阻塞 Redis
    processing_count = 0
    completed_count = 0
    failed_count = 0

    # 扫描所有结果 Key
    for key in r.scan_iter("task:results:*"):
        status = r.hget(key, "status")
        if status == b"processing":
            processing_count += 1
        elif status == b"completed":
            completed_count += 1
        elif status == b"failed":
            failed_count += 1

    stats = {
        "pending": pending_count,
        "processing": processing_count,
        "completed": completed_count,
        "failed": failed_count,
    }

    for key in r.scan_iter("task:results:*"):
        status = r.hget(key, "status")

        if status in stats:
            stats[status] += 1

        # 如果状态是 completed，记录其 ID
        if status == "completed":
            task_id = key.split(":")[-1]
            stats["completed_ids"].append(task_id)

    return stats


def worker():
    while True:
        # 阻塞式读取，超时时间 10 秒
        raw_task = r.brpop("tasks:pending", timeout=10)
        if not raw_task:
            continue
        task = json.loads(raw_task[1])
        task_id = task["id"]
        res_key = f"task:results:{task_id}"
        print("Preparing task:", task_id)

        # 更新状态为处理中
        r.hset(res_key, "status", "processing")

        try:
            # 策略模式：根据类型处理
            if task["type"] == "text_embedding":
                result = handle_embedding(**task["params"])
            else:
                result = handle_custom(**task["params"])

            # 任务成功，保存结果
            if result is None:
                print(f"Result for {task_id} is empty, therefore removed")
                r.delete(res_key)
            else:
                r.hset(
                    res_key,
                    mapping={"status": "completed", "result": json.dumps(result)},
                )
        # except Exception as e:
        except ValueError as e:
            print(task_id, "failed with exception", e)
            r.hset(res_key, "status", "failed")


def get_task_result(task_id):
    """获取特定任务的结果和状态"""
    key = f"task:results:{task_id}"
    # HGETALL 返回一个字典
    data = r.hgetall(key)

    if not data:
        return {"error": "Task not found", "task_id": task_id}

    # 尝试解析 JSON 格式的结果字段
    if "result" in data:
        try:
            data["result"] = json.loads(data["result"])
        except (json.JSONDecodeError, TypeError):
            pass

    return data


def delete_task(task_id):
    """从 Redis 中彻底删除任务记录"""
    key = f"task:results:{task_id}"

    # 注意：这里只能删除结果记录。
    # 如果任务还在 pending 队列中，List 不支持根据内容高效删除特定 ID。
    # 这是一个“逻辑删除”：Worker 执行完发现结果 Key 不在了，可以自行终止。
    result = r.delete(key)
    return result > 0  # 返回布尔值：是否成功删除


def handle_custom(task_id="", **params):
    if task_id:
        dbo = db_session.query(TaskDBO).get(task_id)
    else:
        dbo = TaskDBO(**params)
    task = Task.from_dbo(dbo)
    return task.execute()


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

        db_session.add_all(embs)
        db_session.commit()
    else:
        stmt = select(Paragraph.id, Paragraph.content).where(
            ~exists().where(
                TextEmbeddings.id == Paragraph.id, TextEmbeddings.chunk_id > 0
            ).correlate(Paragraph),
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
