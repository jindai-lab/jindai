import asyncio
from datetime import datetime
import json
import re
from typing import Any, Dict
from fastapi import APIRouter, Body
import redis.asyncio as redis
from taskiq import TaskiqMessage, TaskiqMiddleware, TaskiqResult
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend

from .config import instance as config
import logging


logger = logging.getLogger("taskiq_redis_stream_storage")


class RedisStorageClient:
    
    def __init__(self, redis_dsn: str):
        """
        初始化Redis存储客户端（复用Taskiq的Redis Broker连接）
        :param redis_broker: Taskiq的RedisBroker实例
        """
        self.redis = redis.from_url(redis_dsn, decode_responses=True)
        # Redis key 前缀（便于区分任务参数和队列数据）
        self.key_prefix = "taskiq:task_params:"

    async def store_task_data(self, task_data: dict[str, Any]) -> None:
        """
        将任务数据存入Redis
        - 主key：{prefix}{task_id}（哈希结构，存储任务详情）
        - 辅助key：{prefix}task_names（集合，存储所有任务名）
        - 辅助key：{prefix}enqueue_times（有序集合，存储任务入队时间）
        """
        try:
            job_id = task_data["job_id"]
            main_key = f"{self.key_prefix}{job_id}"
            
            # 1. 存储任务核心数据（哈希结构，便于单独查询字段）
            # 将非字符串类型数据序列化
            serialized_data = {
                "task_name": task_data["task_name"],
                "args": json.dumps(task_data["args"]),
                "kwargs": json.dumps(task_data["kwargs"]),
                "enqueue_time": task_data["enqueue_time"],
                "created_at": datetime.now().isoformat()
            }
            
            # 异步写入Redis哈希
            await self.redis.hset(main_key, mapping=serialized_data)
            # 设置过期时间（可选，避免Redis数据膨胀，比如7天过期）
            await self.redis.expire(main_key, 60 * 60 * 24 * 7)  # 7天

            # 2. 辅助存储：记录所有任务名（便于统计）
            await self.redis.sadd(f"{self.key_prefix}task_names", task_data["task_name"])

            # 3. 辅助存储：记录任务入队时间（有序集合，便于按时间筛选）
            enqueue_ts = datetime.fromisoformat(task_data["enqueue_time"]).timestamp()
            await self.redis.zadd(
                f"{self.key_prefix}enqueue_times",
                {job_id: enqueue_ts}
            )

        except Exception as e:
            logging.error(f"Failed to store task {job_id} params to Redis: {str(e)}", exc_info=e)

    async def get_task_data(self, job_id: str) -> dict[str, Any]:
        """
        从Redis查询单个任务的参数（方便后续查询使用）
        :return: 解析后的任务数据，None表示不存在
        """
        try:
            main_key = f"{self.key_prefix}{job_id}"
            raw_data = await self.redis.hgetall(main_key)
            
            if not raw_data:
                return {}

            # 反序列化数据
            return {
                "job_id": job_id,
                "task_name": raw_data.get("task_name", ""),
                "args": json.loads(raw_data.get("args", "[]")),
                "kwargs": json.loads(raw_data.get("kwargs", "{}")),
                "enqueue_time": raw_data.get("enqueue_time", ""),
                "queue_name": raw_data.get("queue_name", ""),
                "created_at": raw_data.get("created_at", ""),
            }
        except Exception as e:
            logging.error(f"Failed to get task {job_id} params from Redis: {str(e)}", exc_info=e)
            return {}
        
    async def remove_task_data(self, job_id: str):
        
        try:
            main_key = f"{self.key_prefix}{job_id}"
            await self.redis.hdel(main_key)
        except Exception as e:
            logging.error(f"Failed to remove task {job_id} params from Redis: {str(e)}", exc_info=e)
            return {}


class TaskParamsRedisStorageMiddleware(TaskiqMiddleware):
    """
    Taskiq中间件：任务入队时将参数和名称存储到Redis
    """
    def __init__(self, redis_client: RedisStorageClient):
        self.redis_client = redis_client

    async def pre_send(self, message: TaskiqMessage) -> TaskiqMessage:
        """
        任务发送（入队）前触发：解析数据并存储到Redis
        """
        # 1. 解析任务元数据
        task_data = self._parse_task_data(message)
        
        # 2. 异步存储到Redis（不阻塞任务发送）
        # 使用create_task避免阻塞主线程
        asyncio.create_task(self.redis_client.store_task_data(task_data))
        
        # 3. 返回原始消息，不影响任务正常入队
        return message

    def _parse_task_data(self, message: TaskiqMessage) -> Dict[str, Any]:
        """解析TaskiqMessage，提取需要存储的字段"""
        return {
            "job_id": message.task_id or "unknown_id",
            "task_name": message.task_name or "unknown_task",
            "args": message.args or [],
            "kwargs": message.kwargs or {},
            "enqueue_time": datetime.now().isoformat(),
        }


class StatsMiddleware(TaskiqMiddleware):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.prefix = "taskiq:stats"

    async def pre_enqueue(self, message: TaskiqMessage) -> TaskiqMessage:
        # Task added to the queue
        await self.redis.incr(f"{self.prefix}:queued")
        return message

    async def pre_execute(self, message: TaskiqMessage) -> TaskiqMessage:
        # Worker picked up the task
        await self.redis.decr(f"{self.prefix}:queued")
        await self.redis.incr(f"{self.prefix}:running")
        return message

    async def post_execute(self, message: TaskiqMessage, result: TaskiqResult) -> None:
        # Task finished (Success or Error)
        await self.redis.decr(f"{self.prefix}:running")
        if result.is_err:
            await self.redis.incr(f"{self.prefix}:failed")
        else:
            await self.redis.incr(f"{self.prefix}:completed")

    async def post_save(self, message: TaskiqMessage, result: TaskiqResult) -> None:
        pass


class AsyncWorkerManager:

    def __init__(self, redis_dsn) -> None:
        self.redis_storage_client = RedisStorageClient(redis_dsn)

        result_backend = RedisAsyncResultBackend(
            redis_url=redis_dsn,
        )
        broker = RedisStreamBroker(
            url=redis_dsn,
        ).with_result_backend(result_backend).with_middlewares(
            StatsMiddleware(redis_dsn),
            TaskParamsRedisStorageMiddleware(self.redis_storage_client)
        )
        
        self.broker = broker
        self.redis_stats = redis.from_url(redis_dsn, decode_responses=True)
        self.stats_prefix = "taskiq:stats"
        self._started = False

    async def startup(self):
        if not self._started:
            await self.broker.startup()
        self._started = True
        return True

    async def enqueue(self, task_name: str, *args, **kwargs) -> str:
        """Enqueue a task and increment 'queued' counter"""
        # We use the broker's formatted task name
        await self.startup()
        task = self.broker.find_task(task_name)
        if task:
            kiq = await task.kiq(*args, **kwargs)

            # Atomic increment for stats
            await self.redis_stats.incr(f"{self.stats_prefix}:queued")
        return kiq.task_id

    async def get_stats(self, detailed: bool = False) -> dict[str, int]:
        """Fetch all counters from Redis"""
        await self.startup()
        keys = ["queued", "running", "completed", "failed"]
    
        # Use a pipeline for a single round-trip to Redis
        async with self.redis_stats.pipeline() as pipe:
            for key in keys:
                pipe.get(f"taskiq:stats:{key}")
            values = await pipe.execute()

        stats = {k: int(v) if v else 0 for k, v in zip(keys, values)}
        if detailed:
            ids = await self.redis_stats.keys('*')
            ids = [_ for _ in ids if re.match(r'^[0-9a-f]+$', _)]
            stats['results'] = [
                await self.get_result(jid)
                for jid in ids
            ]
        return stats

    async def clear(self):
        patterns = [
            "*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            keys = await self.redis_stats.keys(pattern)
            if keys:
                await self.redis_stats.delete(*keys)
                total_deleted += len(keys)
        
        print(f"Purged {total_deleted} Redis keys.")
        return total_deleted
    
    async def get_result(self, job_id):
        result = await self.broker.result_backend.get_result(job_id, True)
        result = result.__dict__
        result.update(await self.redis_storage_client.get_task_data(job_id))
        return result
    
    async def remove_result(self, job_id):
        await self.redis_storage_client.remove_task_data(job_id)
        await self.redis_stats.delete(job_id)
    
    async def abort(self, job_id):
        raise NotImplementedError()

    async def shutdown(self):
        """Cleanly close connections"""
        await self.startup()
        await self.broker.shutdown()
        await self.redis_stats.close()

    def get_router(self):
        router = APIRouter(prefix="/worker", tags=["worker"])

        @router.get("/")
        async def get_stats():
            return await self.get_stats()
        
        @router.get("/jobs")
        async def list_jobs():
            return await self.get_stats(True)
        
        @router.post("/{task_name}")
        async def add_task(task_name: str, payload: dict = Body(...)):
            return await self.enqueue(task_name, **payload)

        @router.get("/{job_id}")
        async def get_result(job_id):
            return await self.get_result(job_id)

        @router.delete("/")
        async def clear_all():
            return await self.clear()

        @router.delete("/{job_id}")
        async def remove_result(job_id):
            return await self.remove_result(job_id)
        
        return router

    def register_task(self, async_func, task_name):
        async def wrapped(**kwargs):
            return await async_func(**kwargs)

        self.broker.register_task(wrapped, task_name)


worker_manager = AsyncWorkerManager(config.redis)
broker = worker_manager.broker
