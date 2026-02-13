import asyncio
import json
import re
import threading
from datetime import datetime
import time
from typing import Any, Callable, Dict, Optional

import palitra
from celery import Celery, Task
from celery.events import EventReceiver
from celery.exceptions import Ignore, OperationalError
from celery.result import AsyncResult
from fastapi import APIRouter, Body, FastAPI
from redis import Redis as syncredis
from redis import asyncio as aioredis
from redis.asyncio import Redis

from .config import instance as config


class RedisStorageClient:
    def __init__(self, redis_dsn: str):
        self.redis = aioredis.from_url(redis_dsn, decode_responses=True)
        self.prefix = "task_params:"

    async def get_task_data(self, job_id: str) -> dict:
        """获取任务参数"""
        data = await self.redis.get(f"{self.prefix}{job_id}")
        return json.loads(data) if data else {}

    async def set_task_data(self, job_id: str, data: dict):
        """存储任务参数"""
        await self.redis.set(f"{self.prefix}{job_id}", json.dumps(data))

    async def remove_task_data(self, job_id: str):
        """删除任务参数"""
        await self.redis.delete(f"{self.prefix}{job_id}")

    async def close(self):
        await self.redis.close()


class AsyncTrackedTask(Task):
    """自定义任务类，仅保留终止检查，移除统计计数"""
    abstract = True

    def __call__(self, *args, **kwargs):
        """任务执行入口，检查终止标记"""
        # 检查是否被终止
        if AsyncWorkerManager.instance.is_task_aborted_sync(self.request.id):
            raise Ignore("Task aborted by user")
        return super().__call__(*args, **kwargs)


class AsyncWorkerManager:
    
    instance = None
    
    def __init__(self, redis_dsn: str) -> None:
        AsyncWorkerManager.instance = self
        # 初始化 Redis 存储客户端（存储任务参数）
        self.redis_storage_client = RedisStorageClient(redis_dsn)
        
        # 初始化 Celery
        self.celery_app = Celery(
            "async_worker_manager",
            broker=redis_dsn,
            backend=redis_dsn,
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            result_expires=3600,  # 结果过期时间 1 小时
        )
        
        # 配置自定义任务类
        self.celery_app.Task = AsyncTrackedTask
        
        # 初始化统计用 Redis 客户端
        self.redis_stats = aioredis.from_url(redis_dsn, decode_responses=True)
        self.sync_redis_stats = syncredis.from_url(redis_dsn, decode_responses=True)
        self.stats_prefix = "taskiq:stats"
        self._started = False
        
        # 终止任务的 Redis 键前缀
        self.abort_prefix = "task_abort:"

    async def startup(self):
        """启动 Celery 连接（模拟原 broker.startup）"""
        if not self._started:
            self.celery_app.connection().ensure_connection()
        self._started = True
        return True

    async def enqueue(self, task_name: str, *args, **kwargs) -> str:
        """入队任务，仅手动维护 queued 计数（核心兼容逻辑）"""
        await self.startup()
        
        # 查找 Celery 任务
        try:
            task = self.celery_app.tasks[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found")
        
        # 提交异步任务
        result = task.apply_async(args=args, kwargs=kwargs)
        job_id = result.id
        
        # 仅手动维护 queued 计数（兼容原有逻辑，避免重复）
        await self.redis_stats.incr(f"{self.stats_prefix}:queued")
        
        # 存储任务元信息（保留原需求：task_name + 参数）
        await self.redis_storage_client.set_task_data(
            job_id,
            {
                "task_name": task_name,
                "args": args,
                "kwargs": kwargs,
                "enqueue_time": datetime.now().isoformat()
            }
        )
        
        return job_id

    async def get_stats(self, detailed: bool = False) -> dict[str, Any]:
        """获取任务统计（完全兼容原接口）"""
        await self.startup()
        
        stats = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        }
        
        # 匹配任务参数的 key，过滤出有效 job_id
        pattern = f"{self.redis_storage_client.prefix}*"
        keys = await self.redis_stats.keys(pattern)
        job_ids = [k.replace(self.redis_storage_client.prefix, "") for k in keys]
        
        stats['results'] = [
            await self.get_result(jid)
            for jid in job_ids
        ]
        for result in stats['results']:
            if result['status'] == 'PENDING':
                stats['queued'] += 1
            elif result['status'] == 'SUCCESS':
                stats['completed'] += 1
            elif result['status'] == 'PROCESSING':
                stats['running'] += 1
            else:
                stats['failed'] += 1
        
        return stats

    async def clear(self) -> int:
        """清空所有统计和任务数据（兼容原接口）"""
        patterns = [
            f"{self.stats_prefix}:*",
            f"{self.redis_storage_client.prefix}*",
            f"{self.abort_prefix}*"
        ]
        
        total_deleted = 0
        async with self.redis_stats.pipeline() as pipe:
            for pattern in patterns:
                keys = await self.redis_stats.keys(pattern)
                if keys:
                    await pipe.delete(*keys)
                    total_deleted += len(keys)
            await pipe.execute()
        
        # 清空 Celery 结果（可选）
        self.celery_app.backend.cleanup()
        
        print(f"Purged {total_deleted} Redis keys.")
        return total_deleted

    async def get_result(self, job_id: str) -> dict:
        """获取任务结果（包含参数，兼容原接口）"""
        # 获取 Celery 任务结果
        result = AsyncResult(job_id, app=self.celery_app)
        
        # 构造结果字典
        result_dict = {
            "job_id": job_id,
            "status": result.status,
            "result": result.result if result.successful() else None,
            "error": str(result.result) if result.failed() else None,
            "traceback": result.traceback,
            "date_done": result.date_done.isoformat() if result.date_done else None,
            "aborted": await self.is_task_aborted(job_id)
        }
        
        # 合并任务元信息（task_name + 参数）
        task_params = await self.redis_storage_client.get_task_data(job_id)
        result_dict.update(task_params)
        
        return result_dict

    async def remove_result(self, job_id: str):
        """删除任务结果和参数（兼容原接口）"""
        # 删除任务参数
        await self.redis_storage_client.remove_task_data(job_id)
        # 删除终止标记
        await self.redis_stats.delete(f"{self.abort_prefix}{job_id}")
        
    async def abort(self, job_id: str):
        """实现任务终止功能（核心需求）"""
        await self.startup()
        
        # 设置终止标记（供任务执行时检查）
        await self.redis_stats.set(f"{self.abort_prefix}{job_id}", "1")
        
        # 尝试直接终止正在执行的任务
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            
            # 根据任务状态处理
            if result.status == "STARTED":
                # 运行中：强制终止
                self.celery_app.control.revoke(
                    job_id, terminate=True, signal='SIGTERM'
                )
            elif result.status == "PENDING":
                # 待执行：从队列移除
                self.celery_app.control.revoke(job_id, terminate=False)
            
        except OperationalError as e:
            raise RuntimeError(f"Failed to abort task {job_id}: {str(e)}")

    async def is_task_aborted(self, job_id: str) -> bool:
        """检查任务是否被终止"""
        abort_flag = await self.redis_stats.get(f"{self.abort_prefix}{job_id}")
        return abort_flag == "1"
    
    def is_task_aborted_sync(self, job_id: str) -> bool:
        abort_flag = self.sync_redis_stats.get(f"{self.abort_prefix}{job_id}")
        return abort_flag == "1"

    async def shutdown(self):
        """优雅关闭连接（兼容原接口）"""
        await self.startup()
        # 关闭 Redis 连接
        self.sync_redis_stats.close()
        await self.redis_stats.close()
        await self.redis_storage_client.close()
        # 关闭 Celery 连接
        self.celery_app.close()

    def register_task(self, async_func: Callable, task_name: str):
        """注册异步任务（兼容原接口）"""
        @self.celery_app.task(name=task_name, bind=True)
        def palitra_wrapped(self, *a, **k):
            async def wrapped_task(self, *args, **kwargs):
                # 检查终止标记
                if await AsyncWorkerManager.instance.is_task_aborted(self.request.id):
                    raise Ignore("Task aborted before execution")
                return await async_func(*args, **kwargs)
            palitra.run(wrapped_task(self, *a, **k))    
        
        return palitra_wrapped

    def get_router(self) -> APIRouter:
        """生成 FastAPI 路由（完全兼容原接口）"""
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
        async def get_result(job_id: str):
            return await self.get_result(job_id)

        @router.delete("/")
        async def clear_all():
            return await self.clear()

        @router.delete("/{job_id}")
        async def remove_result(job_id: str):
            return await self.remove_result(job_id)
        
        @router.post("/{job_id}/abort")
        async def abort_job(job_id: str):
            await self.abort(job_id)
            return {"status": "success", "message": f"Task {job_id} aborted"}

        return router


worker_manager = AsyncWorkerManager(config.redis)
celery = worker_manager.celery_app
