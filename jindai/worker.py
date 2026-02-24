import asyncio
import inspect
import json
from datetime import datetime
import logging
from typing import Any, Callable

import palitra
from celery import Celery, Task
from celery.exceptions import Ignore, OperationalError
from celery.result import AsyncResult
from fastapi import APIRouter, Body, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from redis import Redis as syncredis
from redis import asyncio as aioredis

from .config import config
from .config import oidc_validator
from .helpers import inspect_function_signature


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

        # 初始化统计用 Redis 客户端
        self._started = False

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

        # 存储任务元信息（保留原需求：task_name + 参数）
        await self.redis_storage_client.set_task_data(
            job_id,
            {
                "task_name": task_name,
                "args": args,
                "kwargs": kwargs,
                "enqueue_time": datetime.now().isoformat(),
            },
        )

        return job_id

    async def get_stats(self) -> dict[str, Any]:
        """获取任务统计（完全兼容原接口）"""
        await self.startup()

        stats = {"queued": 0, "running": 0, "completed": 0, "failed": 0}

        # 匹配任务参数的 key，过滤出有效 job_id
        pattern = f"{self.redis_storage_client.prefix}*"
        keys = await self.redis_storage_client.redis.keys(pattern)
        job_ids = [k.replace(self.redis_storage_client.prefix, "") for k in keys]

        stats["results"] = [await self.get_result(jid) for jid in job_ids]
        for result in stats["results"]:
            if result["status"] == "PENDING":
                stats["queued"] += 1
            elif result["status"] == "SUCCESS":
                stats["completed"] += 1
            elif result["status"] == "STARTED":
                stats["running"] += 1
            else:
                stats["failed"] += 1

        return stats

    async def clear(self) -> int:
        """清空所有统计和任务数据"""
        await self.redis_storage_client.redis.delete("celery")

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
        }

        # 合并任务元信息（task_name + 参数）
        task_params = await self.redis_storage_client.get_task_data(job_id)
        result_dict.update(task_params)

        return result_dict

    async def remove_result(self, job_id: str):
        """删除任务结果和参数（兼容原接口）"""
        # 删除任务参数
        await self.redis_storage_client.remove_task_data(job_id)

    async def abort(self, job_id: str):
        """实现任务终止功能（核心需求）"""
        await self.startup()

        # 尝试直接终止正在执行的任务
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            result.revoke(terminate=True, signal="SIGTERM")
        except OperationalError as e:
            raise RuntimeError(f"Failed to abort task {job_id}: {str(e)}")

    async def shutdown(self):
        """优雅关闭连接（兼容原接口）"""
        await self.startup()
        # 关闭 Redis 连接
        await self.redis_storage_client.close()
        # 关闭 Celery 连接
        self.celery_app.close()

    def register_task(self, async_func: Callable, task_name: str, ignore_result=False):
        """注册异步任务（兼容原接口）"""

        @self.celery_app.task(
            name=task_name, bind=True, ignore_result=ignore_result, track_started=True
        )
        def palitra_wrapped(task, *a, **k):
            return palitra.run(async_func(*a, **k))

        palitra_wrapped.__orig__ = async_func

        return palitra_wrapped

    def get_tasks(self):
        ret = {}
        for name, task in self.celery_app.tasks.items():
            if "." in name:
                continue
            func = getattr(task, "__orig__", lambda: 0)
            ret[name] = inspect_function_signature(func)
        return ret

    def get_wsrouter(self) -> APIRouter:
        wsrouter = APIRouter(prefix="/jobs")

        @wsrouter.websocket("/stats")
        async def jobs_websocket(websocket: WebSocket, token: str = Query(None)):
            try:
                await oidc_validator.validate_token(token)
            except:
                await websocket.close()
                return
            
            await websocket.accept()
            try:
                while True:                    
                    stats = await self.get_stats()
                    await websocket.send_json(stats)
                    await asyncio.sleep(5)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logging.error(f"Websocket error: {e}")

        return wsrouter

    def get_router(self) -> APIRouter:
        """生成 FastAPI 路由（完全兼容原接口）"""
        router = APIRouter(prefix="/worker", tags=["worker"])

        @router.get("/")
        def list_tasks():
            return self.get_tasks()

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


worker_manager = AsyncWorkerManager(config.redis + "/0")
celery = worker_manager.celery_app
