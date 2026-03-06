"""Async worker manager for Jindai application.

This module provides:
- AsyncWorkerManager: Manages Celery-based async task execution
- Redis storage for task parameters
- WebSocket support for job status updates
- FastAPI routes for task management
"""

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
    """Redis client for storing task parameters."""

    def __init__(self, redis_dsn: str):
        """Initialize Redis storage client.

        Args:
            redis_dsn: Redis connection string.
        """
        self.redis = aioredis.from_url(redis_dsn, decode_responses=True)
        self.prefix = "task_params:"

    async def get_task_data(self, job_id: str) -> dict:
        """Get task parameters from Redis.

        Args:
            job_id: Task job ID.

        Returns:
            Task parameters dictionary.
        """
        data = await self.redis.get(f"{self.prefix}{job_id}")
        return json.loads(data) if data else {}

    async def set_task_data(self, job_id: str, data: dict):
        """Store task parameters in Redis.

        Args:
            job_id: Task job ID.
            data: Task parameters dictionary.
        """
        await self.redis.set(f"{self.prefix}{job_id}", json.dumps(data))

    async def remove_task_data(self, job_id: str):
        """Remove task parameters from Redis.

        Args:
            job_id: Task job ID.
        """
        await self.redis.delete(f"{self.prefix}{job_id}")

    async def close(self):
        """Close Redis connection."""
        await self.redis.close()


class AsyncWorkerManager:
    """Manager for async Celery workers.

    Handles task queuing, result retrieval, and job status monitoring.
    """

    instance = None

    def __init__(self, redis_dsn: str) -> None:
        """Initialize async worker manager.

        Args:
            redis_dsn: Redis connection string for broker and backend.
        """
        AsyncWorkerManager.instance = self
        # Initialize Redis storage client (stores task parameters)
        self.redis_storage_client = RedisStorageClient(redis_dsn)

        # Initialize Celery
        self.celery_app = Celery(
            "async_worker_manager",
            broker=redis_dsn,
            backend=redis_dsn,
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            result_expires=3600,  # Result expiration time: 1 hour
        )

        # Initialize stats Redis client
        self._started = False

    async def startup(self):
        """Start Celery connection (simulates original broker.startup)."""
        if not self._started:
            self.celery_app.connection().ensure_connection()
        self._started = True
        return True

    async def enqueue(self, task_name: str, *args, **kwargs) -> str:
        """Enqueue a task, only manually maintains queued count (core compatibility logic).

        Args:
            task_name: Name of the task to enqueue.
            *args: Positional arguments for the task.
            **kwargs: Keyword arguments for the task.

        Returns:
            Job ID of the enqueued task.

        Raises:
            ValueError: If task not found.
        """
        await self.startup()

        # Find Celery task
        try:
            task = self.celery_app.tasks[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found")

        # Submit async task
        result = task.apply_async(args=args, kwargs=kwargs)
        job_id = result.id

        # Store task metadata (retains original requirement: task_name + parameters)
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
        """Get task statistics (fully compatible with original interface).

        Returns:
            Dictionary with queued, running, completed, failed counts and results.
        """
        await self.startup()

        stats = {"queued": 0, "running": 0, "completed": 0, "failed": 0}

        # Match task parameter keys to filter valid job IDs
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
        """Clear all statistics and task data.

        Returns:
            Number of deleted keys.
        """
        return await self.redis_storage_client.redis.delete("celery")

    async def get_result(self, job_id: str) -> dict:
        """Get task result (includes parameters, compatible with original interface).

        Args:
            job_id: Task job ID.

        Returns:
            Dictionary with task result, status, error, and metadata.
        """
        # Get Celery task result
        result = AsyncResult(job_id, app=self.celery_app)

        # Construct result dictionary
        result_dict = {
            "job_id": job_id,
            "status": result.status,
            "result": result.result if result.successful() else None,
            "error": str(result.result) if result.failed() else None,
            "traceback": result.traceback,
            "date_done": result.date_done.isoformat() if result.date_done else None,
        }

        # Merge task metadata (task_name + parameters)
        task_params = await self.redis_storage_client.get_task_data(job_id)
        result_dict.update(task_params)

        return result_dict

    async def remove_result(self, job_id: str):
        """Remove task result and parameters (compatible with original interface).

        Args:
            job_id: Task job ID.
        """
        # Delete task parameters
        await self.redis_storage_client.remove_task_data(job_id)

    async def abort(self, job_id: str):
        """Abort a running task (core requirement).

        Args:
            job_id: Task job ID.

        Raises:
            RuntimeError: If abort fails.
        """
        await self.startup()

        # Try to directly terminate running task
        try:
            result = AsyncResult(job_id, app=self.celery_app)
            result.revoke(terminate=True, signal="SIGTERM")
        except OperationalError as e:
            raise RuntimeError(f"Failed to abort task {job_id}: {str(e)}")

    async def shutdown(self):
        """Gracefully close connections (compatible with original interface)."""
        await self.startup()
        # Close Redis connection
        await self.redis_storage_client.close()
        # Close Celery connection
        self.celery_app.close()

    def register_task(self, async_func: Callable, task_name: str, ignore_result: bool = False):
        """Register an async task (compatible with original interface).

        Args:
            async_func: Async function to register as task.
            task_name: Name for the task.
            ignore_result: Whether to ignore task result.

        Returns:
            Registered Celery task.
        """
        @self.celery_app.task(
            name=task_name, bind=True, ignore_result=ignore_result, track_started=True
        )
        def palitra_wrapped(task, *a, **k):
            return palitra.run(async_func(*a, **k))

        palitra_wrapped.__orig__ = async_func

        return palitra_wrapped

    def get_tasks(self):
        """Get registered tasks with their signatures.

        Returns:
            Dictionary mapping task names to their parameter signatures.
        """
        ret = {}
        for name, task in self.celery_app.tasks.items():
            if "." in name:
                continue
            func = getattr(task, "__orig__", lambda: 0)
            ret[name] = inspect_function_signature(func)
        return ret

    def get_wsrouter(self) -> APIRouter:
        """Get WebSocket router for job status updates.

        Returns:
            APIRouter with WebSocket endpoints.
        """
        wsrouter = APIRouter(prefix="/jobs")

        @wsrouter.websocket("/stats")
        async def jobs_websocket(websocket: WebSocket, token: str = Query(None)):
            """WebSocket endpoint for real-time job statistics.

            Args:
                websocket: WebSocket connection.
                token: Authentication token.
            """
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
        """Generate FastAPI router (fully compatible with original interface).

        Returns:
            APIRouter with task management endpoints.
        """
        router = APIRouter(prefix="/worker", tags=["worker"])

        @router.get("/")
        def list_tasks():
            """List all registered tasks.

            Returns:
                Dictionary of task names to their parameter signatures.
            """
            return self.get_tasks()

        @router.post("/{task_name}")
        async def add_task(task_name: str, payload: dict = Body(...)):
            """Enqueue a task.

            Args:
                task_name: Name of the task.
                payload: Task parameters.

            Returns:
                Job ID of the enqueued task.
            """
            return await self.enqueue(task_name, **payload)

        @router.get("/{job_id}")
        async def get_result(job_id: str):
            """Get task result.

            Args:
                job_id: Task job ID.

            Returns:
                Task result dictionary.
            """
            return await self.get_result(job_id)

        @router.delete("/")
        async def clear_all():
            """Clear all task data.

            Returns:
                Number of deleted keys.
            """
            return await self.clear()

        @router.delete("/{job_id}")
        async def remove_result(job_id: str):
            """Remove task result.

            Args:
                job_id: Task job ID.

            Returns:
                Result of remove operation.
            """
            return await self.remove_result(job_id)

        @router.post("/{job_id}/abort")
        async def abort_job(job_id: str):
            """Abort a running task.

            Args:
                job_id: Task job ID.

            Returns:
                Abort status message.
            """
            await self.abort(job_id)
            return {"status": "success", "message": f"Task {job_id} aborted"}

        return router


worker_manager = AsyncWorkerManager(config.redis + "/0")
celery = worker_manager.celery_app
