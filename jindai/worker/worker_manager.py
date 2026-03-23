"""WorkerManager - A distributed task worker using Redis."""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from fastapi import FastAPI

import redis

from jindai.helpers import inspect_function_signature

from .logger import RedisLogger, TaskLoggerAdapter
from .models import TaskMetadata, TaskStatus


class WorkerManager:
    """
    A distributed task worker manager that uses Redis for task queue management,
    result storage, and logging.
    
    Features:
    - Register async functions as tasks
    - Process tasks from Redis queue
    - Store task metadata and results
    - Provide logger for tasks to publish messages
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize WorkerManager.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            redis_url: Redis URL (overrides host/port/db)
        """
        if redis_url:
            self.redis = redis.from_url(redis_url, decode_responses=True)
        else:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
            )
        
        # Task registry: name -> async function
        self.tasks: Dict[str, Callable] = {}
        
        # Task metadata cache
        self._task_metadata_cache: Dict[str, TaskMetadata] = {}
        
        # Worker state
        self._running = False
        self._workers: List[asyncio.Task] = []
        
        # Queue key
        self._queue_key = "task:queue"
        
        # Default logger (can be overridden per task)
        self.logger = RedisLogger(self.redis)
    
    def register_task(
        self,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> Callable:
        """
        Register an async function as a task.
        
        Can be used as a decorator or directly:
            @wm.register_task(name="add")
            async def add(a, b):
                return a + b
            
            wm.register_task(add, name="subtract")
        
        Args:
            func: The async function to register
            name: Task name (defaults to function name)
            
        Returns:
            The registered function
        """
        def decorator(f: Callable) -> Callable:
            task_name = name or f.__name__
            self.tasks[task_name] = f
            return f
        
        if func is not None:
            # Called directly: register_task(func, name)
            task_name = name or func.__name__
            self.tasks[task_name] = func
            return func
        
        # Called as decorator: @register_task(name="...")
        return decorator
    
    def submit_task(
        self,
        name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a task to the Redis queue.
        
        Args:
            name: Task name (must be registered)
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            
        Returns:
            Task ID (UUID)
        """
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' is not registered")
        
        task_id = str(uuid.uuid4())
        args = args or []
        kwargs = kwargs or {}
        
        # Create task metadata
        metadata = TaskMetadata(
            task_id=task_id,
            task_name=name,
            status=TaskStatus.QUEUED,
            args=args,
            kwargs=kwargs,
            created_at=datetime.now(),
        )
        
        # Store metadata in Redis
        self._set_task_metadata(task_id, metadata)
        
        # Push to Redis queue (single global queue)
        self.redis.rpush(self._queue_key, task_id)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status and metadata.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task metadata dictionary
        """
        metadata = self._get_task_metadata(task_id)
        if metadata is None:
            return {}
        return metadata.to_dict()
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result, or None if not found or not completed
        """
        result_key = f"task:result:{task_id}"
        result = self.redis.get(result_key)
        if result:
            return json.loads(result)
        return None
    
    def get_task_logs(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get task log messages.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of log messages
        """
        logger = RedisLogger(self.redis, task_id=task_id)
        return logger.get_logs(task_id)
    
    def _get_task_metadata(self, task_id: str) -> Optional[TaskMetadata]:
        """Get task metadata from Redis."""
        meta_key = f"task:meta:{task_id}"
        data = self.redis.hgetall(meta_key)
        if not data:
            return None
        return TaskMetadata.from_dict(data)
    
    def _set_task_metadata(self, task_id: str, metadata: TaskMetadata) -> None:
        """Set task metadata in Redis."""
        meta_key = f"task:meta:{task_id}"
        data = metadata.to_dict()
        self.redis.hset(meta_key, mapping=data)
    
    async def _process_task(self, task_id: str, task_name: str) -> None:
        """
        Process a single task.
        
        Args:
            task_id: Task ID
            task_name: Task name
        """
        # Get task function
        if task_name not in self.tasks:
            self._update_task_status(task_id, TaskStatus.FAILED, error=f"Task '{task_name}' not found")
            return
        
        func = self.tasks[task_name]
        
        # Get task metadata
        metadata = self._get_task_metadata(task_id)
        if metadata is None:
            self._update_task_status(task_id, TaskStatus.FAILED, error="Task metadata not found")
            return
        
        # Update status to processing
        self._update_task_status(task_id, TaskStatus.PROCESSING, started_at=datetime.now())
        
        # Create logger for this task
        task_logger = RedisLogger(self.redis, task_id=task_id)
        logger_adapter = TaskLoggerAdapter(task_logger)
        
        try:
            # Execute the task
            if asyncio.iscoroutinefunction(func):
                result = await func(*metadata.args, **metadata.kwargs)
            else:
                result = func(*metadata.args, **metadata.kwargs)
            
            # Store result
            self._update_task_status(
                task_id,
                TaskStatus.SUCCESS,
                result=result,
                completed_at=datetime.now(),
            )
            
        except Exception as e:
            # Handle task failure
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self._update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=error_msg,
                completed_at=datetime.now(),
            )
    
    def _update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """
        Update task status in Redis.
        
        Args:
            task_id: Task ID
            status: New status
            result: Task result (optional)
            error: Error message (optional)
            started_at: Start time (optional)
            completed_at: Completion time (optional)
        """
        metadata = self._get_task_metadata(task_id)
        if metadata is None:
            return
        
        metadata.status = status
        if started_at:
            metadata.started_at = started_at
        if completed_at:
            metadata.completed_at = completed_at
        if result is not None:
            metadata.result = result
        if error:
            metadata.error = error
        
        self._set_task_metadata(task_id, metadata)
        
        # Store result if successful
        if status == TaskStatus.SUCCESS and result is not None:
            result_key = f"task:result:{task_id}"
            self.redis.set(result_key, json.dumps(result))
    
    async def _worker_loop(
        self,
        task_names: List[str],
        poll_interval: float = 1.0,
    ) -> None:
        """
        Worker loop that processes tasks from Redis queue.
        
        Args:
            task_names: List of task names to process
            poll_interval: Polling interval in seconds
        """
        while self._running:
            # Blocking pop from single global queue
            result = self.redis.blpop(self._queue_key, timeout=1)
            
            if result:
                _, task_id = result
                # Get task name from metadata
                metadata = self._get_task_metadata(task_id)
                if metadata:
                    await self._process_task(task_id, metadata.task_name)
            
            await asyncio.sleep(poll_interval)
    
    async def start_worker(
        self,
        task_names: Optional[List[str]] = None,
        concurrency: int = 1,
        poll_interval: float = 1.0,
    ) -> None:
        """
        Start the worker to process tasks.
        
        Args:
            task_names: List of task names to process. If None, processes all registered tasks.
            concurrency: Number of concurrent workers
            poll_interval: Polling interval in seconds
        """
        if self._running:
            raise RuntimeError("Worker is already running")
        
        self._running = True
        task_names = task_names or list(self.tasks.keys())
        
        # Create worker tasks directly (non-blocking)
        for _ in range(concurrency):
            task = asyncio.create_task(
                self._worker_loop(task_names, poll_interval)
            )
            self._workers.append(task)
    
    def stop_worker(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all worker tasks
        for task in self._workers:
            task.cancel()
        
        # Wait for all workers to finish
        if self._workers:
            import asyncio
            asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> Any:
        """Lifespan context for worker management.
        
        Starts the worker when the app starts and stops it when the app shuts down.
        
        Args:
            app: FastAPI application.
        
        Yields:
            None.
        """
        # Start the worker
        await self.start_worker()
        yield
        # Stop the worker on shutdown
        self.stop_worker()
    
    def get_registered_tasks(self) -> List[str]:
        """Get list of registered task names."""
        return {
            k: inspect_function_signature(v)
            for k, v in self.tasks.items()
        }
    
    def get_queue_length(self, task_name: Optional[str] = None) -> int:
        """
        Get the length of the task queue.
        
        Args:
            task_name: Optional task name (ignored, kept for backward compatibility)
            
        Returns:
            Number of tasks in the queue
        """
        return self.redis.llen(self._queue_key)
    
    def clear_tasks(self) -> bool:
        """
        Clear the task queue.
        
        Returns:
            Number of tasks cleared
        """
        self.redis.delete(self._queue_key)
    
        for task_id in self.list_tasks():
            self.delete_task(task_id)
        
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.
        
        - For queued or completed tasks: directly delete
        - For processing tasks: stop and delete
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was deleted, False if task not found
        """
        metadata = self._get_task_metadata(task_id)
        if metadata is None:
            return False
        
        # If task is processing, stop it
        if metadata.status == TaskStatus.PROCESSING:
            # Note: In a real distributed system, you would need
            # a mechanism to signal the worker to stop the task.
            # For now, we'll just mark it as failed and delete.
            self._update_task_status(
                task_id,
                TaskStatus.FAILED,
                error="Task was deleted while processing",
                completed_at=datetime.now(),
            )
        
        # Remove from queue if queued
        self.redis.lrem(self._queue_key, 0, task_id)
        
        # Delete metadata and result
        meta_key = f"task:meta:{task_id}"
        result_key = f"task:result:{task_id}"
        log_key = f"task:log:{task_id}"
        
        self.redis.delete(meta_key, result_key, log_key)
        
        return True
    
    def get_status_summary(self) -> Dict[str, int]:
        """
        Get summary of task statuses.
        
        Returns:
            Dictionary with status counts: {queued, processing, success, failed}
        """
        summary = {
            "queued": 0,
            "processing": 0,
            "success": 0,
            "failed": 0,
            "results": []
        }
        
        # Scan all task metadata keys
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match="task:meta:*", count=100)
            for key in keys:
                data = self.redis.hgetall(key)
                if data and "status" in data:
                    status = data["status"]
                    if status in summary:
                        summary[status] += 1
                summary['results'].append(data)
            
            if cursor == 0:
                break
        
        return summary
    
    def list_tasks(self) -> List[str]:
        """
        List all task IDs from task:meta keys.
        
        Returns:
            List of task IDs (UUID strings)
        """
        task_ids = []
        
        # Scan all task metadata keys
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match="task:meta:*", count=100)
            for key in keys:
                # Extract task_id from key pattern "task:meta:{task_id}"
                task_id = key.replace("task:meta:", "")
                task_ids.append(task_id)
            
            if cursor == 0:
                break
        
        return task_ids
