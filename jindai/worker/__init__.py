"""WorkerManager - A distributed task worker using Redis."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse

from .worker_manager import WorkerManager
from .logger import RedisLogger, TaskLoggerAdapter
from .models import TaskStatus, TaskMetadata

from jindai.app import router as api_router
from jindai.config import config
from jindai.app import get_current_admin
from jindai.plugin import Plugin
from jindai.maintenance import maintenance_manager

# Global worker manager instance
manager = WorkerManager(redis_url=config.redis + '/0')

# API Router for worker management
router = APIRouter(prefix='/worker', tags=['WorkerManager'])

# Expose worker_manager for use by other plugins
worker_manager = manager


# ==================== Task Submission Endpoints ====================

@router.post("/tasks")
async def submit_task(
    task_name: str = Body(embed=True),
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_admin),
) -> Dict[str, str]:
    """
    Submit a task to the Redis queue.
    
    Args:
        task_name: Name of the registered task
        args: Positional arguments for the task
        kwargs: Keyword arguments for the task
        
    Returns:
        Dictionary with task_id
        
    Raises:
        HTTPException: If task is not registered
    """
    try:
        task_id = manager.submit_task(task_name, args or [], kwargs or {})
        return {"task_id": task_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/tasks")
async def list_tasks(
    offset: int = 0,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    List all tasks with metadata.
    
    Args:
        offset: Pagination offset
        limit: Page size limit
        
    Returns:
        Dictionary with results and count
    """
    # Scan all task metadata keys
    cursor = offset
    tasks = []
    
    while True:
        cursor, keys = manager.redis.scan(cursor, match="task:meta:*", count=limit)
        for key in keys:
            data = manager.redis.hgetall(key)
            if data and "task_id" in data:
                tasks.append(TaskMetadata.from_dict(data).to_dict())
        
        if cursor == 0:
            break
        
    total = len(tasks)
    
    return {"results": tasks, "count": total, "total": total}


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
) -> Dict[str, Any]:
    """
    Get task status and metadata.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task metadata dictionary
    """
    metadata = manager._get_task_metadata(task_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return metadata.to_dict()


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
) -> Dict[str, bool]:
    """
    Delete a task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Dictionary with success flag
    """
    success = manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"success": True}


# ==================== Task Result Endpoints ====================

@router.get("/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
) -> Dict[str, Any]:
    """
    Get task result.
    
    Args:
        task_id: Task ID
        
    Returns:
        Dictionary with result or error
    """
    metadata = manager._get_task_metadata(task_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    if metadata.status == TaskStatus.SUCCESS:
        result = manager.get_task_result(task_id)
        return {"status": "success", "result": result}
    elif metadata.status == TaskStatus.FAILED:
        return {"status": "failed", "error": metadata.error}
    else:
        return {"status": metadata.status.value, "result": None}


@router.get("/tasks/{task_id}/logs")
async def get_task_logs(
    task_id: str,
) -> List[Dict[str, Any]]:
    """
    Get task log messages.
    
    Args:
        task_id: Task ID
        
    Returns:
        List of log messages
    """
    return manager.get_task_logs(task_id)


# ==================== Worker Management Endpoints ====================

@router.get("/registered")
async def list_registered_tasks() -> Dict[str, Dict]:
    """
    List all registered task names.
    
    Returns:
        List of task names
    """
    return manager.get_registered_tasks()


@router.get("/queues")
async def list_queues(
) -> Dict[str, int]:
    """
    Get queue lengths for all registered tasks.
    
    Returns:
        Dictionary mapping task names to queue lengths
    """
    queues = {}
    for task_name in manager.get_registered_tasks():
        queues[task_name] = manager.get_queue_length(task_name)
    return queues


@router.post("/queues/{task_name}/clear")
async def clear_queue(
    task_name: str,
) -> Dict[str, int]:
    """
    Clear a task queue.
    
    Args:
        task_name: Task name
        
    Returns:
        Dictionary with number of tasks cleared
    """
    cleared = manager.clear_queue(task_name)
    return {"cleared": cleared}


@router.get("/status")
async def get_status_summary() -> Dict[str, int]:
    """
    Get summary of task statuses.
    
    Returns:
        Dictionary with status counts: {queued, processing, success, failed}
    """
    return manager.get_status_summary()


# ==================== WebSocket Endpoints for Real-time Logging ====================

@router.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time task log streaming.
    
    Clients can subscribe to logs by sending a JSON message with:
    - action: "subscribe" or "unsubscribe"
    - task_id: (optional) specific task ID to subscribe to
    - all: (optional) if true, subscribe to all logs
    
    Example:
        {"action": "subscribe", "task_id": "abc-123"}
        {"action": "unsubscribe", "task_id": "abc-123"}
    """
    await websocket.accept()
    
    # Track subscriptions
    subscriptions = set()
    all_logs = False
    
    # Redis pub/sub
    pubsub = manager.redis.pubsub()
    
    try:
        while True:
            # Check for incoming messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=5.0
                )
                message = json.loads(data)
                
                action = message.get("action")
                task_id = message.get("task_id")
                
                if action == "subscribe":
                    if task_id:
                        subscriptions.add(task_id)
                        # Subscribe to task-specific log channel
                        log_key = f"task:log:{task_id}"
                        # We'll poll this key for new logs
                    elif message.get("all"):
                        all_logs = True
                        pubsub.subscribe("task:logs")
                
                elif action == "unsubscribe":
                    if task_id:
                        subscriptions.discard(task_id)
                    elif message.get("all"):
                        all_logs = False
                        pubsub.unsubscribe("task:logs")
                
                elif action == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue
            
            # Check for new logs
            if all_logs:
                try:
                    message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
                    if message:
                        log_entry = json.loads(message['data'])
                        await websocket.send_text(json.dumps({
                            "type": "log",
                            "data": log_entry
                        }))
                except Exception:
                    pass
            
            # Check task-specific logs
            for tid in list(subscriptions):
                logs = manager.get_task_logs(tid)
                if logs:
                    for log in logs:
                        await websocket.send_text(json.dumps({
                            "type": "log",
                            "task_id": tid,
                            "data": log
                        }))
                    
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.close()
        
        
@router.websocket('/stats')
async def worker_stats(websocket: WebSocket):
    """
    WebSocket endpoint for real-time worker statistics.
    
    Clients can subscribe to stats by sending a JSON message with:
    - action: "subscribe" or "unsubscribe"
    
    Example:
        {"action": "subscribe"}
        {"action": "unsubscribe"}
    """
    await websocket.accept()
    
    try:
        while True:
            stats = manager.get_status_summary()
            await websocket.send_text(json.dumps(stats))
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        pass

# Register routers

api_router.include_router(router)

tasks_to_reg = [
    (maintenance_manager.custom_task, "custom"),
    (maintenance_manager.ocr, "ocr"),
    (maintenance_manager.update_text_embeddings, "text_embedding"),
    (maintenance_manager.sync_terms, "sync_terms"),
    (maintenance_manager.update_pdate_from_url, "sync_pdate"),
    (maintenance_manager.sync_sources, "sync_sources"),
    (maintenance_manager.cleanup_unused_datasets, "cleanup_datasets"),
    (maintenance_manager.test_task, "test_task"),
]
for func_ref, name in tasks_to_reg:
    worker_manager.register_task(func_ref, name)
