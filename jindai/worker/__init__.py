"""WorkerManager - A distributed task worker using Redis."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse

from .worker_manager import WorkerManager
from .logger import RedisLogger, TaskLoggerAdapter
from .models import TaskStatus, TaskMetadata

from jindai.config import config
from jindai.models import get_current_admin
from jindai.plugin import Plugin
from jindai.maintenance import maintenance_manager

# Global worker manager instance
manager = WorkerManager(redis_url=config.redis + '/0')

# Expose worker_manager for use by other plugins
worker_manager = manager


# ==================== Task Submission Endpoints ====================

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
