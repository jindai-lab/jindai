"""Jindai core package.

This package provides the main application components including:
- API web service (FastAPI)
- Database models and session management
- Pipeline processing system
- Plugin management
- Task execution engine
- File storage management
"""

from . import helpers, models
from .app import app, serve_static
from .config import config
from .models import Dataset, Paragraph, TaskDBO, UserInfo, get_db_session, is_uuid_literal
from .pipeline import Pipeline, PipelineStage
from .plugin import Plugin, PluginManager
from .resources import router, wsrouter
from .storage import storage
from .task import Task

__all__ = [
    "app",
    "config",
    "storage",
    "router",
    "wsrouter",
    "Pipeline",
    "PipelineStage",
    "Plugin",
    "PluginManager",
    "Task",
    "Dataset",
    "Paragraph",
    "TaskDBO",
    "UserInfo",
    "get_db_session",
    "serve_static",
    "helpers",
    "models",
]
