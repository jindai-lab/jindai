"""Jindai core package"""
from . import helpers, models
from .app import app, serve_static
from .config import config
from .models import Dataset, Paragraph, TaskDBO, UserInfo, get_db_session
from .pipeline import Pipeline, PipelineStage
from .plugin import Plugin, PluginManager
from .resources import router, wsrouter
from .storage import storage
from .task import Task

app.include_router(router)
app.include_router(wsrouter)
app.get('/{path:path}')(serve_static)