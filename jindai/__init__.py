"""Jindai core package"""
from . import helpers, models
from .config import instance as config
from .models import Dataset, Paragraph, TaskDBO, UserInfo, get_db
from .pipeline import Pipeline, PipelineStage
from .plugin import Plugin, PluginManager
from .storage import instance as storage
from .task import Task
from .app import app, serve_static
from .resources import router

app.include_router(router)
app.get('/{path:path}')(serve_static)
