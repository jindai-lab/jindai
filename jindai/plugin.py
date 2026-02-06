"""Plugin platform for jindai"""

import glob
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from typing import Callable, Iterator

from fastapi import APIRouter, Depends

from .helpers import get_context
from .app import get_current_admin
from .config import instance as config
from .pipeline import Pipeline, PipelineStage
from .storage import instance as storage


class Plugin:
    """Base class for plugins"""

    def __init__(self, pmanager, **conf) -> None:
        self.config = conf
        self.pmanager = pmanager

    def register_filter(
        self,
        name: str,
        keybind: str = "",
        format_string: str = "",
        icon: str = "",
        handler: Callable = lambda *_: list(),
    ) -> None:
        """Register filter

        :param name: filter name
        :type name: str
        :param keybind: keybind for ui
        :type keybind: str
        :param format_string: format string
        :type format_string: str
        :param icon: mdi icon for ui button
        :type icon: str
        :param handler: handler function
        :type handler: Callable
        """
        self.pmanager.filters[name] = {
            "name": name,
            "keybind": keybind,
            "format": format_string,
            "icon": icon,
            "handler": handler,
        }

    def register_callback(self, name: str, handler: Callable) -> None:
        """Register callback function

        :param name: callback name
        :type name: str
        :param handler: handler function
        :type handler: Callable
        """
        self.pmanager.callbacks[name].append(handler)

    def register_pipelines(self, pipeline_classes) -> None:
        """Register pipeline stage"""
        if isinstance(pipeline_classes, dict):
            pipeline_classes = pipeline_classes.values()

        for cls in pipeline_classes:
            if (
                isinstance(cls, type)
                and issubclass(cls, PipelineStage)
                and cls is not PipelineStage
                and not cls.__name__.startswith("_")
            ):
                Pipeline.ctx[cls.__name__] = cls


class PluginManager:
    """Plugin manager"""

    def __init__(self, plugin_ctx: dict) -> None:
        """Initialize plugin manager

        :param plugin_ctx: Plugin context dictionary
        :type plugin_ctx: dict 
        """
        self.plugins = []
        self.filters = {}
        self.callbacks = defaultdict(list)

        # load plugins

        pls = []
        for plugin_name in config.plugins:
            if plugin_name == "*":
                pls += list(plugin_ctx.keys())
            elif plugin_name.startswith("~"):
                if plugin_name[1:] in pls:
                    pls.remove(plugin_name[1:])
            elif plugin_name in plugin_ctx:
                pls.append(plugin_name)

        for plugin_name in pls:
            if isinstance(plugin_name, tuple) and len(plugin_name) == 2:
                plugin_name, params = plugin_name
            else:
                params = getattr(config, plugin_name, {})

            if isinstance(plugin_name, str):
                plugin_cls = plugin_ctx.get(plugin_name)
            elif isinstance(plugin_name, type):
                plugin_cls = plugin_name

            if not plugin_cls:
                print("Plugin", plugin_name, "not found.")
                continue

            try:
                plugin_instance = plugin_cls(self, **params)
                self.plugins.append(plugin_instance)
                print("Registered plugin:", type(plugin_instance).__name__)
            except Exception as ex:
                print("Error while registering plugin:", plugin_name, ex)
                continue
            
    def get_router(self):
        router = APIRouter(prefix='/plugins', tags=['Plugins'], dependencies=[Depends(get_current_admin)])
               
        @router.get("/")
        def plugin_list():
            return [type(pl).__name__ for pl in self.plugins]

        @router.post("/")
        def plugin_install(url):
            self.install(url)
            return True
        
        @router.get("/pipeline", tags=["Plugins"])
        async def get_pipeline_info():
            """提供流水线帮助信息"""
            from .pipeline import Pipeline

            ctx = Pipeline.ctx
            result = defaultdict(dict)

            for key, val in ctx.items():
                if key in ("DataSourceStage", "MediaItemStage"):
                    continue

                # 获取模块文档或模块名作为分类
                module_doc = (
                    sys.modules[val.__module__].__doc__ if hasattr(val, "__module__") else None
                )
                name = (
                    module_doc or val.__module__.split(".")[-1]
                    if hasattr(val, "__module__")
                    else key
                ).strip()

                result[name][key] = val.get_spec()

                return result

        return router

    def __iter__(self) -> Iterator:
        """Iterate through loaded plugins

        :return: Iterator over loaded plugin instances
        :rtype: Iterator
        """
        yield from self.plugins

    def install(self, file_or_url: str) -> None:
        """Install plugin from local file storage or URL

        :param file_or_url: Path to local file or URL to download plugin
        :type file_or_url: str
        :raises ValueError: If no main directory found in plugin package
        """
        if file_or_url.startswith('https://github.com/') and not file_or_url.endswith('.zip'):
            file_or_url += '/archive/refs/heads/main.zip'
        ziptar = storage.open(file_or_url)
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(ziptar, 'r') as zipped:
            zipped.extractall(path=tmpdir)

        maindir = [adir for adir in glob.glob(os.path.join(
            tmpdir, '*')) if os.path.isdir(adir) and not adir.startswith('.') and not adir.startswith('__')]
        if not maindir:
            shutil.rmtree(tmpdir)
            raise ValueError(
                f"No main directory found while attempting to install from {file_or_url}"
            )
        maindir = maindir[0]

        for dirname in ('plugins', 'sources'):
            source = os.path.join(maindir, dirname)
            if not os.path.exists(source):
                continue

            if dirname == 'sources':
                target = storage.safe_join('/').rstrip('/')
            else:
                target = './' + dirname

            shutil.copytree(source, target)

        maindir = maindir[0]

        for dirname in ("plugins", "sources"):
            source = os.path.join(maindir, dirname)
            if not os.path.exists(source):
                continue

            if dirname == "sources":
                target = storage.safe_join("/").rstrip("/")
            else:
                target = "./" + dirname

            shutil.copytree(source, target)

        shutil.rmtree(tmpdir)


def prepare_plugins() -> 'PluginManager':
    """Prepare plugins"""
    from .plugin import Plugin, PluginManager
    
    if os.path.exists("restarting"):
        os.unlink("restarting")
    plugin_ctx = get_context("plugins", Plugin)
    return PluginManager(plugin_ctx)


plugins = prepare_plugins()