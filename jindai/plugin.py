"""Plugin platform for Jindai application.

This module provides:
- Plugin: Base class for plugins
- PluginManager: Manager for loading and managing plugins
- prepare_plugins: Function to initialize the plugin system
"""

import asyncio
import glob
import inspect
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, cast

from fastapi import APIRouter, Depends

from .helpers import get_context
from .models import get_current_admin
from .config import config
from .pipeline import Pipeline, PipelineStage
from .storage import storage


class Plugin:
    """Base class for plugins.

    Plugins can register filters, callbacks, and pipeline stages
    to extend the functionality of the Jindai application.
    """

    def __init__(self, pmanager: "PluginManager", **conf) -> None:
        """Initialize plugin.

        Args:
            pmanager: Plugin manager instance.
            **conf: Plugin configuration.
        """
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
        """Register a filter for the UI.

        Args:
            name: Filter name.
            keybind: Keyboard shortcut for UI.
            format_string: Format string for filter display.
            icon: MDI icon for UI button.
            handler: Handler function that returns a list.
        """
        self.pmanager.filters[name] = {
            "name": name,
            "keybind": keybind,
            "format": format_string,
            "icon": icon,
            "handler": handler,
        }

    def register_callback(self, name: str, handler: Callable) -> None:
        """Register a callback function.

        Args:
            name: Callback name.
            handler: Handler function.
        """
        self.pmanager.callbacks[name].append(handler)

    def register_pipelines(self, pipeline_classes) -> None:
        """Register pipeline stage classes.

        Args:
            pipeline_classes: Pipeline stage class or iterable of classes.
        """
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
                
    def register_routes(self, target: APIRouter) -> None:
        """Register routes to target router"""
        pass

    async def startup(self, app: Any = None) -> None:
        """Lifecycle hook called by PluginManager.startup() (app lifespan).

        Default: no-op.  Override to start background work; see WorkerPlugin.
        """

    async def shutdown(self, app: Any = None) -> None:
        """Lifecycle hook called by PluginManager.shutdown() (app lifespan).

        Default: no-op.  Override to release resources started in startup().
        """


class WorkerPlugin(Plugin):
    """Base class for plugins that own resident background loops (workers).

    A worker plugin only has to implement :meth:`worker_loop` (an ``async def``
    that runs until cancelled) and declare ``worker_name`` / ``enabled_key``.
    PluginManager starts it from the FastAPI lifespan and stops it on shutdown,
    which matters because plugin *construction* happens at import time (there is
    no event loop yet), so the asyncio tasks must be created in startup().

    Attributes:
        worker_name: Human readable name used in logs / status endpoints.
        enabled_key: config.yaml key that switches the worker on/off; the same
            key can be overridden per plugin with ``enabled:`` in the plugin's
            own config section (its class name).
    """

    #: Never instantiate the base class (see PluginManager plugin loading).
    abstract = True
    worker_name: str = ""
    enabled_key: str = ""
    #: Value used when neither config.yaml nor the plugin's own config section
    #: defines the switch.
    enabled_default: bool = True

    #: Restart backoff for a loop that dies on an uncaught exception.
    restart_base_sleep: float = 10.0
    restart_max_sleep: float = 600.0

    def __init__(self, pmanager: "PluginManager", **conf) -> None:
        super().__init__(pmanager, **conf)
        self.worker_tasks: List[asyncio.Task] = []
        self.worker_crashes: int = 0
        self._stopping: bool = False

    # ------------------------------------------------------------------ switch
    @property
    def worker_enabled(self) -> bool:
        """Whether the resident loop should run (config.yaml switch)."""
        if "enabled" in self.config:
            return bool(self.config["enabled"])
        if self.enabled_key:
            return bool(getattr(config, self.enabled_key, self.enabled_default))
        return True

    # --------------------------------------------------------------------- loop
    async def worker_loop(self) -> None:  # pragma: no cover - abstract
        """The resident loop. Must return/raise for the worker to stop."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement worker_loop()")

    async def _run_worker(self) -> None:
        """Task body: keep the loop alive, restart it after a backoff."""
        name = self.worker_name or type(self).__name__
        while not self._stopping:
            try:
                await self.worker_loop()
                return  # clean exit (e.g. disabled at runtime) - do not restart
            except asyncio.CancelledError:
                logging.info("%s: background worker cancelled.", name)
                raise
            except Exception as ex:  # noqa: BLE001 - a dead loop is worse
                self.worker_crashes += 1
                sleep_s = min(
                    self.restart_base_sleep * (2 ** min(self.worker_crashes - 1, 6)),
                    self.restart_max_sleep)
                logging.exception(
                    "%s: background worker died (%s: %s) - restarting in %.0fs "
                    "(crash #%d)", name, type(ex).__name__, ex, sleep_s,
                    self.worker_crashes)
                await asyncio.sleep(sleep_s)

    # ---------------------------------------------------------------- lifecycle
    async def startup(self, app: Any = None) -> None:
        """Start the resident loop in the running event loop (lifespan)."""
        name = self.worker_name or type(self).__name__
        if not self.worker_enabled:
            logging.info(
                "%s: background worker disabled (config.%s = false).",
                name, self.enabled_key or self.config.get("enabled"))
            return
        if any(not t.done() for t in self.worker_tasks):
            logging.info("%s: background worker already running.", name)
            return
        self._stopping = False
        self.worker_tasks = [
            asyncio.create_task(self._run_worker(), name=f"{name}_worker")]
        logging.info("%s: background worker scheduled.", name)

    async def shutdown(self, app: Any = None) -> None:
        """Cancel the resident loop(s) and wait for them to unwind."""
        self._stopping = True
        tasks = [t for t in self.worker_tasks if not t.done()]
        self.worker_tasks = []
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logging.info("%s: background worker stopped.",
                     self.worker_name or type(self).__name__)

    # ------------------------------------------------------------------- status
    def worker_details(self) -> Dict[str, Any]:
        """Extra per-plugin status merged into worker_status()."""
        return {}

    def worker_status(self) -> Dict[str, Any]:
        """Status payload served by GET /plugins/workers."""
        out: Dict[str, Any] = {
            "name": self.worker_name or type(self).__name__,
            "plugin": type(self).__name__,
            "enabled": self.worker_enabled,
            "running": any(not t.done() for t in self.worker_tasks),
            "crashes": self.worker_crashes,
        }
        try:
            out.update(self.worker_details())
        except Exception as ex:  # noqa: BLE001 - status must not break the API
            out["details_error"] = f"{type(ex).__name__}: {ex}"
        return out


class PluginManager:
    """Manager for loading and managing plugins.

    Handles plugin discovery, loading, and provides
    FastAPI routes for plugin management.
    """

    def __init__(self, plugin_ctx: dict) -> None:
        """Initialize plugin manager.

        Args:
            plugin_ctx: Plugin context dictionary mapping names to classes.
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
                params = getattr(config, cast(str, plugin_name), {})

            plugin_cls = None
            if isinstance(plugin_name, str):
                plugin_cls = plugin_ctx.get(plugin_name)
            elif isinstance(plugin_name, type):
                plugin_cls = plugin_name

            if not plugin_cls:
                logging.info(f"Plugin {plugin_name} not found.")
                continue

            # Base/mixin plugin classes opt out of instantiation by declaring
            # `abstract = True` in their own body (see WorkerPlugin).
            if plugin_cls.__dict__.get("abstract", False):
                logging.debug("Skipping abstract plugin class: %s", plugin_name)
                continue

            try:
                plugin_instance = plugin_cls(self, **params)
                self.plugins.append(plugin_instance)
                logging.info(f"Registered plugin: {type(plugin_instance).__name__}")
            except Exception as ex:
                logging.error(f"Error while registering plugin: {plugin_name} {ex}")
                continue

    async def startup(self, app: Any = None) -> None:
        """Run every plugin's startup() hook (called from the app lifespan).

        Worker plugins use this to create their asyncio background tasks: plugin
        construction happens at import time, when no event loop is running yet.
        A plugin that fails to start must not take the web service down.

        Args:
            app: The FastAPI application, passed through to the hooks.
        """
        for plugin_instance in self.plugins:
            try:
                result = plugin_instance.startup(app)
                if inspect.isawaitable(result):
                    await result
            except Exception as ex:  # noqa: BLE001 - keep serving regardless
                logging.error("Error while starting plugin: %s %s",
                              type(plugin_instance).__name__, ex)

    async def shutdown(self, app: Any = None) -> None:
        """Run every plugin's shutdown() hook, in reverse load order.

        Args:
            app: The FastAPI application, passed through to the hooks.
        """
        for plugin_instance in reversed(self.plugins):
            try:
                result = plugin_instance.shutdown(app)
                if inspect.isawaitable(result):
                    await result
            except Exception as ex:  # noqa: BLE001 - shutdown must not stall
                logging.error("Error while stopping plugin: %s %s",
                              type(plugin_instance).__name__, ex)

    def worker_status(self) -> List[Dict[str, Any]]:
        """Collect the status of all plugin-owned background workers.

        Returns:
            One dict per WorkerPlugin (empty when no worker plugin is loaded).
        """
        return [pl.worker_status() for pl in self.plugins
                if isinstance(pl, WorkerPlugin)]


    def register_routes(self, target: APIRouter):
        """
        Register FastAPI router for plugin management endpoints.
        """
        
        # Register routes for every plugin first
        for plugin_instance in self.plugins:
            plugin_instance.register_routes(target)
        
        # Register routes for self
        router = APIRouter(prefix='/plugins', tags=['Plugins'], dependencies=[Depends(get_current_admin)])

        @router.get("/")
        def plugin_list():
            """List all registered plugins.

            Returns:
                List of plugin class names.
            """
            return [type(pl).__name__ for pl in self.plugins]

        @router.get("/workers", tags=["Plugins"])
        def plugin_workers():
            """List the resident background workers owned by plugins.

            Returns:
                One status dict per WorkerPlugin (enabled/running/details).
            """
            return self.worker_status()


        @router.post("/")
        def plugin_install(url: str):
            """Install a plugin from URL.

            Args:
                url: URL to plugin zip file.

            Returns:
                True if installation succeeded.
            """
            self.install(url)
            return True

        @router.get("/pipeline", tags=["Plugins"])
        async def get_pipeline_info():
            """Get pipeline stage information.

            Returns:
                Dictionary mapping module names to pipeline stage specs.
            """
            from .pipeline import Pipeline

            ctx = Pipeline.ctx
            result = defaultdict(dict)

            for key, val in ctx.items():

                module_doc = (
                    sys.modules[val.__module__].__doc__ if hasattr(val, "__module__") else None
                )
                name = (
                    module_doc or val.__module__.split(".")[-1]
                    if hasattr(val, "__module__")
                    else key
                ).strip().split('\n')[0]
                
                result[name][key] = val.get_spec()

            return result

        target.include_router(router)

    def __iter__(self) -> Iterator:
        """Iterate through loaded plugins.

        Yields:
            Plugin instances.
        """
        yield from self.plugins

    def install(self, file_or_url: str) -> None:
        """Install plugin from local file storage or URL.

        Args:
            file_or_url: Path to local file or URL to download plugin.

        Raises:
            ValueError: If no main directory found in plugin package.
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
    """Prepare and load all plugins.

    Returns:
        PluginManager instance with loaded plugins.
    """
    from .plugin import Plugin, PluginManager

    if os.path.exists("restarting"):
        os.unlink("restarting")
    plugin_ctx = get_context("plugins", Plugin)
    return PluginManager(plugin_ctx)


plugins = prepare_plugins()
