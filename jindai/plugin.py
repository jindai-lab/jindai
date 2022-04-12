"""Plugin platform for jindai"""
from collections import defaultdict
from typing import Callable
from flask import Response, Flask

from .helpers import rest
from .pipeline import Pipeline, PipelineStage
from .config import instance as config


class Plugin:
    """Base class for plugins
    """

    def __init__(self, pmanager, **conf) -> None:
        self.config = conf
        self.pmanager = pmanager

    def register_filter(self, name: str, keybind: str, format_string: str,
                        icon: str, handler: Callable) -> None:
        """Register filter

        :param name: filter name
        :type name: str
        :param keybind: keybind for ui
        :type keybind: str
        :param format_string: format string,
            use {imageitem} and {paragraph} for the selected item
        :type format_string: str
        :param icon: mdi icon for ui button
        :type icon: str
        :param handler: handler function
        :type handler: Callable
        """
        self.pmanager.filters[name] = {
            'name': name,
            'keybind': keybind,
            'format': format_string,
            'icon': icon,
            'handler': handler
        }

    def register_callback(self, name: str, handler: Callable):
        """Register callback function

        :param name: callback name
        :type name: str
        :param handler: handler function
        :type handler: Callable
        """
        self.pmanager.callbacks[name].append(handler)

    def register_pipelines(self, pipeline_classes):
        """Register pipeline stage"""
        if isinstance(pipeline_classes, dict):
            pipeline_classes = pipeline_classes.values()

        for cls in pipeline_classes:
            if isinstance(cls, type) and issubclass(cls, PipelineStage) \
                    and cls is not PipelineStage:
                Pipeline.ctx[cls.__name__] = cls


class PluginManager:
    """Plugin manager"""

    def __init__(self, plugin_ctx: dict, app: Flask) -> None:
        self.plugins = []
        self.filters = {}
        self.callbacks = defaultdict(list)
        self.app = app

        @app.route('/api/plugins/styles.css')
        def plugins_style():
            """Returns css from all enabled plugins

            Returns:
                Response: css document
            """
            css = '\n'.join(
                [handler() or '' for handler in self.callbacks['css']])
            return Response(css, mimetype='text/css')

        @app.route('/api/plugins/filters', methods=["GET", "POST"])
        @rest()
        def plugin_pages():
            """Returns names for special filters in every plugins
            """
            return [
                dict(spec, handler='')
                for spec in self.filters.values()
            ]

        # load plugins

        pls = []
        for plugin_name in config.plugins:
            if plugin_name == '*':
                pls += list(plugin_ctx.keys())
            elif plugin_name.startswith('~'):
                if plugin_name[1:] in pls:
                    pls.remove(plugin_name[1:])
            else:
                pls.append(plugin_name)

        for plugin_name in pls:
            if isinstance(plugin_name, tuple) and len(plugin_name) == 2:
                plugin_name, params = plugin_name
            else:
                params = {}

            if isinstance(plugin_name, str):
                plugin_cls = plugin_ctx.get(plugin_name)
            elif isinstance(plugin_name, type):
                plugin_cls = plugin_name

            if not plugin_cls:
                print('Plugin', plugin_name, 'not found.')
                continue

            try:
                plugin_instance = plugin_cls(self, **params)
                self.plugins.append(plugin_instance)
                print('Registered plugin:', type(plugin_instance).__name__)
            except Exception as ex:
                print('Error while registering plugin:', plugin_name, ex)
                continue

    def __iter__(self):
        """Iterate through loaded plugins"""
        yield from self.plugins
