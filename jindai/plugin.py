"""Plugin platform for jindai"""
import glob
from distutils.dir_util import copy_tree
import shutil
import os
import tempfile
import zipfile
from collections import defaultdict
from typing import Callable

from flask import Flask, Response

from .config import instance as config
from .helpers import rest
from .storage import instance as storage
from .pipeline import Pipeline, PipelineStage


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
            use {mediaitem} and {paragraph} for the selected items and paragraphs
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

        @app.route('/api/plugins')
        @rest(role='admin')
        def plugin_list():
            return [type(pl).__name__ for pl in self.plugins]

        @app.route('/api/plugins', methods=['POST'])
        @rest(role='admin')
        def plugin_install(url):
            self.install(url)
            return True

        # load plugins

        pls = []
        for plugin_name in config.plugins:
            if plugin_name == '*':
                pls += list(plugin_ctx.keys())
            elif plugin_name.startswith('~'):
                if plugin_name[1:] in pls:
                    pls.remove(plugin_name[1:])
            elif plugin_name in plugin_ctx:
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

    def install(self, file_or_url: str):
        """Install plugin from local file storage"""
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
                f'No main directory found while attempting to install from {file_or_url}')
        maindir = maindir[0]

        for dirname in ('plugins', 'sources'):
            source = os.path.join(maindir, dirname)
            if not os.path.exists(source):
                continue

            if dirname == 'sources':
                target = storage.expand_path('/').rstrip('/')
            else:
                target = os.path.join(config.rootpath, dirname)

            copy_tree(source, target)

        shutil.rmtree(tmpdir)
