"""插件"""
from collections import defaultdict
from flask import Response
from .helpers import rest
from .pipeline import Pipeline, PipelineStage
from .config import instance as config


class Plugin:
    """插件基类"""

    def __init__(self, app, **conf) -> None:
        self.config = conf
        self.app = app

    def get_filters(self):
        """获取特殊过滤器"""
        return {}

    def get_callbacks(self):
        """获取回调函数"""
        return []

    def run_callback(self, name, *args, **kwargs):
        """运行回调函数"""
        name = name.replace('-', '_') + '_callback'
        return getattr(self, name)(*args, **kwargs)

    def register_pipelines(self, pipeline_classes):
        """注册处理管道"""
        if isinstance(pipeline_classes, dict):
            pipeline_classes = pipeline_classes.values()

        for cls in pipeline_classes:
            if isinstance(cls, type) and issubclass(cls, PipelineStage) \
                and cls is not PipelineStage:
                Pipeline.ctx[cls.__name__] = cls


class PluginManager:
    """插件模拟器"""

    def __init__(self, plugin_ctx, app) -> None:
        self.plugins = []
        self.pages = {}
        self.callbacks = defaultdict(list)

        @app.route('/api/plugins/styles.css')
        def plugins_style():
            """Returns css from all enabled plugins

            Returns:
                Response: css document
            """
            css = '\n'.join([p.run_callback('css')
                            for p in self.callbacks['css']])
            return Response(css, mimetype='text/css')

        @app.route('/api/plugins/pages', methods=["GET", "POST"])
        @rest()
        def plugin_pages():
            """Returns names for special pages in every plugins
            """
            return self.pages

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
                plugin_name = plugin_ctx.get(plugin_name)

            if not plugin_name:
                print('Plugin', plugin_name, 'not found.')
                continue

            try:
                plugin_name = plugin_name(app, **params)

                self.pages.update(**plugin_name.get_filters())

                for name in plugin_name.get_callbacks():
                    self.callbacks[name].append(plugin_name)

                self.plugins.append(plugin_name)
                print('Registered plugin:', type(plugin_name).__name__)
            except Exception as ex:
                print('Error while registering plugin:', plugin_name, ex)
                continue

    def __iter__(self):
        """获取所有已加载的插件"""
        yield from self.plugins
