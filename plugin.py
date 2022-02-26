from collections import defaultdict
import os

from flask import Response
from models import get_context
from pipeline import Pipeline, PipelineStage
from helpers import rest
import config


class Plugin:
    
    def __init__(self, app, **config) -> None:
        self.config = config
        self.app = app
        
    def get_pages(self):
        return []
    
    def get_callbacks(self):
        return []
        
    def run_callback(self, name, *args, **kwargs):
        name = name.replace('-', '_') + '_callback'
        return getattr(self, name)(*args, **kwargs)
    
    def register_pipelines(self, pipeline_classes):
        if isinstance(pipeline_classes, dict):
            pipeline_classes = pipeline_classes.values()

        for c in pipeline_classes:
            if isinstance(c, type) and issubclass(c, PipelineStage):
                Pipeline.pipeline_ctx[c.__name__] = c
    

class PluginManager:

    def __init__(self, app) -> None:
        self.plugins = []
        self.pages = []
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

        import plugins as _plugins
        pls = getattr(config, 'plugins', ['*'])
        if pls == ['*']:
            pls = list(get_context('plugins', Plugin).values())

        for pl in pls:
            if isinstance(pl, tuple) and len(pl) == 2:
                pl, kwargs = pl
            else:
                kwargs = {}

            if isinstance(pl, str):
                if '.' in pl:
                    plpkg, plname = pl.rsplit('.', 1)
                    pkg = __import__('plugins.' + plpkg)
                    for seg in pl.split('.'):
                        pkg = getattr(pkg, seg)
                    pl = pkg
                else:
                    pl = getattr(_plugins, pl)

            try:
                pl = pl(app, **kwargs)

                self.pages += pl.get_pages()

                for name in pl.get_callbacks():
                    self.callbacks[name].append(pl)

                self.plugins.append(pl)
                print('Registered plugin:', type(pl).__name__)
            except Exception as ex:
                print('Error while registering plugin:', pl, ex)
                continue

