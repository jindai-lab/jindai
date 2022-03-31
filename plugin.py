from collections import defaultdict
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
        return {}
    
    def get_callbacks(self):
        return []
        
    def run_callback(self, name, *args, **kwargs):
        name = name.replace('-', '_') + '_callback'
        return getattr(self, name)(*args, **kwargs)
    
    def register_pipelines(self, pipeline_classes):
        if isinstance(pipeline_classes, dict):
            pipeline_classes = pipeline_classes.values()

        for c in pipeline_classes:
            if isinstance(c, type) and issubclass(c, PipelineStage) and c is not PipelineStage:
                Pipeline.pipeline_ctx[c.__name__] = c
    

class PluginManager:

    def __init__(self, app) -> None:
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

        all_plugins = get_context('plugins', Plugin)
        pls = []
        for pl in config.plugins:
            if pl == '*':
                pls += list(all_plugins.keys())
            elif pl.startswith('~'):
                if pl[1:] in pls:
                    pls.remove(pl[1:])
            else:
                pls.append(pl)

        for pl in pls:
            if isinstance(pl, tuple) and len(pl) == 2:
                pl, kwargs = pl
            else:
                kwargs = {}

            if isinstance(pl, str):
                pl = all_plugins.get(pl)
                
            if not pl:
                print('Plugin', pl, 'not found.')
                continue

            try:
                pl = pl(app, **kwargs)

                self.pages.update(**pl.get_pages())

                for name in pl.get_callbacks():
                    self.callbacks[name].append(pl)

                self.plugins.append(pl)
                print('Registered plugin:', type(pl).__name__)
            except Exception as ex:
                print('Error while registering plugin:', pl, ex)
                continue

    def __iter__(self):
        yield from self.plugins
