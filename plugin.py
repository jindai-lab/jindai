from pipeline import Pipeline, PipelineStage
import os


class Plugin:
    
    def __init__(self, app, **config) -> None:
        self.config = config
        self.app = app
        
    def get_pages(self):
        return []
    
    def handle_page(self, ds, post_args):
        return [], {}, {}

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
    
    @property
    def daemon(self):
        return os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
