import os
from jindai import Plugin
from jindai.helpers import get_context
from jindai.pipeline import PipelineStage


class PipelinesPlugin(Plugin):
    
    def __init__(self, app, **config) -> None:
        super().__init__(app, **config)
        ctx = get_context(os.path.join('plugins', 'pipelines'), PipelineStage)
        self.register_pipelines(ctx)
