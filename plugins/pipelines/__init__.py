"""Register pipeline stages"""

import os

from jindai import Plugin
from jindai.helpers import get_context
from jindai.pipeline import PipelineStage


class PipelinesPlugin(Plugin):
    """Plugin for registering pipeline stages"""

    def __init__(self, pmanager, **config) -> None:
        super().__init__(pmanager, **config)
        ctx = get_context(os.path.relpath(os.path.dirname(__file__), os.path.realpath('.')), PipelineStage)
        self.register_pipelines(ctx)
