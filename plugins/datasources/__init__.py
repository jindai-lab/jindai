"""数据源插件"""

import os

from jindai import Plugin, PipelineStage
from jindai.helpers import get_context


class DatasourcePlugin(Plugin):
    """数据源插件"""

    def __init__(self, app, **config):
        super().__init__(app, **config)
        ctx = get_context(os.path.join('plugins', 'datasources'), PipelineStage)
        self.register_pipelines(ctx)
