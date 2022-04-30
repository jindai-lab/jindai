"""Plugin for loading data sources"""

import os

from jindai import Plugin, PipelineStage
from jindai.helpers import get_context


class DatasourcePlugin(Plugin):
    """Datasource Plugin"""

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        ctx = get_context(os.path.join(
            'plugins', 'datasources'), PipelineStage)
        self.register_pipelines(ctx)
