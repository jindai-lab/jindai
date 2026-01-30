"""Plugin for loading data sources"""

import os

from jindai import PipelineStage, Plugin
from jindai.helpers import get_context


class DatasourcePlugin(Plugin):
    """Datasource Plugin"""

    def __init__(self, pmanager, **config) -> None:
        super().__init__(pmanager, **config)
        ctx = get_context(os.path.join(
            'plugins', 'datasources'), PipelineStage)
        self.register_pipelines(ctx)
