"""Plugin for loading data sources into the pipeline.

This plugin provides various data source implementations that can be used
to import data from different formats and sources into the Jindai pipeline.
"""

import os

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.helpers import get_context


class DatasourcePlugin(Plugin):
    """Plugin for registering data source pipeline stages.
    
    This plugin discovers and registers all data source classes from the
    datasources module, making them available for use in pipeline configurations.
    """

    def __init__(self, pmanager, **config) -> None:
        """Initialize the DatasourcePlugin.
        
        Args:
            pmanager: The pipeline manager instance.
            **config: Additional configuration options passed to the parent Plugin.
        """
        super().__init__(pmanager, **config)
        ctx = get_context(os.path.join(
            'plugins', 'datasources'), PipelineStage)
        self.register_pipelines(ctx)
