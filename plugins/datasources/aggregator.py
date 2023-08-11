from jindai.pipeline import Pipeline, PipelineStage
from jindai.models import Paragraph


class AggregateDataSource(PipelineStage):
    """Aggregate results from multiple data sources
    @zhs 聚合不同数据源的数据
    """
    
    def __init__(self, pipeline):
        """
        Args:
            pipeline (pipeline): Pipeline of data sources
                @zhs 各数据源
        """
        self.sources = Pipeline(pipeline, self.logger)

    def resolve(self, para: Paragraph):
        for source in self.sources.stages:
            yield from source.resolve(para)
