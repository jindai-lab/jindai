from models import Paragraph, get_context
from datasource import DataSource
from pipeline import PipelineStage, Pipeline

class Task:

    datasource_ctx = get_context('datasources', DataSource)
    pipeline_ctx = get_context('pipelines', PipelineStage)

    def __init__(self, datasource, pipeline, concurrent=3, resume_next=False):
        name, args = datasource
        self.datasource = Task.datasource_ctx[name](**args)
        self.pipeline = Pipeline([Task.pipeline_ctx[name](**args) for name, args in pipeline], concurrent, resume_next)

    def execute(self):
        rs = self.datasource.fetch()
        for _ in self.pipeline.applyParagraphs(rs): pass
        return self.pipeline.summarize()
