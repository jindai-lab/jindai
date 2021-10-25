from models import Paragraph, get_context
from datasource import DataSource
from pipeline import PipelineStage, Pipeline

class Task:

    datasource_ctx = None
    
    def __init__(self, datasource, pipeline, concurrent=3, resume_next=False):
        if isinstance(datasource, dict) and len(datasource) == 1:
            (name, args), = datasource.items()
            if name.startswith('$'): name = name[1:]
        else:
            name, args = datasource
        self.datasource = Task.datasource_ctx[name](**args)
        self.pipeline = Pipeline(pipeline, concurrent, resume_next)

    def execute(self):
        rs = self.datasource.fetch()
        for _ in self.pipeline.applyParagraphs(rs): pass
        return self.pipeline.summarize()


Pipeline.pipeline_ctx = get_context('pipelines', PipelineStage)
Task.datasource_ctx = get_context('datasources', DataSource)
