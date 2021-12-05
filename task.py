import threading
import traceback
from models import get_context
from datasource import DataSource
from pipeline import PipelineStage, Pipeline
from queue import Queue

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
        self.queue = Queue()
        self.alive = True
        self.returned = None

        self.datasource.logger = self.log
        self.pipeline.logger = self.log

    def execute(self):
        rs = self.datasource.fetch()
        for _ in self.pipeline.applyParagraphs(rs): pass
        return self.pipeline.summarize()

    def log(self, *args):
        s = ' '.join(map(str, args))
        print(s)
        self.queue.put(s)
        
    def run(self):
        def _run():
            try:
                self.returned = self.execute()
            except Exception as ex:
                self.log('Error:', ex)
                self.log(traceback.format_exc())
            self.alive = False
        
        self.alive = True
        thr = threading.Thread(target=_run)
        thr.start()
        return thr
    
    def fetch_log(self):
        while not self.queue.empty():
            yield self.queue.get() + '\n'

    @staticmethod
    def from_dbo(t):
        return Task(datasource=(t.datasource, t.datasource_config), pipeline=t.pipeline, concurrent=t.concurrent, resume_next=t.resume_next)


Pipeline.pipeline_ctx = get_context('pipelines', PipelineStage)
Task.datasource_ctx = get_context('datasources', DataSource)
