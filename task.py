import threading
import traceback
from models import get_context
from datasource import DataSource
from pipeline import PipelineStage, Pipeline
from queue import deque

class Task:

    datasource_ctx = None
    
    def __init__(self, datasource, pipeline, concurrent=3, resume_next=False):
        if isinstance(datasource, dict) and len(datasource) == 1:
            (name, args), = datasource.items()
            if name.startswith('$'): name = name[1:]
        else:
            name, args = datasource
        self.datasource = Task.datasource_ctx[name](**args)
        self.pipeline = Pipeline(pipeline, concurrent, resume_next, self.log)
        self.queue = deque()
        self.alive = True
        self.returned = None

        self.datasource.logger = lambda *x: self.log(type(self.datasource).__name__, *x)

    def execute(self):
        try:
            rs = self.datasource.fetch()
            for _ in self.pipeline.applyParagraphs(rs): pass
            return self.pipeline.summarize()
        except Exception as ex:
            return {'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)}

    def log(self, *args):
        s = ' '.join(map(str, args))
        print(s)
        self.queue.append(s)
        
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
        while self.queue:
            yield self.queue.popleft() + '\n'

    @staticmethod
    def from_dbo(t):
        return Task(datasource=(t.datasource, t.datasource_config), pipeline=t.pipeline, concurrent=t.concurrent, resume_next=t.resume_next)


Pipeline.pipeline_ctx = get_context('pipelines', PipelineStage)
Task.datasource_ctx = get_context('datasources', DataSource)
