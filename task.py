import threading
import traceback
from queue import Queue
from models import Paragraph, get_context
from pipeline import PipelineStage, Pipeline
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from helpers import safe_import


class Task:

    def __init__(self, params : dict, stages, concurrent=3, logger: str = 'deque', resume_next : bool = False,
                 verbose: bool = False, tqdm: bool = False) -> None:
        """
        Args:
            init_params (dict|Paragraph): 初始参数
            stages (Any): 各流程阶段的信息
            concurrent (int): 并发运行的数量
            logger (str): 记录日志的方式，deque 或 print
            verbose (bool): 进入和离开阶段时记录日志
            tqdm (bool): 使用 tqdm 记录处理的段落数量
        """
        
        self.alive = True
        self.returned = None
        self.resume_next = resume_next
        self.concurrent = concurrent
                
        self.logger = print if logger == 'print' else self.log_enqueue
        self._logs = deque()
        self.verbose = verbose
        
        self.pipeline = Pipeline(stages)
        self.pipeline.logger = self.logger
        self.params = params
        
        if tqdm:
            self.pbar = safe_import('tqdm').tqdm()
        else:
            class _FakeTqdm:
                def update(self, i):
                    pass
            self.pbar = _FakeTqdm()
                
    def execute(self):
        tpe = ThreadPoolExecutor(max_workers=self.concurrent)
        self.pbar.n = 0
        q = Queue()
        futures = []
        
        def _execute(p, stage):
            self.pbar.update(1)
            if stage is None:
                return
            
            try:
                for ps in stage.flow(p):
                    q.put(ps)
            except Exception as ex:
                self.logger('Error:', ex)
                self.logger(traceback.format_tb(ex.__traceback__))
                if not self.resume_next:
                    self.alive = False

        try:
            if self.pipeline.stages:
                q.put((Paragraph(**self.params), self.pipeline.stages[0]))
                                
                while self.alive:
                    if not q.empty():
                        futures.append(tpe.submit(_execute, *q.get()))
                    else:
                        if futures:
                            wait(futures)
                            futures = []
                        else:
                            break
            
            if self.alive:
                return self.pipeline.summarize()
            else:
                raise InterruptedError()
            
        except KeyboardInterrupt:
            self.alive = False
        except Exception as ex:
            self.alive = False
            return {'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)}            
        
    def run(self):
        """新建守护线程执行处理流程
        """
        def _run():
            try:
                self.returned = self.execute()
            except Exception as ex:
                self.logger('Error:', ex)
                self.logger(traceback.format_exc())
            self.alive = False

        self.alive = True
        thr = threading.Thread(target=_run)
        thr.start()
        return thr

    def stop(self):
        self.alive = False

    def log_enqueue(self, *args):
        s = ' '.join(map(str, args))
        self._logs.append(s)

    def log_fetch(self):
        while self._logs:
            yield self._logs.popleft() + '\n'

    @staticmethod
    def from_dbo(t, **args):
        if t.pipeline:
            return Task(params=t.pipeline[0][1], stages=t.pipeline, concurrent=t.concurrent, resume_next=t.resume_next, **args)
        else:
            return Task({}, [])


Pipeline.pipeline_ctx = get_context('pipelines', PipelineStage)
