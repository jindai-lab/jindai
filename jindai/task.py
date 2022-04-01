"""任务处理相关"""
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue

from .helpers import safe_import
from .models import Paragraph
from .pipeline import Pipeline


class Task:
    """任务对象"""

    def __init__(self, params: dict, stages, concurrent=3, logger: str = 'deque',
                 resume_next: bool = False, verbose: bool = False) -> None:
        """
        Args:
            params (dict|Paragraph): 初始参数
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

        self.logger = self.log_enqueue if logger == 'deque' else print
        self._logs = deque()
        self.verbose = verbose

        self.pipeline = Pipeline(stages, self.logger)
        self.params = params

        if self.logger == 'pbar':
            self.pbar = safe_import('tqdm').tqdm()
        else:
            class _FakeTqdm:
                """Stub for tqdm"""

                def update(self, _):
                    """Stub update"""

            self.pbar = _FakeTqdm()

    def execute(self):
        """执行任务"""
        tpe = ThreadPoolExecutor(max_workers=self.concurrent)
        self.pbar.n = 0
        queue = Queue()
        futures = []

        def _execute(input_paragraph, stage):
            self.pbar.update(1)
            if stage is None:
                return None

            try:
                for paragraph in stage.flow(input_paragraph):
                    queue.put(paragraph)
            except Exception as ex:
                self.logger('Error:', ex)
                self.logger(traceback.format_tb(ex.__traceback__))
                if not self.resume_next:
                    self.alive = False

        try:
            if self.pipeline.stages:
                queue.put((Paragraph(**self.params), self.pipeline.stages[0]))

                while self.alive:
                    if not queue.empty():
                        futures.append(tpe.submit(_execute, *queue.get()))
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
        
        return None

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
        """停止任务"""
        self.alive = False

    def log_enqueue(self, *args):
        """记录日志"""
        log_str = ' '.join(map(str, args))
        self._logs.append(log_str)

    def log_fetch(self):
        """获取日志"""
        while self._logs:
            yield self._logs.popleft() + '\n'

    @staticmethod
    def from_dbo(db_object, **kwargs):
        """从数据库对象构建"""
        if db_object.pipeline:
            return Task(params=db_object.pipeline[0][1],
                        stages=db_object.pipeline,
                        concurrent=db_object.concurrent,
                        resume_next=db_object.resume_next,
                        **kwargs)
        else:
            return Task({}, [])
