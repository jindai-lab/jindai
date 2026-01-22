"""Task processing module"""

import ctypes
import datetime
import time
import uuid
import traceback
from collections import deque
from queue import PriorityQueue
from threading import Lock, Thread
from typing import Callable
from tqdm import tqdm

from .models import Paragraph, db_session
from .pipeline import Pipeline


class WorkersPool:
    
    def __init__(self, workers: int, interval: float = 0.1) -> None:
        self.workers = workers
        self._threads = {}
        self._lock = Lock()
        self._read_lock = Lock()
        self._interval = interval
        
    def count(self):
        return len(self._threads)
        
    def submit(self, func, *args, **kwargs):
        tid = uuid.uuid4()
        
        def _func():
            func(*args, **kwargs)
            self._threads.pop(tid)
        
        while self.count() >= self.workers:
            time.sleep(self._interval)
            
        with self._lock:
            thr = Thread(target=_func)
            self._threads[tid] = thr
        
        thr.start()
            
    def stop(self):
        
        def _terminate_thread(thread, exc_type = SystemExit):
            if not thread.is_alive():
                return

            exc = ctypes.py_object(exc_type)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)

            if res == 0:
                raise ValueError("nonexistent thread id")
            elif res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
                raise SystemError("PyThreadState_SetAsyncExc failed")
            
        for thr in list(self._threads.values()):
            _terminate_thread(thr)
            
    def debug_print(self):
        print('running threads:', self.count())
        print(' ', '\n  '.join(map(str, self._threads.keys())))
        

class Task:
    """Task object"""

    def __init__(self, params: dict, stages, concurrent=3, log: Callable = None,
                 resume_next: bool = False, verbose: bool = False, use_tqdm: bool = True) -> None:
        """Initialize the task object

        :param params: Parameters, used as the first paragraph/input to the pipeline
        :type params: dict
        :param stages: Pipeline stages
        :type stages: list
        :param concurrent: thread pool size, defaults to 3
        :type concurrent: int, optional
        :param log: the method to be called for logging, defaults to print
        :type log: Callable, optional
        :param resume_next: continue execution when encountering errors, defaults to False
        :type resume_next: bool, optional
        :param verbose: logging debug info, defaults to False
        :type verbose: bool, optional
        :param use_tqdm: use tqdm progress bar if applicable, defaults to True
        :type use_tqdm: bool, optional
        """

        self.alive = True
        self.returned = None
        self.resume_next = resume_next
        self.concurrent = concurrent

        self.log = log or self.print
        self.logs = deque()
        self.verbose = verbose

        self.pipeline = Pipeline(stages, self.log, verbose)
        self.params = params

        self._pbar = tqdm()
        self._queue = PriorityQueue()
        
        self._workers = None
         
    def _thread_execute(self, priority, fc):
        input_paragraph, stage = fc

        self._pbar.update(1)
        if self.verbose:
            self.log(type(stage).__name__, getattr(input_paragraph, 'id', '%x' % id(input_paragraph)))
        if stage is None:
            return None

        try:
            priority -= 1
            counter = 0
            for fc in stage.flow(input_paragraph):
                if fc[1] is None:
                    continue
                self._queue.put((priority, (id(fc[0]), id(fc[1])), fc))
                counter += 1
                if not self.alive:
                    break
            self._pbar.n = (self._pbar.n or 0) + 1
        except Exception as ex:
            self.log_exception('Error while executing', ex)
            if not self.resume_next:
                self.alive = False
                
    def print(self, *args):
        self.logs.append(args)

    def execute(self):
        """Execute the task
        :return: Summarized result, or exception in execution
        :rtype: dict
        """
        
        self._queue = PriorityQueue()
        self._workers = WorkersPool(self.concurrent)
        self._pbar.reset()
        
        self.pipeline.gctx = {}
        
        try:
            if self.pipeline.stages:
                self._queue.put((0, 0, (Paragraph.from_dict(self.params),
                                  self.pipeline.stages[0])))
                self._pbar.n += 1

                while self.alive:
                    while self.logs:
                        log = self.logs.popleft()
                        print(*log)
                        
                    if self._queue.empty():
                        if self._workers.count() > 0:
                            time.sleep(0.1)
                        else:
                            break
                    else:
                        if self._workers.count() >= self.concurrent:
                            time.sleep(0.1)
                        else:
                            priority, _, job = self._queue.get()
                            self._workers.submit(self._thread_execute, priority, job)
            
            if self.alive:
                return self.pipeline.summarize()
        except KeyboardInterrupt:
            self.alive = False
        except Exception as ex:
            self.alive = False
            self.log_exception('Error while executing task', ex)
            return {
                '__exception__': str(ex),
                '__tracestack__': traceback.format_tb(ex.__traceback__)
            }
        finally:
            for log in self.logs:
                print(*log)
            self.logs.clear()

        return None

    def log_exception(self, info, exc):
        self.log(info, type(exc).__name__, exc)
        self.log('\n'.join(traceback.format_tb(exc.__traceback__)))
        
    def stop(self):
        """Stop task"""
        self.alive = False
        if self._workers: 
            self._workers.stop()

    @staticmethod
    def from_dbo(dbo, **kwargs):
        """Get task from TaskDBO

        :param dbo: TaskDBO
        :type dbo: TaskDBO
        :return: task object according to DBO
        :rtype: Task
        """
        
        if dbo.pipeline:
            dbo.last_run = datetime.datetime.now()
            db_session.commit()
            
            return Task(params={},
                        stages=dbo.pipeline,
                        concurrent=dbo.concurrent,
                        resume_next=dbo.resume_next,
                        **kwargs)
        else:
            return Task({}, [])
