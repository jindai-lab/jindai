"""Task processing module"""

from collections import deque
import os
import sys
from threading import Thread, Lock
import time
import traceback
from queue import PriorityQueue
from typing import Callable
import ctypes
import nanoid

from .helpers import safe_import
from .models import Paragraph
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
        tid = nanoid.generate()
        
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
        

class TqdmFactory:

    class _TqdmProxy:
        """Proxy for tqdm"""

        def __init__(self):
            self.pbar = safe_import('tqdm').tqdm()
            self._lock = Lock()

        def update(self, inc: int):
            """Update pbar value

            :param inc: inc value
            :type inc: int
            """
            self.pbar.update(inc)

        def reset(self):
            """Reset count and total"""
            self.pbar.n = 0
            self.pbar.total = None

        def inc_total(self, inc: int):
            """Thread-safe incresement for pbar.total

            :param inc: inc value
            :type inc: int
            """
            with self._lock:
                if self.pbar.total is None:
                    self.pbar.total = inc
                else:
                    self.pbar.total += inc

    class _FakeTqdm:
        def update(self, _):
            """Stub update"""

        def inc_total(self, _):
            """Stub inc_total"""

        def reset(self):
            """Stub reset"""
            
    @staticmethod
    def get_tqdm(fake=False):
        if not os.isatty(sys.stdout.fileno()):
            fake = True
            
        if fake:
            return TqdmFactory._FakeTqdm()
        
        return TqdmFactory._TqdmProxy()


class Task:
    """Task object"""

    def __init__(self, params: dict, stages, concurrent=3, logger: Callable = None,
                 resume_next: bool = False, verbose: bool = False, use_tqdm: bool = True) -> None:
        """Initialize the task object

        :param params: Parameters, used as the first paragraph/input to the pipeline
        :type params: dict
        :param stages: Pipeline stages
        :type stages: list
        :param concurrent: thread pool size, defaults to 3
        :type concurrent: int, optional
        :param logger: the method to be called for logging, defaults to print
        :type logger: Callable, optional
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

        self.logger = logger or self.print
        self.logs = deque()
        self.verbose = verbose

        self.pipeline = Pipeline(stages, self.logger)
        self.params = params

        self._pbar = TqdmFactory.get_tqdm(fake=verbose or not use_tqdm)
        self._queue = None
        
        self._workers = WorkersPool(1)
         
    def _thread_execute(self, priority, fc, gctx):
        input_paragraph, stage = fc

        self._pbar.update(1)
        if self.verbose:
            self.logger(type(stage).__name__, getattr(input_paragraph, 'id', '%x' % id(input_paragraph)))
        if stage is None:
            return None

        try:
            priority -= 1
            counter = 0
            for fc in stage.flow(input_paragraph, gctx):
                if fc[1] is None:
                    continue
                self._queue.put((priority, id(fc[0]), fc))
                counter += 1
                if not self.alive:
                    break
            self._pbar.inc_total(counter)
        except Exception as ex:
            self.logger('Error while executing', type(ex).__name__, ex)
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
        
        gctx = {}
        
        try:
            if self.pipeline.stages:
                self._queue.put((0, 0, (Paragraph(**self.params),
                                  self.pipeline.stages[0])))
                self._pbar.inc_total(1)

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
                            self._workers.submit(self._thread_execute, priority, job, gctx)
            
            if self.alive:
                return self.pipeline.summarize()
        except KeyboardInterrupt:
            self.alive = False
        except Exception as ex:
            self.alive = False
            self.log_exception('Error while executing task', ex)
            traceback.print_exc()
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
        self.logger(info, type(exc).__name__, exc)
        self.logger('\n'.join(traceback.format_tb(exc.__traceback__)))
        
    def run(self, callback = None):
        """Create a daemon thread to execute the task
        """
        def _run():
            try:
                self.returned = self.execute()
            except Exception as ex:
                self.logger('Error while running task', type(ex).__name__, ex)
            self.alive = False
            if callback:
                callback(self)

        self.alive = True
        thr = Thread(target=_run)
        thr.start()
        return thr

    def stop(self):
        """Stop task"""
        self.alive = False
        self._workers.stop()

    @staticmethod
    def from_dbo(db_object, **kwargs):
        """Get task from TaskDBO

        :param db_object: TaskDBO
        :type db_object: TaskDBO
        :return: task object according to DBO
        :rtype: Task
        """
        if db_object.pipeline:
            return Task(params={},
                        stages=db_object.pipeline,
                        concurrent=db_object.concurrent,
                        resume_next=db_object.resume_next,
                        **kwargs)
        else:
            return Task({}, [])
