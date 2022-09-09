"""Task processing module"""

import os
import sys
import threading
import time
import traceback
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .helpers import safe_import
from .models import Paragraph
from .pipeline import Pipeline


class _TqdmProxy:
    """Proxy for tqdm"""

    def __init__(self):
        self.pbar = safe_import('tqdm').tqdm()
        self._lock = threading.Lock()

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


class Task:
    """Task object"""

    def __init__(self, params: dict, stages, concurrent=3, logger: Callable = print,
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
        :param verbose: use tqdm progress bar if applicable, defaults to True
        :type verbose: bool, optional
        """

        self.alive = True
        self.returned = None
        self.resume_next = resume_next
        self.concurrent = concurrent

        self.logger = logger
        self.verbose = verbose

        self.pipeline = Pipeline(stages, self.logger)
        self.params = params

        self.pbar = _TqdmProxy() if not verbose and use_tqdm and os.isatty(sys.stdout.fileno()) else _FakeTqdm()
        
        self.queue = None

    def _thread_execute(self, priority, job):
        input_paragraph, stage = job

        if self.verbose: self.logger(type(stage).__name__, id(input_paragraph))
        self.pbar.update(1)
        if self.verbose:
            self.logger(type(stage).__name__, '%x' % id(input_paragraph))
        if stage is None:
            return None

        try:
            priority -= 1
            counter = 0
            for job in stage.flow(input_paragraph):
                self.queue.put((priority, id(job[0]), job))
                counter += 1
                if not self.alive:
                    break
            self.pbar.inc_total(counter)
        except Exception as ex:
            self.logger('Error:', ex)
            self.logger('\n'.join(traceback.format_tb(ex.__traceback__)))
            if not self.resume_next:
                self.alive = False

    def execute(self):
        """Execute the task
        :return: Summarized result, or exception in execution
        :rtype: dict
        """
        self.queue = PriorityQueue()
        tpe = ThreadPoolExecutor(max_workers=self.concurrent)
        futures = {}
        self.pbar.reset()
        
        try:
            if self.pipeline.stages:
                self.queue.put((0, 0, (Paragraph(**self.params),
                                  self.pipeline.stages[0])))
                self.pbar.inc_total(1)

                while self.alive:
                    if self.queue.empty():
                        if futures:  # there are pending jobs
                            time.sleep(0.1)
                        else:
                            break  # exit working loop
                    else:
                        if len(futures) >= self.concurrent:
                            time.sleep(0.1)
                        else:
                            priority, _, job = self.queue.get()
                            future = tpe.submit(self._thread_execute, priority, job)
                            futures[id(future)] = future
                            future.add_done_callback(lambda h: futures.pop(id(h), ''))
            if self.alive:
                return self.pipeline.summarize()
        except KeyboardInterrupt:
            self.alive = False
        except Exception as ex:
            self.alive = False
            self.logger(type(ex).__name__, ex)
            traceback.print_exc()
            return {
                '__exception__': str(ex),
                '__tracestack__': traceback.format_tb(ex.__traceback__)
            }
        finally:
            for future in list(futures.values()):
                if not future.done():
                    future.cancel()

        return None

    def run(self):
        """Create a daemon thread to execute the task
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
        """Stop task"""
        self.alive = False

    @staticmethod
    def from_dbo(db_object, **kwargs):
        """Get task from TaskDBO

        :param db_object: TaskDBO
        :type db_object: TaskDBO
        :return: task object according to DBO
        :rtype: Task
        """
        if db_object.pipeline:
            return Task(params=db_object.pipeline[0][1],
                        stages=db_object.pipeline,
                        concurrent=db_object.concurrent,
                        resume_next=db_object.resume_next,
                        **kwargs)
        else:
            return Task({}, [])
