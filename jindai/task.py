"""Task processing module"""

import ctypes
import datetime
import time
import traceback
import uuid
from collections import deque
from queue import PriorityQueue
from threading import Lock, Thread
from typing import Callable

from tqdm import tqdm

from .models import Paragraph
from .pipeline import Pipeline


class WorkersPool:
    def __init__(self, workers: int, interval: float = 0.1) -> None:
        """Initialize workers pool

        :param workers: Number of worker threads
        :type workers: int
        :param interval: Sleep interval when pool is full, defaults to 0.1
        :type interval: float, optional
        """
        self.workers = workers
        self._threads = {}
        self._lock = Lock()
        self._read_lock = Lock()
        self._interval = interval

    def count(self) -> int:
        """Get current number of running threads

        :return: Number of active threads
        :rtype: int
        """
        return len(self._threads)

    def submit(self, func, *args, **kwargs) -> None:
            """Submit a function to be executed by a worker thread

            :param func: Function to execute
            :type func: Callable
            :param args: Positional arguments for the function
            :param kwargs: Keyword arguments for the function
            """
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

    def stop(self) -> None:
            """Stop all worker threads forcefully"""
            
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

            def debug_print(self) -> None:
                """Print debug information about running threads"""
                print("running threads:", self.count())
                print(" ", "\n  ".join(map(str, self._threads.keys())))

import asyncio
import datetime
import traceback
from collections import deque
from typing import Callable, Any
from tqdm import tqdm

class Task:
    """异步任务对象 - 针对 FastAPI 优化"""

    def __init__(
        self,
        params: dict,
        stages: list,
        concurrent: int = 3,
        log: Callable = None,
        resume_next: bool = False,
        verbose: bool = False,
        use_tqdm: bool = True,
    ) -> None:
        self.alive = True
        self.resume_next = resume_next
        self.concurrent = concurrent
        self.verbose = verbose
        self.params = params

        # 日志处理
        self.log_func = log or self.default_log
        self.logs = deque()

        # 核心组件 (注意：Pipeline 内部的方法也需要是 async 的)
        self.pipeline = Pipeline(stages, self.log_func, verbose)
        
        # 异步优先级队列
        self._queue = asyncio.PriorityQueue()
        self._pbar = tqdm(disable=not use_tqdm)
        self._worker_tasks = []

    def default_log(self, *args) -> None:
        """非阻塞地记录日志"""
        self.logs.append(args)

    async def _worker(self):
        """消费者协程：替代原本的线程池 worker"""
        while self.alive:
            try:
                # 异步获取任务，队列为空时会自动挂起不占 CPU
                priority, _, job = await self._queue.get()
                
                await self._async_execute(priority, job)
                
                # 必须调用 task_done 才能配合后续的 queue.join()
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_exception("Worker Error", e)

    async def _async_execute(self, priority: int, fc: tuple) -> None:
        """异步执行逻辑，替代原 _thread_execute"""
        input_paragraph, stage = fc

        self._pbar.update(1)
        if self.verbose:
            self.log_func(type(stage).__name__, getattr(input_paragraph, 'id', '%x' % id(input_paragraph)))
        
        if stage is None:
            return

        try:
            # 优先级处理
            new_priority = priority - 1
            
            # 使用 async for 驱动异步生成器 
            async for next_fc in stage.flow(input_paragraph):
                if next_fc[1] is None:
                    continue
                
                # 异步入队：(优先级, 唯一ID, 数据)
                # 使用 id() 作为排序占位符防止元组比较结果数据导致报错
                await self._queue.put((new_priority, id(next_fc[0]), next_fc))
                
                if not self.alive:
                    break
                    
            self._pbar.n = (self._pbar.n or 0) + 1
        except Exception as ex:
            self.log_exception('Error while executing', ex)
            if not self.resume_next:
                self.alive = False
                
    def execute(self):
        return asyncio.run(self.execute_async())

    async def execute_async(self) -> dict[str, Any] | None:
        """主入口：替代原有的 execute"""
        self.pipeline.gctx = {}
        self._pbar.reset()

        try:
            if self.pipeline.stages:
                await self._queue.put(
                    (0, 0, (Paragraph.from_dict(self.params), self.pipeline.stages[0]))
                )
                self._pbar.n += 1

                self._worker_tasks = [
                    asyncio.create_task(self._worker()) for _ in range(self.concurrent)
                ]

                log_monitor = asyncio.create_task(self._log_monitor())
                
                await self._queue.join()
                
                self.alive = False
                for t in self._worker_tasks: t.cancel()
                log_monitor.cancel()

            if self.alive:
                return await self.pipeline.summarize()

        except asyncio.CancelledError:
            self.alive = False
        except Exception as ex:
            self.alive = False
            self.log_exception("Critical task error", ex)
            return {
                "__exception__": str(ex),
                "__tracestack__": "".join(traceback.format_exception(type(ex), ex, ex.__traceback__)),
            }
        finally:
            self._flush_logs()
            self._pbar.close()
        return None

    async def _log_monitor(self):
        """异步监控并打印日志"""
        while self.alive:
            self._flush_logs()
            await asyncio.sleep(0.1)

    def _flush_logs(self):
        """将 deque 中的日志批量吐出"""
        while self.logs:
            log = self.logs.popleft()
            print(*log)

    def log_exception(self, info: str, exc: Exception) -> None:
        self.log_func(info, type(exc).__name__, exc)
        self.log_func("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def stop(self) -> None:
        self.alive = False
        for t in self._worker_tasks:
            t.cancel()

    @staticmethod
    def from_dbo(dbo, **kwargs) -> "Task":
        if dbo.pipeline:
            dbo.last_run = datetime.datetime.now()
            # 假设你已经迁移到了异步数据库驱动
            # await async_session.merge(dbo)
            
            return Task(
                params={},
                stages=dbo.pipeline,
                concurrent=dbo.concurrent,
                resume_next=dbo.resume_next,
                **kwargs,
            )
        return Task({}, [])