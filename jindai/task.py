"""Task processing module for Jindai application.

This module provides:
- Task: Main class for executing pipeline tasks
- Async execution with concurrent workers
- Priority queue for processing order
- Progress tracking with tqdm
- Error handling and resume capabilities
"""

import asyncio
import logging
import traceback
from collections import deque
from typing import Any, Callable

from tqdm import tqdm

from .models import Paragraph
from .pipeline import Pipeline


class Task:
    """Task executor for pipeline processing.

    Manages the execution of pipeline stages on paragraphs
    with support for concurrent processing, progress tracking,
    and error recovery.
    """

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
        """Initialize task.

        Args:
            params: Task parameters for initial paragraph.
            stages: List of pipeline stages.
            concurrent: Number of concurrent workers.
            log: Logging function.
            resume_next: Continue on error if True.
            verbose: Enable verbose logging.
            use_tqdm: Show progress bar.
        """
        self.alive = True
        self.resume_next = resume_next
        self.concurrent = concurrent
        self.verbose = verbose
        self.params = params

        # Logging
        self.log_func = log or self.default_log
        self.logs = deque()

        # Core components (note: Pipeline methods also need to be async)
        self.pipeline = Pipeline(stages, self.log_func, verbose)

        # Async priority queue
        self._queue = asyncio.PriorityQueue()
        self._pbar = tqdm(disable=not use_tqdm)
        self._worker_tasks = []

    def default_log(self, *args) -> None:
        """Non-blocking log recording."""
        self.logs.append(args)

    async def _worker(self):
        """Consumer coroutine: replaces thread pool worker."""
        while self.alive:
            try:
                # Async get task, automatically suspends when queue is empty
                priority, _, job = await self._queue.get()

                await self._async_execute(priority, job)

                # Must call task_done to配合后续的 queue.join()
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_exception("Worker Error", e)

    async def _async_execute(self, priority: int, fc: tuple) -> None:
        """Async execution logic, replaces _thread_execute.

        Args:
            priority: Task priority.
            fc: Tuple of (paragraph, stage).
        """
        input_paragraph, stage = fc

        self._pbar.update(1)
        if self.verbose:
            self.log_func(type(stage).__name__, getattr(input_paragraph, 'id', '%x' % id(input_paragraph)))

        if stage is None:
            return

        try:
            # Priority handling
            new_priority = priority - 1

            # Use async for to drive async generator
            async for next_fc in stage.flow(input_paragraph):
                if next_fc[1] is None:
                    continue

                # Async enqueue: (priority, unique ID, data)
                # Use id() as sorting placeholder to prevent tuple comparison errors
                await self._queue.put((new_priority, id(next_fc[0]), next_fc))

                if not self.alive:
                    break

            self._pbar.n = (self._pbar.n or 0) + 1
        except Exception as ex:
            self.log_exception('Error while executing', ex)
            if not self.resume_next:
                self.alive = False

    def execute(self):
        """Synchronous execution entry point.

        Returns:
            Task result.
        """
        return asyncio.run(self.execute_async())

    async def execute_async(self) -> dict[str, Any] | None:
        """Main entry point: replaces execute.

        Returns:
            Task result or None.
        """
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
                for t in self._worker_tasks:
                    t.cancel()
                log_monitor.cancel()

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
        """Async log monitoring and printing."""
        while self.alive:
            self._flush_logs()
            await asyncio.sleep(0.1)

    def _flush_logs(self):
        """Batch flush logs from deque."""
        while self.logs:
            log = self.logs.popleft()
            log = ' '.join(map(str, log))
            logging.info(log)

    def log_exception(self, info: str, exc: Exception) -> None:
        """Log exception with traceback.

        Args:
            info: Error information.
            exc: Exception instance.
        """
        self.log_func(info, type(exc).__name__, exc)
        self.log_func("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def stop(self) -> None:
        """Stop task execution."""
        self.alive = False
        for t in self._worker_tasks:
            t.cancel()

    @staticmethod
    def from_dbo(dbo, **kwargs) -> "Task":
        """Create task from TaskDBO.

        Args:
            dbo: TaskDBO instance.
            **kwargs: Additional keyword arguments.

        Returns:
            Task instance.
        """
        if dbo.pipeline:
            return Task(
                params={},
                stages=dbo.pipeline,
                concurrent=dbo.concurrent,
                resume_next=dbo.resume_next,
                **kwargs,
            )
        return Task({}, [])
