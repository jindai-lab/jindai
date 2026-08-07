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
        queue_buffer: int = 0,
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
            queue_buffer: Extra buffer multiplier for the bounded work queue.
                The queue capacity is ``concurrent * (2 + queue_buffer)``.
                A bounded queue provides backpressure so that a stage which
                fans out a large number of paragraphs (e.g. a data source)
                does not enqueue all of them at once and exhaust memory.
        """
        self.alive = True
        self.resume_next = resume_next
        self.concurrent = concurrent
        self.verbose = verbose
        self.params = params

        # Logging
        self.log_func = log or print

        # Core components (note: Pipeline methods also need to be async)
        self.pipeline = Pipeline(stages, self.log_func, verbose)

        # Bounded async priority queue.
        #
        # The bound is intentionally small relative to the number of workers:
        # a pipeline stage (particularly DataSourceStage.fetch) can yield far
        # more paragraphs than the workers can concurrently process. With an
        # unbounded queue every produced paragraph - and any heavy payload it
        # carries (text, images, ...) - would sit in memory waiting. A bounded
        # queue applies backpressure; when it is full, the producing worker
        # drains a pending item inline (see ``_enqueue``), which keeps memory
        # bounded without deadlocking the worker pool.
        self._queue_maxsize = max(self.concurrent * (2 + queue_buffer), 4)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self._queue_maxsize
        )
        self._pbar = tqdm(disable=not use_tqdm)
        self._worker_tasks = []

    async def _worker(self):
        """Consumer coroutine: pulls jobs from the bounded queue.

        The worker keeps draining the queue until **both** ``alive`` is False
        and the queue is empty. This distinction matters: ``alive=False`` only
        signals producers to stop fanning out, but items already in the queue
        must still be consumed so that every ``get()`` is paired with a
        ``task_done()`` and ``queue.join()`` can return. Bailing out the moment
        ``alive`` flips False would strand queued items and hang ``join()``.
        """
        while True:
            try:
                # Async get task; a timeout lets the loop re-check the
                # alive/empty state periodically so it can exit once all
                # producers have stopped and the queue is drained.
                priority, _, job = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                if not self.alive and self._queue.empty():
                    break
                continue

            # ``task_done`` MUST run even if execution raises, otherwise the
            # queue's unfinished-task counter drifts and ``queue.join()`` would
            # hang forever.
            try:
                await self._async_execute(priority, job)
            except Exception as e:
                self.log_exception("Worker Error", e)
            finally:
                self._queue.task_done()

    async def _enqueue(self, item: tuple) -> None:
        """Enqueue a job with backpressure.

        Uses ``put_nowait`` on the bounded queue. When the queue is full the
        current worker does **not** block (which could deadlock once every
        worker is blocked on a full queue while also being the only consumers).
        Instead it drains the highest-priority pending job inline and then
        retries. This effectively serializes oversized fan-outs (e.g. a data
        source yielding thousands of paragraphs) while keeping real
        concurrency bounded by ``self.concurrent`` and memory usage bounded by
        the queue capacity.

        Args:
            item: A ``(priority, tie_breaker, (paragraph, stage))`` tuple.
        """
        while self.alive:
            try:
                self._queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                # Make room by processing one pending job in this worker.
                # get_nowait on a PriorityQueue returns the highest-priority
                # item, which is consistent with normal worker ordering.
                try:
                    priority, _, job = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Extremely unlikely after a QueueFull; yield and retry.
                    await asyncio.sleep(0)
                    continue

                # Guard with try/finally so the removed item is always
                # accounted for; otherwise join() could hang on error.
                try:
                    await self._async_execute(priority, job)
                finally:
                    self._queue.task_done()

    async def _async_execute(self, priority: int, fc: tuple) -> None:
        """Async execution logic for a single job.

        Args:
            priority: Task priority.
            fc: Tuple of (paragraph, stage).
        """
        input_paragraph, stage = fc

        if self.verbose:
            self.log_func(type(stage).__name__, getattr(input_paragraph, 'id', '%x' % id(input_paragraph)))

        if stage is None:
            self._pbar.update(1)
            return

        try:
            # Priority handling
            new_priority = priority - 1

            # Use async for to drive async generator. Enqueuing goes through
            # the bounded ``_enqueue`` so a large fan-out is throttled instead
            # of flooding the queue.
            async for next_fc in stage.flow(input_paragraph):
                if next_fc[1] is None:
                    continue

                # Async enqueue: (priority, unique ID, data).
                # Use id() as sorting placeholder to prevent tuple comparison
                # errors when paragraphs compare equal.
                await self._enqueue(
                    (new_priority, id(next_fc[0]), next_fc)
                )

                if not self.alive:
                    break

            self._pbar.update(1)
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
                # Seed the queue with the initial paragraph/first stage.
                # Workers are started afterwards, so the seed always fits.
                await self._enqueue(
                    (0, 0, (Paragraph.from_dict(self.params), self.pipeline.stages[0]))
                )

                self._worker_tasks = [
                    asyncio.create_task(self._worker()) for _ in range(self.concurrent)
                ]

                await self._queue.join()

                self.alive = False
                for t in self._worker_tasks:
                    t.cancel()

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
            self._pbar.close()
        return None

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