"""Task queue plugin"""

import itertools

from jindai import Plugin
from .taskqueue import TaskQueue
from .localqueue import TaskLocalQueue
from .remotequeue import TaskRemoteQueue


class MultiWorkerTaskQueue(TaskQueue):
    """Multiple workers task queue
    """

    def __init__(self, n=3, workers=None) -> None:
        super().__init__(n)
        if workers:
            self._workers = [TaskRemoteQueue(w) for w in workers]
        else:
            self._workers = [TaskLocalQueue(n)]

    @property
    def jobs(self) -> list:
        """Queue status"""
        return list(itertools.chain(*[w.jobs for w in self._workers]))

    @property
    def parallel_n(self) -> int:
        return sum([w.parallel_n for w in self._workers])

    @property
    def capabilities(self) -> list:
        caps = set()
        for worker in self._workers:
            for cap in worker.capabilities:
                caps.add(cap)
        return list(caps)

    def _select_worker(self, task_dbo):

        def _capabilities(pipeline):
            for stage in pipeline:
                if isinstance(stage, dict):
                    stage, = stage.items()

                if isinstance(stage, tuple):
                    name, params = stage
                    yield name.strip('$')
                    for subpip in ('iffalse', 'iftrue', 'pipeline'):
                        if subpip in params:
                            yield from _capabilities(params[subpip])

        def _has_capabilities(capabilities):
            capabilities = set(capabilities)
            return lambda w: set(w.capabilities).issuperset(capabilities)

        return sorted(filter(
            _has_capabilities(
                _capabilities(task_dbo.pipeline)),
            self._workers), key=lambda w: w.parallel_n - w.running_num).pop()

    def enqueue(self, task_dbo, run_by):
        worker = self._select_worker(task_dbo)
        return worker.enqueue(task_dbo, run_by)

    def remove(self, key: str):
        """Remove/stop specified task"""
        job = self.get(key)
        if job:
            return job.worker.remove(job.key)


class TaskQueuePlugin(Plugin):
    """Task queue plugin
    """

    def __init__(self, pmanager, n=3, workers=None) -> None:
        super().__init__(pmanager)
        self.handler = MultiWorkerTaskQueue(n, workers)
        self.handler.register_api(self.pmanager.app)
        pmanager.task_queue = self.handler
