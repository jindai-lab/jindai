"""Local task queue"""

import threading
from queue import deque
import datetime
import uuid
import time
import traceback

from jindai.models import TaskDBO
from jindai.task import Task
from .taskqueue import TaskQueue
from .announcer import announcer


class Job:
    """Local task queue job"""

    def __init__(self, task_dbo: TaskDBO, run_by: str, worker: TaskQueue):
        self.task_dbo = task_dbo
        self.run_by = run_by
        self.queued_at = datetime.datetime.now()
        self.uuid = str(uuid.uuid4())
        self.task = None
        self.status = 'pending'
        self.exception = {}
        self._worker = worker

    @property
    def key(self):
        """Get task key"""
        return f'{self.task_dbo.name}@{self.uuid}'

    @property
    def result(self):
        """Get task result"""
        if self.exception:
            return self.exception
        if self.task is None:
            return
        if self.task.alive:
            return
        return self.task.returned

    @property
    def result_type(self):
        """
        Get type of result, should be one of the following:
            - exception
            - redirect
            - file:<ext>
            - dict
            - null
            - list and other native python types
        """
        if self.status != 'stopped':
            return ''

        obj = self.result
        if isinstance(obj, dict):
            if '__exception__' in obj:
                return 'exception'
            if '__redirect__' in obj:
                return 'redirect'
            if '__file_ext__' in obj:
                return 'file:' + obj['__file_ext__']
            return 'dict'
        elif obj is None:
            return 'null'

        return type(obj).__name__

    def as_dict(self):
        """Return a dict representing current job"""

        return {
            'run_by': self.run_by,
            'name': self.task_dbo.name,
            'key': self.key,
            'status': self.status,
            'result_type': self.result_type,
            'queued_at': self.queued_at.strftime('%Y-%m-%d %H:%M:%S')
        }

    def run_task(self):
        """Run task"""

        def _callback(_):
            self.status = 'stopped'
            announcer.announce("updated")

        self.task = Task.from_dbo(
            self.task_dbo, logger=announcer.logger(self.key))
        self.status = 'running'
        self.task.run(_callback)

    def stop(self):
        """Stop current job"""
        if self.task:
            self.task.stop()

    @property
    def worker(self):
        """Return readonly current task queue"""
        return self._worker


class TaskLocalQueue(TaskQueue):

    def __init__(self, n=3):
        super().__init__(n)
        self._queue = deque()
        self._jobs = deque()
        self.running = False
        self._working_thread = None

    @property
    def jobs(self) -> list:
        return list(self._jobs)

    def enqueue(self, task_dbo, run_by):
        job = Job(task_dbo, run_by, self)
        self._queue.append(job)
        self._jobs.append(job)

        if not self.running:
            self.start()

        return job.key

    def _working(self):
        """Handling the queue"""

        while self.running:
            if self._queue and self.running_num < self.parallel_n:  # can run new task
                job = self._queue.popleft()

                try:
                    job.run_task()
                except Exception as ex:
                    job.status = 'stopped'
                    job.exception = {
                        '__exception__': f'Error while initializing task: {ex.__class__.__name__}: {ex}',
                        '__tracestack__': traceback.format_tb(ex.__traceback__)
                    }

                announcer.announce("updated")

            elif not self._queue and self.running_num == 0:  # all tasks done
                self.running = False

            time.sleep(0.5)

    def start(self):
        """Start handling tasks"""
        self.running = True
        self._working_thread = threading.Thread(target=self._working)
        self._working_thread.start()

    def stop(self):
        """Stop handling"""
        self.running = False

    def remove(self, key: str):
        """Remove/stop specified task"""

        job = self.get(key)

        if job:
            job.stop()
            if job in self._queue:
                self._queue.remove(job)
            self._jobs.remove(job)

            announcer.announce("updated")

        return job is not None
