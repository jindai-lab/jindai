"""任务队列"""
import datetime
import os
import threading
import time
import traceback
import json
from io import BytesIO
import uuid
from queue import deque, Queue, Full

from flask import Response, jsonify, request, send_file, stream_with_context
from PyMongoWrapper import F
from jindai import Plugin
from jindai.helpers import logined, rest
from jindai.models import TaskDBO
from jindai.task import Task


class MessageAnnouncer:
    """Message announcer"""

    def __init__(self):
        self.listeners = []

    def listen(self):
        """listen to the announcer"""
        message_queue = Queue(maxsize=5)
        self.listeners.append(message_queue)
        return message_queue

    def announce(self, msg):
        """Announce message"""
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except Full:
                del self.listeners[i]

    def log(self, *args):
        """Announce log"""
        self.announce(' '.join([str(_) for _ in args]))

    def logger(self, prefix):
        """Get prefixed logger"""
        return lambda *args: self.log(prefix, '|', *args)


announcer = MessageAnnouncer()


class Job:

    def __init__(self, task_dbo, run_by):
        self.task_dbo = task_dbo
        self.run_by = run_by
        self.queued_at = datetime.datetime.now()
        self.uuid = str(uuid.uuid4())
        self.task = None
        self.status = 'pending'
        self.exception = {}

    @property
    def key(self):
        return f'{self.task_dbo.name}@{self.uuid}'

    @property
    def result(self):
        if self.exception:
            return self.exception
        if self.task is None:
            return
        if self.task.alive:
            return
        return self.task.returned

    @property
    def result_type(self):
        if self.status != 'stopped':
            return ''
        
        obj = self.result
        if isinstance(obj, dict):
            if '__exception__' in obj:
                return 'exception'
            if '__redirect__' in obj:
                return 'redirect'
            if '__file_ext__' in obj:
                return 'file'
            return 'dict'
        elif obj is None:
            return 'null'
        
        return type(obj).__name__

    def as_dict(self):
        return {
            'run_by': self.run_by,
            'name': self.task_dbo.name,
            'key': self.key,
            'status': self.status,
            'result_type': self.result_type,
            'queued_at': self.queued_at.strftime('%Y-%m-%d %H:%M:%S')
        }

    def run_task(self):

        def _callback(t):
            self.status = 'stopped'
            announcer.announce("updated")

        self.task = Task.from_dbo(self.task_dbo, logger=announcer.logger(self.key))
        self.status = 'running'
        self.task.run(_callback)

    def stop(self):
        if self.task:
            self.task.stop()


class TasksQueue(Plugin):
    """Handling queue
    """

    def __init__(self, pmanager, n=3):
        """
        Args:
            n (int, optional): maximal concurrent tasks
        """
        super().__init__(pmanager)
        app = self.pmanager.app
        self.pmanager.task_queue = self
        self.queue = deque()
        self.jobs = deque()
        self.running = False
        self.parallel_n = n
        self._working_thread = None

        @app.route('/api/queue/', methods=['PUT'])
        @rest(mapping={'id': 'task_id'})
        def enqueue_task(task_id=''):
            task_dbo = TaskDBO.first(F.id == task_id)
            assert task_dbo, 'No such task, or you do not have permission.'
            task_dbo.last_run = datetime.datetime.utcnow()
            task_dbo.save()
            
            job = Job(task_dbo, run_by=logined())
            self.queue.append(job)
            self.jobs.append(job)
            if not self.running:
                self.start()
            return job.key

        @app.route('/api/queue/events', methods=['GET'])
        @rest()
        def listen_events():

            def stream():
                messages = announcer.listen()
                while True:
                    msg = messages.get()
                    if msg == 'updated':
                        yield f'data: {json.dumps(self.filter_status())}\n\n'
                    else:
                        yield f'data: {json.dumps({"log": msg})}\n\n'

            resp = Response(stream_with_context(stream()), status=200,
                            mimetype="text/event-stream")
            resp.headers['Cache-Control'] = 'no-cache'
            resp.headers['X-Accel-Buffering'] = 'no'
            return resp

        @app.route('/api/queue/<path:task_id>', methods=['DELETE'])
        @rest()
        def dequeue_task(task_id):
            return self.remove(task_id)

        @app.route('/api/queue/<path:task_id>', methods=['GET'])
        @rest(cache=True)
        def fetch_task(task_id):
            job = self.get(task_id)
            if not job or job.status != 'stopped':
                return 'Not found or not finished', 404

            result = job.result
            result_type = job.result_type

            if result_type == 'list':
                offset, limit = int(request.args.get('offset', 0)), int(
                    request.args.get('limit', 0))
                if limit == 0:
                    return {
                        'results': result,
                        'total': len(result)
                    }

                return {
                    'results': result[offset:offset+limit],
                    'total': len(result)
                }

            elif result_type == 'file':
                buf = BytesIO(result['data'])
                buf.seek(0)
                return send_file(
                    buf, 'application/octstream',
                    download_name=os.path.basename(f"{task_id}.{result['__file_ext__']}"))

            return jsonify(result)

        @app.route('/api/queue/', methods=['GET'])
        @rest()
        def list_queue():
            return self.filter_status()

    def filter_status(self):
        """Generate request-specific status"""

        status = self.status
        if not logined('admin'):
            status = [_ for _ in status if _['run_by'] == logined()]
        return status

    def start(self):
        """Start handling tasks"""

        self.running = True
        self._working_thread = threading.Thread(target=self._working)
        self._working_thread.start()

    @property
    def status(self) -> dict:
        """Queue status"""
        return [_.as_dict() for _ in self.jobs]
        
    def _working(self):
        """Handling the queue"""

        while self.running:
            running_num = sum([1 for _ in self.jobs if _.status == 'running'])

            if self.queue and running_num < self.parallel_n:  # can run new task
                job = self.queue.popleft()
                
                try:
                    job.run_task()
                except Exception as ex:
                    job.status = 'stopped'
                    job.exception = {
                        '__exception__': f'Error while initializing task: {ex.__class__.__name__}: {ex}',
                        '__tracestack__': traceback.format_tb(ex.__traceback__) + [
                            self.pmanager.app.json_encoder().encode(task_dbo.as_dict())
                        ]
                    }
                
                announcer.announce("updated")

            elif not self.queue and running_num == 0:  # all tasks done
                self.running = False

            time.sleep(0.5)

    def stop(self):
        """Stop handling"""
        self.running = False

    def get(self, key: str):
        """Get job by key"""
        for j in self.jobs:
            if j.key == key:
                return j

    def remove(self, key: str):
        """Remove/stop specified task"""

        job = self.get(key)

        if job:
            job.stop()
            if job in self.queue:
                self.queue.remove(job)
            self.jobs.remove(job)

            announcer.announce("updated")
    
        return job is not None
