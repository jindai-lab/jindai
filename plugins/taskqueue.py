"""任务队列"""
import datetime
import itertools
import os
import threading
import time
import traceback
import json
from io import BytesIO
import uuid
from wsgiref import headers
import requests
from queue import deque, Queue, Full
from flask import Response, jsonify, request, send_file, stream_with_context

from PyMongoWrapper import F
from jindai import Plugin
from jindai.helpers import logined, rest, safe_import
from jindai.models import TaskDBO
from jindai.pipeline import Pipeline
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

    def __init__(self, task_dbo, run_by, worker):
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
                return 'file:' + obj['__file_ext__']
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

        self.task = Task.from_dbo(
            self.task_dbo, logger=announcer.logger(self.key))
        self.status = 'running'
        self.task.run(_callback)

    def stop(self):
        if self.task:
            self.task.stop()

    @property
    def worker(self):
        return self._worker


class TaskQueue:
    """Handling queue
    """

    def __init__(self, n=3):
        """
        Args:
            n (int, optional): maximal concurrent tasks
        """
        self._parallel_n = n
        self._abilities = list(Pipeline.ctx)

    @property
    def parallel_n(self) -> int:
        """Get maximal concurrent tasks number

        Returns:
            int
        """
        return self._parallel_n
    
    @property
    def abilities(self) -> list:
        return self._abilities

    def register_api(self, app):
        """Register api entrypoint to flask app

        Args:
            app (Flask): flask app
        """

        @app.route('/api/queue/', methods=['PUT'])
        @rest(mapping={'id': 'task_id'})
        def enqueue_task(task_id='', task=None, run_by=''):
            """Enqueue task api

            Args:
                task_id (str, optional): Task ID. Defaults to ''.
                task (dict, optional): Task dictionary. Defaults to None.

            Returns:
                str: Queued job key
            """
            if task is None:
                task_dbo = TaskDBO.first(F.id == task_id)
            else:
                task_dbo = TaskDBO().fill_dict(task)

            assert task_dbo, 'No such task, or you do not have permission.'

            task_dbo.last_run = datetime.datetime.utcnow()
            task_dbo.save()
            
            if not run_by or not logined('admin'): run_by = logined()
            
            return self.enqueue(task_dbo, run_by=run_by)

        @app.route('/api/queue/config', methods=['GET'])
        @rest()
        def queue_config():
            """Get config of current worker

            Returns:
                dict: a dict representing current config
            """
            return {
                'parallel_n': self.parallel_n,
                'abilities': self.abilities,
            }

        @app.route('/api/queue/events', methods=['GET'])
        @rest()
        def listen_events():
            """Provide event stream
            """

            def stream():
                messages = announcer.listen()
                while True:
                    msg = messages.get()
                    if msg == 'updated':
                        yield f'data: {json.dumps(filter_status())}\n\n'
                    else:
                        yield f'data: {json.dumps({"log": msg})}\n\n'

            resp = Response(stream_with_context(stream()), status=200,
                            mimetype="text/event-stream")
            resp.headers['Cache-Control'] = 'no-cache'
            resp.headers['X-Accel-Buffering'] = 'no'
            return resp

        @app.route('/api/queue/<path:key>', methods=['DELETE'])
        @rest()
        def dequeue_qjob(key):
            """Remove task from queue or stop the job

            Args:
                task_id (str): Task key

            Returns:
                bool: true if successful
            """
            return self.remove(key)

        @app.route('/api/queue/<path:key>', methods=['GET'])
        @rest(cache=True)
        def fetch_task(key):
            """Get queued job result

            Args:
                key (str): job key

            Returns:
                Response: JSON data or octstream
            """
            job = self.get(key)
            if not job or job.status != 'stopped':
                return 'Not found or not finished', 404

            result = job.result
            result_type = job.result_type

            if result_type == 'list':
                offset, limit = int(request.args.get('offset', 0)), int(
                    request.args.get('limit', 0))
                if limit == 0:
                    limit = len(result)

                return {
                    'results': result[offset:offset+limit],
                    'total': len(result)
                }

            elif result_type.startswith('file:'):
                ext = result_type.split(':')[1]
                buf = BytesIO(result['data'])
                buf.seek(0)
                return send_file(
                    buf, 'application/octstream',
                    download_name=os.path.basename(f"{key}.{ext}"))

            return jsonify(result)

        @app.route('/api/queue/', methods=['GET'])
        @rest()
        def list_queue():
            """Get queue content

            Returns:
                list: queue job statuses
            """
            return filter_status()

        def filter_status():
            """Generate request-specific status"""
            status = [_.as_dict() for _ in self.jobs]
            if not logined('admin'):
                status = [_ for _ in status if _['run_by'] == logined()]
            return status

    @property
    def jobs(self) -> list:
        """Queue jobs, a list of Job-like objects"""
        return []

    @property
    def running_num(self) -> int:
        """Get number of running jobs

        Returns:
            int: number of running jobs
        """
        return sum([1 for _ in self.jobs if _.status == 'running'])

    def enqueue(self, task_dbo: TaskDBO, run_by: str):
        """Enqueue a task

        Args:
            task_dbo (TaskDBO): task db object
            run_by (str): name

        Raises:
            NotImplemented: if not implemented
        """
        raise NotImplemented()

    def get(self, key: str):
        """Get job by key"""
        for j in self.jobs:
            if j.key == key:
                return j


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
                        '__tracestack__': traceback.format_tb(ex.__traceback__) + [
                            self.pmanager.app.json_encoder().encode(task_dbo.as_dict())
                        ]
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


class TaskRemoteQueue(TaskQueue):

    def __init__(self, entrypoint) -> None:
        self._base = entrypoint.rstrip('/') + '/'
        self._config = {}
        self._queue = []
        self.alive = True
        self.listen_events()

    @property
    def jobs(self):
        
        class _JobProxy(dict):
            
            def __init__(self, __dict, remote):
                super().__init__(__dict)
                self.remote = remote
            
            def __getattribute__(self, __name: str):
                if __name in self:
                    return self[__name]
                return super().__getattribute__(__name)

            def as_dict(self):
                return dict(self, worker=None, remote=None)
            
            @property
            def result(self):
                buf = self.remote.call(f'queue/{self.key}', 'get', raw=True)
                if self.result_type.startswith('file:'):
                    return {
                        'data': buf,
                        '__file_ext__': self.result_type.split(':')[1]
                    }
                else:
                    buf = json.loads(buf)['result']
                    if self.result_type == 'list':
                        return buf['results']
                    else:
                        return buf
                
        return [_JobProxy(_, self) for _ in self._queue]

    def update_config(self):
        self._config = self.call('queue/config') or {}
        self._parallel_n = self._config.get('parallel_n', 0)
        
        ab = self.call('help/pipelines') or {}
        self._abilities = list(itertools.chain(*[v.keys() for v in ab.values()]))

    def call(self, api_name, method='get', data=None, raw=False):
        from PyMongoWrapper.dbo import create_dbo_json_encoder
        method = method.upper()
        try:
            headers = {}
            if isinstance(data, dict):
                data = json.dumps(cls=create_dbo_json_encoder(json.JSONEncoder), obj=data)
                headers['content-type'] = 'application/json'
            resp = requests.request(method, self._base + api_name, data=data, headers=headers)
            if not raw:
                resp = resp.json().get('result')
            else:
                resp = resp.content
            return resp
        except Exception as ex:
            print('Error while calling remote queue', self._base, ':', ex)

    def stop(self):
        self.alive = False

    def remove(self, key):
        return self.call(f'queue/{key}', 'delete')

    def enqueue(self, task_dbo: TaskDBO, run_by: str):
        return self.call('queue/', 'put', {'task': task_dbo.as_dict(), 'run_by': run_by})

    def listen_events(self):
        sse = safe_import('sseclient')

        def _listen():
            while self.alive:
                try:
                    self.update_config()
                    msgs = sse.SSEClient(self._base + 'queue/events')
                    
                    for msg in msgs:
                        print(msg)
                        self._queue.clear()
                        for q in json.loads(msg.data):
                            q['worker'] = self
                            self._queue.append(q)
                        announcer.announce('updated')

                    if not self.alive:
                        break
                    time.sleep(0.1)
                except Exception as ex:
                    print(ex)
                    return

        threading.Thread(target=_listen).start()


class MultiWorkerTaskQueue(TaskQueue):

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
    def abilities(self) -> list:
        ab = set()
        for w in self._workers:
            for e in w.abilities:
                ab.add(e)
        return list(ab)

    def _select_worker(self, task_dbo):
        
        def _abilities(pipeline):
            for p in pipeline:
                if isinstance(p, dict):
                    p, = p.items()
                
                if isinstance(p, tuple):
                    k, v = p
                    yield k.strip('$')
                    for subpip in ('iffalse', 'iftrue', 'pipeline'):
                        if subpip in v:
                            yield from _abilities(v[subpip])
                    
        def _has_abilities(abilities):
            abilities = set(abilities)
            return lambda w: set(w.abilities).issuperset(abilities)
        
        return sorted(filter(_has_abilities(_abilities(task_dbo.pipeline)), self._workers), key=lambda w: w.parallel_n - w.running_num).pop()

    def enqueue(self, task_dbo, run_by):
        worker = self._select_worker(task_dbo)
        return worker.enqueue(task_dbo, run_by)

    def remove(self, key: str):
        """Remove/stop specified task"""
        job = self.get(key)
        if job:
            return job.worker.remove(job.key)


class TaskQueuePlugin(Plugin):

    def __init__(self, pmanager, n=3, workers=None) -> None:
        super().__init__(pmanager)
        self.handler = MultiWorkerTaskQueue(n, workers)
        self.handler.register_api(self.pmanager.app)
        pmanager.task_queue = self.handler
