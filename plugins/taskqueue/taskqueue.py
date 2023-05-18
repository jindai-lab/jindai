"""Task Queue"""
import datetime
import json
import os
from io import BytesIO
import queue

from flask import Response, jsonify, request, send_file, stream_with_context
from PyMongoWrapper import F
from jindai.helpers import logined, rest
from jindai.models import TaskDBO
from jindai.pipeline import Pipeline

from .announcer import announcer


class TaskQueue:
    """Handling queue
    """

    def __init__(self, n=3, verbose=False):
        """
        Args:
            n (int, optional): maximal concurrent tasks
        """
        self._parallel_n = n
        self._capabilities = list(Pipeline.ctx)
        self.verbose = verbose

    @property
    def parallel_n(self) -> int:
        """Get maximal concurrent tasks number

        Returns:
            int
        """
        return self._parallel_n

    @property
    def capabilities(self) -> list:
        """Capabilities of the current worker

        Returns:
            list: names of supported pipeline stages
        """
        return self._capabilities

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

            if not run_by or not logined('admin'):
                run_by = logined()

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
                'capabilities': self.capabilities,
            }

        @app.route('/api/queue/events', methods=['GET'])
        @rest()
        def listen_events():
            """Provide event stream
            """

            def stream():
                messages = announcer.listen()
                yield f'data: {json.dumps(filter_status())}\n\n'
                while True:
                    try:
                        msg = messages.get(timeout=20)
                    except queue.Empty:
                        break
                    if msg == 'updated':
                        yield f'data: {json.dumps(filter_status())}\n\n'
                    elif msg == 'pulse':
                        continue
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
        
        @app.route('/api/queue/logs/<path:key>', methods=['GET'])
        @rest()
        def fetch_task_log(key):
            job = self.get(key)
            if not job or job.status != 'stopped':
                return 'Not found or not finished', 404

            return list(job.logs)

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
        raise NotImplementedError()

    def get(self, key: str):
        """Get job by key"""
        for j in self.jobs:
            if j.key == key:
                return j

    def remove(self, key):
        """Remove job with specified key

        Args:
            key (str): key

        Raises:
            NotImplementedError: remove job
        """
        raise NotImplementedError()
