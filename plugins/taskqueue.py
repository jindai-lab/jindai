"""任务队列"""
import datetime
import os
import threading
import time
import traceback
from io import BytesIO
from queue import deque

from flask import Response, jsonify, request, send_file
from PyMongoWrapper import F
from jindai import Plugin
from jindai.helpers import logined, logs_view, rest
from jindai.models import TaskDBO
from jindai.task import Task


class TasksQueue(Plugin):
    """处理任务队列
    """

    def __init__(self, app, n=3):
        """
        Args:
            n (int, optional): 最大同时处理的任务数量
        """
        super().__init__(app)
        app.task_queue = self
        self.queue = deque()
        self.results = {}
        self.task_queue = {}
        self.running = False
        self.parallel_n = n
        self._working_thread = None

        @app.route('/api/queue/', methods=['PUT'])
        @rest()
        def enqueue_task(id=''):
            task_dbo = TaskDBO.first(F.id == id)
            assert task_dbo, 'No such task, or you do not have permission.'
            task_dbo.last_run = datetime.datetime.utcnow()
            task_dbo.save()
            task_dbo.task = None
            key = self.enqueue(task_dbo, run_by=logined())
            return key

        @app.route('/api/queue/logs/<path:key>', methods=['GET'])
        @rest()
        def logs_task(key):
            task_dbo = self.find(key)
            if task_dbo:
                return logs_view(task_dbo)

            return f'No such key: {key}', 404

        @app.route('/api/queue/<path:_id>', methods=['DELETE'])
        @rest()
        def dequeue_task(task_id):
            if task_id in self.results:
                del self.results[task_id]
                # emit('queue', self.status)
                return True

            return self.remove(task_id)

        @app.route('/api/queue/<path:_id>', methods=['GET'])
        @rest(cache=True)
        def fetch_task(_id):
            if _id not in self.results:
                return Response('No such id: ' + _id, 404)

            result = self.results[_id]

            if isinstance(result, list):
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
            if result is None:
                return None

            if isinstance(result, dict) and '__file_ext__' in result and 'data' in result:
                buf = BytesIO(result['data'])
                buf.seek(0)
                return send_file(
                    buf, 'application/octstream',
                    as_attachment=True,
                    attachment_filename=os.path.basename(f"{_id}.{result['__file_ext__']}"))

            return jsonify(result)

        @app.route('/api/queue/', methods=['GET'])
        @rest()
        def list_queue():
            status = self.status
            status['running'] = [_ for _ in status['running']
                                 if logined('admin') or _.run_by == logined()]
            status['finished'] = [_ for _ in status['finished']
                                  if logined('admin') or not _.get('run_by') \
                                    or _['run_by'] == logined()
                                  ]
            status['waiting'] = [_ for _ in status['waiting']
                                 if logined('admin') or _.run_by == logined()
                                 ]
            return status

    def start(self):
        """开始处理任务"""
        self.running = True
        self._working_thread = threading.Thread(target=self.working)
        self._working_thread.start()

    @property
    def status(self) -> dict:
        """任务队列状态"""
        return {
            'running': list(self.task_queue),
            'finished': [{
                'id': k,
                'name': k.split('@')[0],
                'viewable': isinstance(v, list)
                or (isinstance(v, dict) and 'exception' in v)
                or (isinstance(v, dict) and 'redirect' in v),
                'isnull': v is None,
                'last_run': datetime.datetime.strptime(k.split('@')[-1], '%Y%m%d %H%M%S')
                .strftime('%Y-%m-%d %H:%M:%S'),
                'file_ext': 'json' if not isinstance(v, dict) else v.get('__file_ext__', 'json'),
                'run_by': v.get('run_by', '') if isinstance(v, dict) else ''
            } for k, v in self.results.items()],
            'waiting': [k for k, _ in self.queue]
        }

    def working(self):
        """处理任务队列"""
        while self.running:
            if self.queue and len(self.task_queue) < self.parallel_n:  # can run new task
                tkey, task_dbo = self.queue.popleft()
                self.task_queue[tkey] = task_dbo
                try:
                    task_dbo.task = Task.from_dbo(task_dbo)
                    task_dbo.task.run()
                except Exception as ex:
                    self.results[tkey] = {
                        'run_by': task_dbo.run_by,
                        'exception': f'初始化任务时出错: {ex.__class__.__name__}: {ex}',
                        'tracestack': traceback.format_tb(ex.__traceback__) + [
                            self.app.json_encoder().encode(task_dbo.as_dict())
                        ]}
                    self.task_queue.pop(tkey)

            elif not self.queue and not self.task_queue:  # all tasks done
                self.running = False

            else:
                done = []
                for k, val in self.task_queue.items():
                    if not val.task.alive:
                        done.append(k)
                        self.results[k] = val.task.returned
                for k in done:
                    self.task_queue.pop(k)
            time.sleep(0.5)

    def enqueue(self, val, key='', run_by=''):
        """将新任务加入队列"""
        val.run_by = run_by
        if not key:
            key = f'{val.name}@{datetime.datetime.utcnow().strftime("%Y%m%d %H%M%S")}'

        self.queue.append((key, val))

        if not self.running:
            self.start()

        return key
        # emit('queue', self.status)

    def stop(self):
        """停止运行"""
        self.running = False

    def find(self, key: str):
        """返回指定任务"""
        if key in self.task_queue:
            return self.task_queue[key]

        for k, val in self.queue:
            if k == key:
                return val
        return None

    def remove(self, key: str):
        """删除指定任务"""

        def _remove_queue(key):
            todel = None
            for todel, _ in self.queue:
                if todel == key:
                    break
            else:
                return False

            self.queue.remove(todel)
            return True

        def _remove_running(key):
            if key in self.task_queue:
                task_dbo = self.task_queue.pop(key)
                task_dbo.task.stop()
                return True

            return False

        if _remove_queue(key):
            return True

        return _remove_running(key)
