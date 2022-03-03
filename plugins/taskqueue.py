import datetime
import logging
import os
import threading
import time
import traceback
from io import BytesIO
from queue import deque

from flask import Response, jsonify, request, send_file
from helpers import logined, logs_view, rest
from models import F, TaskDBO
from plugin import Plugin
from task import Task


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
        self.taskdbos = {}
        self.running = False
        self.n = n

        @app.route('/api/queue/', methods=['PUT'])
        @rest()
        def enqueue_task(id=''):
            t = TaskDBO.first(F.id == id)
            assert t, 'No such task, or you do not have permission.'
            t.last_run = datetime.datetime.utcnow()
            t.save()
            t._task = None
            key = self.enqueue(t, run_by=logined())
            return key

        @app.route('/api/queue/logs/<path:key>', methods=['GET'])
        @rest()
        def logs_task(key):
            t = self.find(key)
            if t:
                return logs_view(t)
            else:
                return f'No such key: {key}', 404

        @app.route('/api/queue/<path:_id>', methods=['DELETE'])
        @rest()
        def dequeue_task(_id):
            if _id in self.results:
                del self.results[_id]
                # emit('queue', self.status)
                return True
            else:
                return self.remove(_id)

        @app.route('/api/queue/<path:_id>', methods=['GET'])
        @rest(cache=True)
        def fetch_task(_id):
            if _id not in self.results:
                return Response('No such id: ' + _id, 404)
            r = self.results[_id]

            if isinstance(r, list):
                offset, limit = int(request.args.get('offset', 0)), int(
                    request.args.get('limit', 0))
                if limit == 0:
                    return {
                        'results': r,
                        'total': len(r)
                    }
                else:
                    return {
                        'results': r[offset:offset+limit],
                        'total': len(r)
                    }
            elif r is None:
                return None
            else:
                if isinstance(r, dict) and '__file_ext__' in r and 'data' in r:
                    buf = BytesIO(r['data'])
                    buf.seek(0)
                    return send_file(buf, 'application/octstream', as_attachment=True, attachment_filename=os.path.basename(_id + '.' + r['__file_ext__']))
                else:
                    return jsonify(r)

        @app.route('/api/queue/', methods=['GET'])
        @rest()
        def list_queue():
            st = self.status
            st['running'] = [_ for _ in st['running'] if logined('admin') or _.run_by == logined()]
            st['finished'] = [_ for _ in st['finished'] if logined('admin') or not _.get('run_by') or _['run_by'] == logined()]
            st['waiting'] = [_ for _ in st['waiting'] if logined('admin') or _.run_by == logined()]
            return st


    def start(self):
        """开始处理任务"""
        self.running = True
        self._workingThread = threading.Thread(target=self.working)
        self._workingThread.start()

    @property
    def status(self) -> dict:
        """任务队列状态"""
        return {
            'running': list(self.taskdbos),
            'finished': [{
                'id': k,
                'name': k.split('@')[0],
                'viewable': isinstance(v, list) or (isinstance(v, dict) and 'exception' in v) or (isinstance(v, dict) and 'redirect' in v),
                'isnull': v is None,
                'last_run': datetime.datetime.strptime(k.split('@')[-1], '%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                'file_ext': 'json' if not isinstance(v, dict) else v.get('__file_ext__', 'json'),
                'run_by': v.get('run_by', '') if isinstance(v, dict) else ''
            } for k, v in self.results.items()],
            'waiting': [k for k, _ in self.queue]
        }

    def working(self):
        """处理任务队列"""
        while self.running:
            if self.queue and len(self.taskdbos) < self.n:  # can run new task
                tkey, t = self.queue.popleft()
                self.taskdbos[tkey] = t
                try:
                    t._task = Task.from_dbo(t)
                    t._task.run()
                except Exception as ex:
                    self.results[tkey] = {'run_by': t.run_by, 'exception': f'初始化任务时出错: {ex.__class__.__name__}: {ex}', 'tracestack': traceback.format_tb(ex.__traceback__) + [
                        self.app.json_encoder().encode(t.as_dict())
                    ]}
                    self.taskdbos.pop(tkey)

            elif not self.queue and not self.taskdbos:  # all tasks done
                self.running = False

            else:
                done = []
                for k, v in self.taskdbos.items():
                    if not v._task.alive:
                        done.append(k)
                        self.results[k] = v._task.returned
                for k in done:
                    self.taskdbos.pop(k)
            time.sleep(0.5)

    def enqueue(self, val, key='', run_by=''):
        """将新任务加入队列"""
        val.run_by = run_by
        if not key:
            key = f'{val.name}@{datetime.datetime.utcnow().strftime("%Y%m%d %H%M%S")}'

        self.queue.append((key, val))

        if not self.running:
            logging.info('start background thread')
            self.start()

        return key
        # emit('queue', self.status)

    def stop(self):
        """停止运行"""
        self.running = False

    def find(self, key: str):
        """返回指定任务"""
        if key in self.taskdbos:
            return self.taskdbos[key]
        else:
            for k, v in self.queue:
                if k == key:
                    return v
            return None

    def remove(self, key: str):
        """删除指定任务"""

        def _remove_queue(key):
            for todel, _ in self.queue:
                if todel == key:
                    break
            else:
                return False
            self.queue.remove(todel)
            return True

        def _remove_running(key):
            if key in self.taskdbos:
                t = self.taskdbos.pop(key)
                t._task.stop()
            else:
                return False

        if _remove_queue(key):
            return True
        else:
            return _remove_running(key)
