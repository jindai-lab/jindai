"""Remote queue"""

import itertools
import json
import threading
import time

import requests
from PyMongoWrapper.dbo import create_dbo_json_encoder
from jindai.common import DictObject
from jindai.helpers import safe_import
from jindai.models import TaskDBO

from .announcer import announcer
from .taskqueue import TaskQueue


class JobProxy(DictObject):
    """Proxy remote job
    """

    def __init__(self, __dict, remote):
        super().__init__(__dict)
        self.remote = remote

    def as_dict(self):
        """Return dict representing the current job

        Returns:
            dict: dict
        """
        return dict(self, worker=None, remote=None)

    @property
    def result(self):
        """Return the result from remote job

        Returns:
            dict | None: result
        """
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


class TaskRemoteQueue(TaskQueue):
    """Remote task queue"""

    def __init__(self, entrypoint, verbose=False) -> None:
        super().__init__(verbose=verbose)
        self._base = entrypoint.rstrip('/') + '/'
        self._config = {}
        self._queue = []
        self.alive = True
        self.listen_events()

    @property
    def jobs(self):
        """Get remote workers"""
        return [JobProxy(_, self) for _ in self._queue]

    def update_config(self):
        """Update config for remote worker"""
        self._config = self.call('queue/config') or {}
        self._parallel_n = self._config.get('parallel_n', 0)

        capabilities = self._config.get('capabilities')
        if not capabilities:
            capabilities = self.call('help/pipelines') or {}
            capabilities = list(itertools.chain(
                *[v.keys() for v in capabilities.values()]))

        self._capabilities = capabilities

    def call(self, api_name, method='get', data=None, raw=False):
        """Call remote worker api"""
        method = method.upper()
        try:
            headers = {}
            if isinstance(data, dict):
                data = json.dumps(cls=create_dbo_json_encoder(
                    json.JSONEncoder), obj=data)
                headers['content-type'] = 'application/json'
            resp = requests.request(method, self._base + api_name,
                                    data=data, headers=headers, timeout=30)
            if not raw:
                resp = resp.json().get('result')
            else:
                resp = resp.content
            return resp
        except Exception as ex:
            print('Error while calling remote queue', self._base, ':', ex)

    def stop(self):
        """Stop running"""
        self.alive = False

    def remove(self, key):
        """Remove key from remote worker"""
        return self.call(f'queue/{key}', 'delete')

    def enqueue(self, task_dbo: TaskDBO, run_by: str):
        """Enqueue a TaskDBO object to remote worker"""
        return self.call('queue/', 'put', {'task': task_dbo.as_dict(), 'run_by': run_by})

    def listen_events(self):
        """Listen to remote worker messages"""

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
