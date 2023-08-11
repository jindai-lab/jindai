"""Scheduler module"""

import datetime
import os
import re
import time
from threading import Thread

from PyMongoWrapper import ObjectId, F
from jindai import Plugin
from jindai.helpers import safe_import, APICrudEndpoint
from jindai.models import TaskDBO, db

schedule = safe_import('schedule')


class SchedulerJob(db.DbObject):
    """DB object for scheduler jobs"""
    cron = str


class ScheduledTask:
    """Scheduled task"""

    def __init__(self, pmanager, task_id: str):
        self.task_dbo = TaskDBO.first(F.id == ObjectId(task_id))
        self.key = ObjectId()
        self.pmanager = pmanager

    def __call__(self):
        dbo = TaskDBO.first(F.id == self.task_dbo.id)
        assert dbo
        self.pmanager.task_queue.enqueue(dbo, run_by='scheduler')

    def __repr__(self) -> str:
        return f'{self.key}: {self.task_dbo.name}'

    @property
    def as_dict(self):
        """Return a dict representing self"""
        return {
            'key': self.key,
            'name': self.task_dbo.name
        }
        
        
class SchedulerCrudEndpoint(APICrudEndpoint):
    
    def __init__(self, updater) -> None:        
        super().__init__('api/plugins', SchedulerJob)
        self.namespace = '/api/plugins/scheduler'
        self.updater = updater
    
    def delete(self, objs, **data):
        result = super().delete(objs)
        self.updater()
        return result
        
    def create(self, **data):
        result = super().create(**data)
        self.updater()
        return result
    
    def read(self, objs, **data):
        result = super().read(objs, **data)
        self.updater()
        return result
    
    def update(self, objs, **data):
        result = super().update(objs, **data)
        self.updater()
        return result


class Scheduler(Plugin):
    """Scheduler"""

    def cron(self, text):
        """
        Args:
            text (str): Description for task, in form of:
                every <weekday>|<number>? <unit>(s) [at <time>] do <task id>，
                e.g. every 5 days at 3:00 do 0123456789ab0123456789ab
        """
        text = text.lower()
        executor = schedule.every
        context = ''
        jobs = []

        for token in text.split():
            token = token.strip()
            if not token:
                continue
            if token == 'every':
                executor = schedule.every
                context = 'start'
            elif re.match(r'\d+', token) and context == 'start':
                executor = executor(int(token))
                context = 'every'
            elif token.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                                   'saturday', 'sunday', 'minute', 'minutes', 'second',
                                   'seconds', 'hour', 'hours', 'week', 'weeks', 'day', 'days']:
                if context == 'start':
                    executor = executor()
                executor = getattr(executor, token.lower())
                context = 'unit'
            elif re.match(r'\d{2}:\d{2}(:\d{2})?', token) and context == 'at':
                executor = executor.at(token)
                context = 'unit'
            elif token in ['at', 'do']:
                context = token
            elif re.match(r'[0-9a-fA-F]{24}', token) and context == 'do':
                jobs.append(executor.do(ScheduledTask(self.pmanager, token)))
                context = 'end'
            else:
                print(
                    f'Unknown token "{token}" in context "{context}": {text}')

        return jobs

    def reload_scheduler(self):
        """Reload jobs"""
        schedule.jobs = []
        for job in SchedulerJob.query():
            self.cron(job.cron)
            
        if schedule.jobs:
            self.run_background_thread()

    def run_background_thread(self):
        """Run in background"""

        if self._thread is not None:
            return

        def background():
            while self.running and schedule.jobs:
                schedule.run_pending()
                if os.path.exists('restarting'):
                    self.running = False
                time.sleep(1)
            self._thread = None

        self._thread = Thread(target=background)
        self._thread.start()

    def __init__(self, pmanager, **_):
        super().__init__(pmanager)
        self._thread = None
        self.running = True

        app = self.pmanager.app
        SchedulerCrudEndpoint(self.reload_scheduler).bind(app, role='admin')
