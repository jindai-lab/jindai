"""Scheduler module"""

import datetime
import os
import re
import time
from threading import Thread

from PyMongoWrapper import ObjectId, F
from jindai import Plugin
from jindai.helpers import rest, safe_import
from jindai.models import TaskDBO, db

schedule = safe_import('schedule')


class SchedulerJob(db.DbObject):
    """DB object for scheduler jobs"""
    cron = str


class JobTask:
    """Scheduled task"""

    def __init__(self, pmanager, task_id: str):
        self.task_dbo = TaskDBO.first(F.id == task_id)
        self.key = ObjectId()
        self.pmanager = pmanager

    def __call__(self):
        dbo = TaskDBO.first(F.id == self.task_dbo.id)
        assert dbo
        dbo.last_run = datetime.datetime.utcnow()
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
                jobs.append(executor.do(JobTask(self.pmanager, token)))
                context = 'end'
            else:
                print(
                    f'Unknown token "{token}" in context "{context}": {text}')

        return jobs

    def list_jobs(self, jobs):
        """List out scheduled tasks"""
        return [
            dict(getattr(j.job_func.func, 'as_dict', {}),
                 repr=repr(j).split(' do')[0].lower())
            for j in jobs
        ]

    def reload_scheduler(self):
        """Reload jobs"""
        SchedulerJob.query({}).delete()
        if schedule.jobs:
            for j in schedule.jobs:
                SchedulerJob(
                    cron=f"{repr(j).split(' do')[0].lower()} do {j.job_func.func.task_dbo.id}"
                ).save()

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

        @app.route('/api/scheduler', methods=['GET'])
        @rest()
        def schedule_list():
            return self.list_jobs(schedule.jobs)

        @app.route('/api/scheduler', methods=['PUT'])
        @rest()
        def schedule_job(text):
            jobs = self.cron(text)
            self.reload_scheduler()
            return self.list_jobs(jobs)

        @app.route('/api/scheduler/<key>', methods=['DELETE'])
        @rest()
        def schedule_delete(key):
            key = ObjectId(key)
            to_del = [
                j
                for j in schedule.jobs
                if j.job_func.func.key == key
            ]
            for job in to_del:
                schedule.jobs.remove(job)
            self.reload_scheduler()
            return len(to_del) > 0

        for j in SchedulerJob.query({}):
            self.cron(j.cron)

        self.run_background_thread()
