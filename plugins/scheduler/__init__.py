import re
from threading import Thread

import schedule
from bson import ObjectId
from helpers import rest
from models import F, TaskDBO, db
from plugin import Plugin


class SchedulerJob(db.DbObject):
    
    cron = str


class JobTask:

    def __init__(self, task_id: str):
        self.task_dbo = TaskDBO.first(F.id == task_id)
        self.key = ObjectId()

    def __call__(self):
        from task import Task
        self.task_dbo = TaskDBO.first(F.id == self.task_dbo.id)
        Task.from_dbo(self.task_dbo).execute()

    def __repr__(self) -> str:
        return f'{self.key}: {self.task_dbo.name}'

    @property
    def dict(self):
        return {
            'key': self.key,
            'name': self.task_dbo.name
        }


class Scheduler(Plugin):

    def cron(self, text):
        """
        Args:
            text (str): 描述任务计划的短语。格式为：every <weekday>|<number>? <unit> [at <time>] do <task id>，例：every 5 days at 3:00 do 0123456789ab0123456789ab
        """
        text = text.lower()
        a = None
        context = ''
        jobs = []

        for token in text.split():
            token = token.strip()
            if not token:
                continue
            if token == 'every':
                a = schedule.every
                context = 'start'
            elif re.match(r'\d+', token) and context == 'every':
                a = a(int(token))
                context = 'every'
            elif token.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'minute', 'minutes', 'second', 'seconds', 'hour', 'hours', 'week', 'weeks', 'day', 'days']:
                if context == 'start': a = a()
                a = getattr(a, token.lower())
                context = 'unit'
            elif re.match(r'\d{2}:\d{2}(:\d{2})?', token) and context == 'at':
                a = a.at(token)
                context = 'unit'
            elif token in ['at', 'do']:
                context = token
            elif re.match(r'[0-9a-fA-F]{24}', token) and context == 'do':
                jobs.append(a.do(JobTask(token)))
                context = 'end'
            else:
                print(f'Unknown token "{token}" in context "{context}"')

        return jobs


    def list_jobs(self, jobs):
        return [
            dict(getattr(j.job_func.func, 'dict', {}),
                repr=repr(j).split(' do')[0].lower())
            for j in jobs
        ]


    def dump_scheduler(self):
        SchedulerJob.query({}).delete()
        for j in schedule.jobs:
            SchedulerJob(cron=f"{repr(j).split(' do')[0].lower()} do {j.job_func.func.task_dbo.id}").save()


    def __init__(self, app):
        super().__init__(app)

        @app.route('/api/scheduler', methods=['GET'])
        @rest()
        def schedule_list():
            return self.list_jobs(schedule.jobs)

        @app.route('/api/scheduler', methods=['PUT'])
        @rest()
        def schedule_job(text):
            jobs = self.cron(text)
            self.dump_scheduler()
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
            for t in to_del:
                schedule.jobs.remove(t)
            self.dump_scheduler()
            return len(to_del) > 0

        def background():
            import time
            while True:
                schedule.run_pending()
                time.sleep(1)

        for j in SchedulerJob.query({}):
            self.cron(j.cron)

        Thread(target=background).start()
