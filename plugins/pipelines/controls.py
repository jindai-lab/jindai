"""Workflow control
@zhs 工作流程控制
"""

from PyMongoWrapper import F
from jindai import PipelineStage, Pipeline, Task, parser
from jindai.helpers import execute_query_expr
from jindai.models import TaskDBO


class FlowControlStage(PipelineStage):
    """Base class for flow control pipeline stages"""

    def __init__(self) -> None:
        self._next = None
        self._pipelines = [getattr(self, a) for a in dir(
            self) if isinstance(getattr(self, a), Pipeline)]
        self._verbose = False
        super().__init__()

    @property
    def logger(self):
        """Logger"""
        return lambda *x: self._logger(self.__class__.__name__, '|', *x)

    @logger.setter
    def logger(self, val):
        self._logger = val
        for pipeline in self._pipelines:
            pipeline.logger = val
            
    @property
    def verbose(self):
        """Print out debug info when verbose is set"""
        return self._verbose
    
    @verbose.setter
    def verbose(self, val):
        self._verbose = val
        for pipeline in self._pipelines:
            pipeline.verbose = val

    @property
    def next(self):
        """Next stage in pipeline"""
        return self._next

    @next.setter
    def next(self, val):
        self._next = val
        for pipeline in self._pipelines:
            if pipeline.stages:
                pipeline.stages[-1].next = val


class RepeatWhile(FlowControlStage):
    """Repeat loops
    @zhs 重复"""

    def __init__(self, pipeline, times=1, cond=''):
        """
        Args:
            pipeline (pipeline): Loop pipeline
                @zhs 要重复执行的流程
            times (int): Max repeat times
                @zhs 重复的次数
            cond (QUERY): Condition for repeat
                @zhs 重复的条件
        """
        self.times = times
        self.times_key = f'REPEATWHILE_{id(self)}_TIMES_COUNTER'
        self.cond = parser.parse(
            cond if cond else f'{self.times_key} < {times}')
        self.pipeline = Pipeline(pipeline, self.logger)
        super().__init__()

    def flow(self, paragraph, gctx):
        self.gctx = gctx
        if paragraph[self.times_key] is None:
            paragraph[self.times_key] = 0

        condition_satisfied = execute_query_expr(self.cond, paragraph)
        paragraph[self.times_key] += 1

        if condition_satisfied and self.pipeline.stages:
            self.pipeline.stages[-1].next = self
            yield paragraph, self.pipeline.stages[0]
        else:
            paragraph[self.times_key] = None
            yield paragraph, self.next

    def summarize(self, result):
        return self.pipeline.summarize(result)


class Condition(FlowControlStage):
    """Conditional execution
    @zhs 条件判断"""

    def __init__(self, cond, iftrue, iffalse):
        """
        Args:
            cond (QUERY): Condition to check
                @zhs 检查的条件
            iftrue (pipeline): Pipeline when condition is satisfied
                @zhs 条件成立时执行的流程
            iffalse (pipeline): Pipeline when condition is not satisfied
                @zhs 条件不成立时执行的流程
        """
        self.cond = parser.parse(cond)
        self.iftrue = Pipeline(iftrue, self.logger)
        self.iffalse = Pipeline(iffalse, self.logger)
        super().__init__()

    def flow(self, paragraph, gctx):
        self.gctx = gctx
        pipeline = self.iftrue
        if not execute_query_expr(self.cond, paragraph):
            pipeline = self.iffalse
        if pipeline.stages:
            yield paragraph, pipeline.stages[0]
        else:
            yield paragraph, self.next
    
    def summarize(self, result):
        self.iftrue.summarize(result)
        self.iffalse.summarize(result)
        

class CallTask(FlowControlStage):
    """Call to other task
    @zhs 调用其他任务"""

    def __init__(self, task, pipeline_only=False, params=''):
        """
        Args:
            task (TASK): Task ID
                @zhs 任务ID
            pipeline_only (bool): Join the pipeline in the current pipeline 
                @zhs 仅调用任务中的处理流程，若为 false，则于 summarize 阶段完整调用该任务
            params (QUERY): Override parameters in the task
                @zhs 设置任务中各处理流程参数
        """
        task = TaskDBO.first(F.id == task)
        assert task, f'No specified task: {task}'
        self.pipeline_only = pipeline_only
        if params:
            params = parser.parse(params)
            for key, val in params.items():
                secs = key.split('.')
                target = task.pipeline
                for sec in secs[1:-1]:
                    if sec.isnumeric():
                        assert isinstance(target, list) and len(
                            target) > int(sec), f'Index error: {sec}'
                        target = target[int(sec)][1]
                    else:
                        assert sec in target, f'No such parameter: {sec}'
                        target = target[sec]
                        if isinstance(target, list) and len(target) == 2 and \
                           isinstance(target[0], str) and isinstance(target[1], dict):
                            target = target[1]
                sec = secs[-1]
                target[sec] = val

        self._pipelines = []
        self.task = Task.from_dbo(task)
        if self.pipeline_only:
            self.pipeline = self.task.pipeline

        super().__init__()

    def flow(self, paragraph, gctx):
        self.gctx = gctx
        if self.pipeline_only and self.pipeline.stages:
            yield paragraph, self.pipeline.stages[0]
        else:
            yield paragraph, self.next

    def summarize(self, _):
        if self.pipeline_only:
            return self.pipeline.summarize()
        else:
            return self.task.execute()
