"""Workflow control
@zhs 工作流程控制
"""

from PyMongoWrapper import F
from bson import ObjectId
from jindai import PipelineStage, Pipeline, Task, parser
from jindai.helpers import evaluateqx
from jindai.models import TaskDBO


class FlowControlStage(PipelineStage):
    """Base class for flow control pipeline stages"""

    def __init__(self) -> None:
        self._verbose = False
        self._next = None
        self._pipelines = [getattr(self, a) for a in dir(
            self) if isinstance(getattr(self, a), Pipeline)]
        super().__init__()

    @property
    def logger(self):
        """Logger"""
        return lambda *x: self._logger(self.instance_name or self.__class__.__name__, '|', *x)

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

        try:
            condition_satisfied = evaluateqx(self.cond, paragraph)
        except Exception as ex:
            self.log_exception('failed to evaluate qx', ex)
            return

        paragraph[self.times_key] += 1

        if condition_satisfied and self.pipeline.stages:
            self.pipeline.stages[-1].next = self
            yield paragraph, self.pipeline.stages[0]
        else:
            del paragraph[self.times_key]
            yield paragraph, self.next

    def summarize(self, result):
        return self.pipeline.summarize(result)


class ForEach(FlowControlStage):
    """For loops
    @zhs 枚举循环"""

    def __init__(self, pipeline, as_name, input_value) -> None:
        """
        Args:
            pipeline (pipeline): Loop pipeline
                @zhs 要重复执行的流程
            as_name (int): Variable name
                @zhs 枚举的变量名
            input_value (QUERY): Value to iterate
                @zhs 要枚举的范围
        """
        self.as_name = as_name
        self.input_value = parser.parse(input_value)
        self.pipeline = Pipeline(pipeline, self.logger)
        super().__init__()

    def flow(self, paragraph, gctx):
        self.gctx = gctx
        try:
            input_value = evaluateqx(self.input_value, paragraph)
        except Exception as ex:
            self.log_exception('failed to evaluate qx', ex)
            return

        for iterval in input_value:
            paragraph[self.as_name] = iterval
            yield paragraph, self.pipeline.stages[0]

        del paragraph[self.as_name]
        yield paragraph, self.next


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
        try:
            if not evaluateqx(self.cond, paragraph):
                pipeline = self.iffalse
        except Exception as ex:
            self.log_exception('failed to evaluate qx', ex)
            return

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

    def __init__(self, task, skip=0, params=''):
        """
        Args:
            task (TASK): Task ID
                @zhs 任务ID
            skip (int): Skip first (n) stages in pipeline
                @zhs 跳过处理流程中的前 n 个阶段 
            params (QUERY): Override parameters in the task
                @zhs 设置任务中各处理流程参数
        """
        task = TaskDBO.first(F.id == ObjectId(task))
        assert task, f'No specified task: {task}'
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

        task = Task.from_dbo(task)
        self.pipeline = task.pipeline
        self.pipeline.stages = self.pipeline.stages[skip:]

        super().__init__()

    def flow(self, paragraph, gctx):
        self.gctx = gctx
        if self.pipeline.stages:
            yield paragraph, self.pipeline.stages[0]
        else:
            yield paragraph, self.next

    def summarize(self, _):
        return self.pipeline.summarize()


class RunTask(CallTask):
    """Run task at summarization phrase
    @zhs 在收尾阶段运行任务"""

    def __init__(self, task, params=''):
        """
        Args:
            task (TASK): Task ID
                @zhs 任务ID
            params (QUERY): Override parameters in the task
                @zhs 设置任务中各处理流程参数
        """
        super().__init__(task, 0, params)

    def flow(self, paragraph, gctx):
        self.gctx = gctx
        yield paragraph, self.next

    def summarize(self, _):
        return self.task.execute()
