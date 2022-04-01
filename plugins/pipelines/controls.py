"""工作流程控制
"""

from PyMongoWrapper import F
from jindai import  PipelineStage, Pipeline
from jindai.helpers import execute_query_expr
from jindai.models import TaskDBO, parser


class FlowControlStage(PipelineStage):
    def __init__(self) -> None:
        super().__init__()
        self._next = None
        self._pipelines = [getattr(self, a) for a in dir(self) if isinstance(getattr(self, a), Pipeline)]

    @property
    def logger(self):
        return lambda *x: self._logger(self.__class__.__name__, '|', *x)

    @logger.setter
    def logger(self, val):
        self._logger = val
        for pl in self._pipelines:
            pl.logger = val
            
    @property
    def next(self):
        return self._next
    
    @next.setter
    def next(self, val):
        self._next = val
        for pl in self._pipelines:
            if pl.stages:
                pl.stages[-1].next = val
    

class RepeatWhile(FlowControlStage):
    """重复"""

    def __init__(self, pipeline, times=1, cond=''):
        """
        Args:
            pipeline (pipeline): 要重复执行的流程
            times (int): 重复的次数
            cond (QUERY): 重复的条件
        """
        self.times = times
        self.times_key = f'REPEATWHILE_{id(self)}_TIMES_COUNTER'
        self.cond = parser.eval(cond) if cond else f'{self.times_key} < {times}'
        self.pipeline = Pipeline(pipeline, self.logger)
        super().__init__()

    def flow(self, p):
        if p[self.times_key] is None:
            p[self.times_key] = 0
        
        flag = execute_query_expr(self.cond, p)
        p[self.times_key] += 1
        
        if flag and self.pipeline.stages:
            self.pipeline.stages[-1].next = self
            yield p, self.pipeline.stages[0]
        else:
            p[self.times_key] = None
            yield p, self.next


class Condition(FlowControlStage):
    """条件判断"""

    def __init__(self, cond, iftrue, iffalse):
        """
        Args:
            cond (QUERY): 检查的条件
            iftrue (pipeline): 条件成立时执行的流程
            iffalse (pipeline): 条件不成立时执行的流程
        """
        self.cond = parser.eval(cond)
        self.iftrue = Pipeline(iftrue, self.logger)
        self.iffalse = Pipeline(iffalse, self.logger)
        super().__init__()
    
    def flow(self, p):
        pl = self.iftrue
        if not execute_query_expr(self.cond, p):
            pl = self.iffalse
        if pl.stages:
            yield p, pl.stages[0]
        else:
            yield p, self.next


class CallTask(FlowControlStage):
    """调用其他任务（流程）"""

    def __init__(self, task, pipeline_only=False, params=''):
        """
        Args:
            task (TASK): 任务ID
            pipeline_only (bool): 仅调用任务中的处理流程，若为 false，则于 summarize 阶段完整调用该任务
            params (QUERY): 设置任务中各数据源和流程参数
        """
        from jindai.task import Task
        t = TaskDBO.first(F.id == task)
        assert t, '指定的任务不存在'
        self.pipeline_only = pipeline_only
        if params:
            params = parser.eval(params)
            for k, v in params.items():
                secs = k.split('.')
                target = t.pipeline
                for sec in secs[1:-1]:
                    if sec.isnumeric():
                        assert isinstance(target, list) and len(target) > int(sec), '请指定正确的下标，从0开始'
                        target = target[int(sec)][1]
                    else:
                        assert sec in target, '不存在该参数'
                        target = target[sec]
                        if isinstance(target, list) and len(target) == 2 and isinstance(target[0], str) and isinstance(target[1], dict):
                            target = target[1]
                sec = secs[-1]
                target[sec] = v
        
        self._pipelines = []
        self.task = Task.from_dbo(t)
        if self.pipeline_only:
            self.pipeline = self.task.pipeline
        
        super().__init__()
        
    def flow(self, p):
        if self.pipeline_only and self.pipeline.stages:
            yield p, self.pipeline.stages[0]
        else:
            yield p, self.next
    
    def summarize(self, r):
        if self.pipeline_only:
            return self.pipeline.summarize()
        else:
            return self.task.execute()
