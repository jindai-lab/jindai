"""工作流程控制
"""

from pipeline import PipelineStage, Pipeline
from .utils import execute_query_expr
from models import TaskDBO, F, parser


class Repeat(PipelineStage):
    """重复"""

    def __init__(self, pipeline, times=1, until=''):
        """
        Args:
            pipeline (pipeline): 要重复执行的流程
            times (int): 重复的次数
            until (str): 重复的终止条件
        """
        self.times = times
        self.until = parser.eval(until) if until else None
        self.pipeline = Pipeline(pipeline, 1, False)

    def resolve(self, p):
        self.pipeline.logger = self.logger
        if self.until:
            while True:
                p = self.pipeline.apply(p)
                if execute_query_expr(self.until, p):
                    break
        else:
            for i in range(self.times):
                p = self.pipeline.apply(p)
        return p


class Condition(PipelineStage):
    """条件判断"""

    def __init__(self, cond, iftrue, iffalse):
        """
        Args:
            cond (str): 检查的条件
            iftrue (pipeline): 条件成立时执行的流程
            iffalse (pipeline): 条件不成立时执行的流程
        """
        self.cond = parser.eval(cond)
        self.iftrue = Pipeline(iftrue, 1, False)
        self.iffalse = Pipeline(iffalse, 1, False)

    def resolve(self, p):
        self.iftrue.logger = self.logger
        self.iffalse.logger = self.logger
        if execute_query_expr(self.cond, p):
            p = self.iftrue.apply(p)
        else:
            p = self.iffalse.apply(p)
        return p


class ConditionalAssignment(PipelineStage):
    """按条件赋值字段"""

    def __init__(self, cond, field):
        """
        Args:
            cond (str): 一行一个检查的条件，与最终要赋予的值之间用=>连缀
            field (str): 要赋值的字段
        """
        self.cond = [parser.eval(_) for _ in cond.split('\n')]
        self.field = field

    def resolve(self, p):
        for c, v in self.cond:
            if execute_query_expr(c, p):
                setattr(p, self.field, v if not isinstance(v, str) or not v.startswith('$') else getattr(p, v[1:], None))
                break
        return p


class CallTask(PipelineStage):
    """调用其他任务（流程）"""

    def __init__(self, task, pipeline_only=False):
        """
        Args:
            task (TASK): 任务ID
            pipeline_only (bool): 仅调用任务中的处理流程，若为 false，则完整调用该任务
        """
        from task import Task
        t = TaskDBO.first(F.id == task)
        assert t, '指定的任务不存在'
        self.task = Task.from_dbo(t)
        self.pipeline_only = pipeline_only

    def resolve(self, p):
        self.task.pipeline.logger = self.logger
        if self.pipeline_only:
            return self.task.pipeline.apply(p)
    
    def summarize(self, r):
        if self.pipeline_only:
            return self.task.pipeline.summarize()
        else:
            self.task.pipeline.logger = self.logger
            self.task.datasource.logger = self.logger
            return self.task.execute()
