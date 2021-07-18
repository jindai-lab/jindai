"""工作流程控制
"""

from pipeline import PipelineStage, Pipeline
from PyMongoWrapper import QueryExprParser
from .utils import execute_query_expr
import re
_p = QueryExprParser(allow_spacing=True)


def build_pipeline(pipeline):
    from task import Task
    return Pipeline([Task.pipeline_ctx[name](**args) for name, args in pipeline], 1, False)


class Repeat(PipelineStage):

    def __init__(self, pipeline, times=1, until=''):
        """
        Args:
            pipeline (pipeline): 要重复执行的流程
            times (int): 重复的次数
            until (str): 重复的终止条件
        """
        self.times = times
        self.until = _p.eval(until) if until else None
        self.pipeline = build_pipeline(pipeline)

    def resolve(self, p):
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

    def __init__(self, cond, iftrue, iffalse):
        """
        Args:
            cond (str): 检查的条件
            iftrue (pipeline): 条件成立时执行的流程
            iffalse (pipeline): 条件不成立时执行的流程
        """
        self.cond = _p.eval(cond)
        self.iftrue = build_pipeline(iftrue)
        self.iffalse = build_pipeline(iffalse)

    def resolve(self, p):
        if execute_query_expr(self.cond, p):
            p = self.iftrue.apply(p)
        else:
            p = self.iffalse.apply(p)
        return p


class ConditionalAssignment(PipelineStage):

    def __init__(self, cond, field):
        """
        Args:
            cond (str): 一行一个检查的条件，与最终要赋予的值之间用=>连缀
            field (str): 要赋值的字段
        """
        self.cond = [_p.eval(_) for _ in cond.split('\n')]
        self.field = field

    def resolve(self, p):
        for c, v in self.cond:
            if execute_query_expr(c, p):
                setattr(p, self.field, v if not isinstance(v, str) or not v.startswith('$') else getattr(p, v[1:], None))
                break
        return p
