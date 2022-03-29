from concurrent.futures import ThreadPoolExecutor
import inspect
from typing import Dict, List, Iterable, Tuple, Any, Union
from collections.abc import Iterable as IterableClass
from models import ImageItem, Paragraph


class PipelineStage:
    """
    流程各阶段。返回应同样为一个段落记录（可以不与数据库中的一致）。
    注意，针对段落的处理可能是并发进行的，流程处理阶段应尽量是无状态的。
    """

    def __init__(self) -> None:
        self._logger = print
        self.next = None

    @property
    def logger(self):
        return lambda *x: self._logger(self.__class__.__name__, '|', *x)

    @logger.setter
    def logger(self, val):
        self._logger = val

    def resolve(self, p: Paragraph) -> Paragraph:
        return p

    def summarize(self, r) -> Any:
        return r

    def flow(self, p) -> Tuple:
        """实现流程控制
        """
        rp = self.resolve(p)
        if isinstance(rp, IterableClass):
            for p in rp:
                yield p, self.next
        elif rp is not None:
            yield rp, self.next


class DataSourceStage(PipelineStage):
    """
    数据源阶段
    """

    class _Implementation:
        def __init__(self) -> None:
            pass

        def fetch(self):
            pass

    def __init__(self, **params) -> None:
        super().__init__()
        self.params = params

    def resolve(self, p: Paragraph) -> Paragraph:
        args = dict(**self.params)
        args.update(p.as_dict())
        Pipeline.ensure_args(type(self)._Implementation, args)
        instance = type(self)._Implementation(**args)
        instance.logger = self.logger
        yield from instance.fetch()


class Pipeline:

    pipeline_ctx = None
    
    def ensure_args(t, args):
        argnames = inspect.getfullargspec(getattr(t, '_Implementation', t).__init__).args[1:]
        toremove = []
        for k in args:
            if k not in argnames or args[k] is None:
                toremove.append(k)
        for k in toremove:
            del args[k]


    def __init__(self, stages: List[Union[Tuple[str, Dict], List, Dict, PipelineStage]], logger = print):
        """
        Args:
            stages: 表示各阶段的 Tuple[<名称>, <配置>], List[<名称>, <配置>], 形如 {$<名称> : <配置>} 的参数所构成的列表，或直接由已初始化好的 PipelineStage
        """
        
        self.stages = []
        self.logger = logger
        if stages:
            for stage in stages:
                if isinstance(stage, dict):
                    (name, kwargs), = stage.items()
                    if name.startswith('$'):
                        name = name[1:]
                    stage = (name, kwargs)
                if isinstance(stage, (tuple, list)) and len(stage) == 2 and Pipeline.pipeline_ctx:
                    name, kwargs = stage
                    t = Pipeline.pipeline_ctx[name]
                    Pipeline.ensure_args(t, kwargs)
                    stage = t(**kwargs)
                stage.logger = self.logger

                if len(self.stages):
                    self.stages[-1].next = stage
                stage.next = None
                self.stages.append(stage)

    def summarize(self):
        """
        Reduce 阶段
        """
        returned = None
        for stage in self.stages:
            returned = stage.summarize(returned)

        return returned
