from typing import Callable, Dict, Iterable, List, Tuple, Any, Union
from PyMongoWrapper import MongoResultSet
from concurrent.futures import ThreadPoolExecutor
from models import Paragraph


class PipelineStage:
    """
    流程各阶段。返回应同样为一个段落记录（可以不与数据库中的一致）。
    注意，针对段落的处理可能是并发进行的。
    """

    def resolve(self, p : Paragraph) -> Paragraph:
        return p

    def summarize(self, r) -> Any:
        return r


class Pipeline:

    pipeline_ctx = None
    
    def __init__(self, stages : List[Union[Tuple[str, Dict], List, Dict, PipelineStage]], concurrent : int, resume_next : bool, logger : Callable = print):
        """
        Args:
            stages: 表示各阶段的 Tuple[<名称>, <配置>], List[<名称>, <配置>], 形如 {$<名称> : <配置>} 的参数所构成的列表，或直接由已初始化好的 PipelineStage
            concurrent (int): 并发运行的数量
            resume_next (bool): 当某个处理阶段发生错误时是否继续
        """
        self.logger = logger
        
        self.stages = []
        if stages:
            for stage in stages:
                if isinstance(stage, dict):
                    (name, kwargs), = stage.items()
                    if name.startswith('$'): name = name[1:]
                    stage = (name, kwargs)
                if isinstance(stage, (tuple, list)) and len(stage) == 2 and Pipeline.pipeline_ctx:
                    name, kwargs = stage
                    stage = Pipeline.pipeline_ctx[name](**kwargs)
                stage.logger = lambda *args: self.logger(stage.__class__.__name__, *args)
                self.stages.append(stage)

        self.concurrent = concurrent
        self.resume_next = resume_next
        self.exception = None

    def stop(self):
        self.exception = InterruptedError()
        
    def apply(self, p : Paragraph):
        if self.exception:
            ex = self.exception
            self.exception = None
            raise ex
            
        if not self.stages:
            return p
        
        for stage in self.stages:
            try:
                p = stage.resolve(p)
                if not p: return
            except Exception as ex:
                if not self.resume_next:
                    raise ex

        return p

    def applyParagraphs(self, rs : Union[MongoResultSet, Iterable[Paragraph]]):
        """
        处理段落
        Args:
            rs (MongoResultSet | Iterable[Paragraph]): 要处理的各个 Paragraph
        """
        if not self.stages:
            return rs

        if self.concurrent > 1:
            def _update_and_do(p):
                p = self.apply(p)
                return p

            with ThreadPoolExecutor(max_workers=self.concurrent) as te:
                r = te.map(_update_and_do, rs)
                return [_ for _ in r if _]

        else:
            def _seq():
                for p in rs:
                    p = self.apply(p)
                    if p: yield p
            
            return _seq()

    def summarize(self):
        """
        Reduce 阶段
        """
        returned = None
        for stage in self.stages:
            returned = stage.summarize(returned)

        return returned
