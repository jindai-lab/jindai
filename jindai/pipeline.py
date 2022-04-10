"""处理流程"""
import inspect
import json
import re
from typing import Dict, List, Tuple, Any, Type, Union
from collections.abc import Iterable as IterableClass
from .models import Paragraph


class PipelineStage:
    """
    流程各阶段。返回应同样为一个段落记录（可以不与数据库中的一致）。
    注意，针对段落的处理可能是并发进行的，流程处理阶段应尽量是无状态的。
    """

    def __init__(self) -> None:
        self._logger = print
        self.next = None

    @classmethod
    def get_spec(cls):
        """获取参数信息"""
        return {
            'name': cls.__name__,
            'doc': (cls.__doc__ or '').strip(),
            'args': PipelineStage._spec(cls)
        }

    @staticmethod
    def _spec(stage_cls):
        args_docs = {}

        for line in (stage_cls.__init__.__doc__ or '').strip().split('\n'):
            match = re.search(r'(\w+)\s\((.+?)\):\s(.*)', line)
            if match:
                matched_groups = match.groups()
                if len(matched_groups) > 2:
                    args_docs[matched_groups[0]] = {'type': matched_groups[1].split(
                        ',')[0], 'description': matched_groups[2]}

        args_spec = inspect.getfullargspec(stage_cls.__init__)
        args_defaults = dict(zip(reversed(args_spec.args),
                             reversed(args_spec.defaults or [])))

        for arg in args_spec.args[1:]:
            if arg not in args_docs:
                args_docs[arg] = {}
            if arg in args_defaults:
                args_docs[arg]['default'] = json.dumps(
                    args_defaults[arg], ensure_ascii=False)

        return [
             {
                    'name': key,
                    'type': val.get('type'),
                    'description': val.get('description'),
                    'default': val.get('default')
                } for key, val in args_docs.items() if 'type' in val
        ]

    @property
    def logger(self):
        """获取日志记录方法"""
        return lambda *x: self._logger(self.__class__.__name__, '|', *x)

    @logger.setter
    def logger(self, val):
        """设置日志记录方法"""
        self._logger = val

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Map 阶段，处理段落"""

    def summarize(self, result) -> Any:
        """Reduce 阶段，处理上一步骤返回的结果"""

    def flow(self, paragraph: Union[Paragraph, IterableClass]) -> Tuple:
        """实现流程控制
        """
        results = self.resolve(paragraph)
        if isinstance(results, IterableClass):
            for result in results:
                yield result, self.next
        elif results is not None:
            yield results, self.next


class DataSourceStage(PipelineStage):
    """
    数据源阶段
    """

    @classmethod
    def get_spec(cls):
        return {
            'name': cls.__name__,
            'doc': (cls.__doc__ or '').strip(),
            'args': PipelineStage._spec(cls.Implementation)
        }

    class Implementation:
        """数据源的具体实现"""

        def __init__(self) -> None:
            self.logger = print

        def fetch(self):
            """获取实际数据"""

    def __init__(self, **params) -> None:
        super().__init__()
        self.params = params

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """用输入的语段信息更新数据源的参数"""

        args = dict(**self.params)
        args.update(paragraph.as_dict())
        Pipeline.ensure_args(type(self), args)
        instance = type(self).Implementation(**args)
        instance.logger = self.logger
        yield from instance.fetch()


class Pipeline:
    """处理流程"""

    ctx = {}

    @staticmethod
    def ensure_args(stage_type: Type, args: Dict):
        """确保参数名称符合定义"""
        argnames = [_['name'] for _ in stage_type.get_spec()['args']]

        toremove = []
        for k in args:
            if k not in argnames or args[k] is None:
                toremove.append(k)
        for k in toremove:
            del args[k]

    def __init__(self, stages: List[Union[Tuple[str, Dict], List, Dict, PipelineStage]],
                 logger=print):
        """
        Args:
            stages: 表示各阶段的 Tuple[<名称>, <配置>], List[<名称>, <配置>],
                形如 {$<名称> : <配置>} 的参数所构成的列表，或直接由已初始化好的 PipelineStage
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
                if isinstance(stage, (tuple, list)) and len(stage) == 2 and Pipeline.ctx:
                    name, kwargs = stage
                    stage_type = Pipeline.ctx[name]
                    Pipeline.ensure_args(stage_type, kwargs)
                    stage = stage_type(**kwargs)
                stage.logger = self.logger

                if self.stages:
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
