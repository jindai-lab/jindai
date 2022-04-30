"""Pipeline"""
import inspect
import json
import re
from typing import Dict, List, Tuple, Any, Type, Union, Callable
from collections.abc import Iterable as IterableClass
from collections import defaultdict
from .models import Paragraph


class PipelineStage:
    """Stages of the process.
    Note that processing against paragraphs may take place concurrently and that process processing stages should be stateless as far as possible.
    """

    def __init__(self) -> None:
        self._logger = print
        self.next = None

    @classmethod
    def get_spec(cls):
        """Get specification info of the current stage"""
        return {
            'name': cls.__name__,
            'doc': (cls.__doc__ or '').strip(),
            'args': PipelineStage._spec(cls)
        }

    @staticmethod
    def _spec(stage_cls: Type):
        """Get argument info for `__init__` method of stage_cls

        :param stage_cls: a class
        :type stage_cls: Type
        """

        def _parse_docstring(docstring):
            args_docs = defaultdict(dict)

            if 'Args:' in docstring:
                for line in docstring.strip().split('\n'):
                    arg_name, arg_type, arg_doc = '', '', ''
                    match = re.search(r'(\w+)\s+\((.+?)\):\s+(.*)', line)
                    if match:
                        arg_name, arg_type, arg_doc = match.groups()
                    else:
                        arg_doc += line

                    if arg_name:
                        args_docs[arg_name] = {
                            'type': arg_type.split(',')[0],
                            'description': arg_doc
                        }
            elif ':param ' in docstring:
                doc_directive, arg_type, arg_name, arg_doc = '', '', '', ''
                for line in docstring.strip().split('\n'):
                    match = re.search(
                        r':(param|type)(\s+\w+)?\s+(\w+):\s(.*)$', line)
                    if match:
                        doc_directive, arg_type, arg_name, arg_doc = match.groups()
                    else:
                        arg_doc += line.lstrip()

                    if arg_name:
                        if doc_directive == 'type':
                            args_docs[arg_name]['type'] = arg_doc
                        else:
                            args_docs[arg_name]['description'] = arg_doc
                            if arg_type and arg_type.strip():
                                args_docs[arg_name]['type'] = arg_type.strip()
            return args_docs

        args_docs = _parse_docstring(stage_cls.__init__.__doc__ or '')

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
        """Get logging method

        :return: logging method
        :rtype: Callable
        """
        return lambda *x: self._logger(self.__class__.__name__, '|', *x)

    @logger.setter
    def logger(self, val: Callable):
        """Setting logging method

        :param val: logging method
        :type val: Callable
        """
        self._logger = val

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Map period, handling paragraph.

        :param paragraph: Paragraph to process
        :type paragraph: Paragraph
        :return: None if excluded from further processing;
            A Paragraph object (which may not match the one in the database),
            or iterable multiple objects for next stage.
        :rtype: Paragraph | Iterable[Paragraph] | None
        """

    def summarize(self, result) -> Dict:
        """Reduce period, handling result from the last stage

        :param result: result from the last stage, None if the current stage
            is placed at the first place
        :type result: dict
        :return: Summarized reuslt, None for default.
        :rtype: dict | None
        """

    def flow(self, paragraph: Union[Paragraph, IterableClass]) -> Tuple:
        """Flow control

        :param paragraph: Paragraph to process
        :type paragraph: Union[Paragraph, IterableClass]
        :return: Iterator
        :rtype: Tuple
        :yield: a tuple in form of (<result/iterable multiple results>, next pipeline stage)
        :rtype: Iterator[Tuple]
        """
        results = self.resolve(paragraph)
        if isinstance(results, IterableClass):
            for result in results:
                yield result, self.next
        elif results is not None:
            yield results, self.next


class DataSourceStage(PipelineStage):
    """PipelineStage for data sources
    """

    @classmethod
    def get_spec(cls):
        """Overwrite the method for getting specifications

        :return: Name, docstring and argument info
        :rtype: dict
        """
        return {
            'name': cls.__name__,
            'doc': (cls.__doc__ or '').strip(),
            'args': PipelineStage._spec(cls.Implementation)
        }

    class Implementation:
        """Implementing the data source"""

        def __init__(self) -> None:
            self.logger = print

        def fetch(self):
            """Yield data (Paragraph objects)"""

    def __init__(self, **params) -> None:
        super().__init__()
        self.params = params

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Update the parameters of the data source with
            the input paragraph

        :param paragraph: Paragraph object containing parameters for data source
        :type paragraph: Paragraph
        :return: an iterator
        :yield: Paragraphs from the data source
        :rtype: Paragraph
        """

        args = dict(**self.params)
        args.update(paragraph.as_dict())
        Pipeline.ensure_args(type(self), args)
        instance = type(self).Implementation(**args)
        instance.logger = self.logger
        yield from instance.fetch()


class Pipeline:
    """Pipeline"""

    ctx = {}

    @staticmethod
    def ensure_args(stage_type: Type, args: Dict):
        """Ensure arguments are in compliance with definition

        :param stage_type: the class to in compliance with
        :type stage_type: Type
        :param args: a dictionary containing arguments
        :type args: Dict
        """
        argnames = [_['name'] for _ in stage_type.get_spec()['args']]

        toremove = []
        for k in args:
            if k not in argnames or args[k] is None:
                toremove.append(k)
        for k in toremove:
            del args[k]

    def __init__(self, stages: List[Union[Tuple[str, Dict], List, Dict, PipelineStage]],
                 logger: Callable = print):
        """Initialize the pipeline

        :param stages: pipeline stage info in one of the following forms:
                - Tuple[<PipelineStage name>, <parameters>]
                - List[<PipelineStage name>, <parameters>]
                - {$<PipelineStage name> : <parameters>}
        :type stages: List[Union[Tuple[str, Dict], List, Dict, PipelineStage]]
        :param logger: Logging method, defaults to print
        :type logger: Callable, optional
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
                assert isinstance(
                    stage, PipelineStage), f'unknown format for {stage}'

                stage.logger = self.logger

                if self.stages:
                    self.stages[-1].next = stage
                stage.next = None
                self.stages.append(stage)

    def summarize(self):
        """Reduce period
        """
        returned = None
        for stage in self.stages:
            returned = stage.summarize(returned)

        return returned
