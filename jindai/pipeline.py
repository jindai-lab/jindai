"""Pipeline"""

import inspect
import sys
import traceback
from collections import defaultdict
from collections.abc import Iterable as IterableClass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Tuple, Type, Union
import regex as re
import asyncio

from .config import instance as config
from .storage import instance as storage
from .models import Paragraph


ResolveResultType = Paragraph | None
ResolveResultType |= Tuple[ResolveResultType, "PipelineStage"]
ResolveReturn = (
    ResolveResultType
    | Iterable[ResolveResultType]
    | Awaitable[ResolveResultType]
    | Awaitable[Iterable[ResolveResultType]]
)


class PipelineStage:
    """Stages of the process.
    Note that processing against paragraphs may take place concurrently and
    that process processing stages should be stateless as far as possible.
    """

    def __init__(self, name="") -> None:
        """Initialize pipeline stage

        :param name: Instance name for logging, defaults to empty string
        :type name: str, optional
        """
        self._log = lambda *x: print(*x, file=sys.stderr)
        self.next = None
        self.gctx = {}
        self.verbose = False
        self.instance_name = name

    @classmethod
    def get_spec(cls) -> dict[str, str]:
        """Get specification info of the current stage"""
        return {
            "name": cls.__name__,
            "doc": (cls.__doc__ or "").strip(),
            "args": PipelineStage._spec(cls),
        }

    @staticmethod
    def _spec(stage_cls: Type, method="__init__") -> list:
        """Get argument info for method of stage_cls

        :param stage_cls: a class
        :type stage_cls: Type
        """

        def _parse_docstring(docstring):
            args_docs = defaultdict(dict)

            if "Args:" in docstring:
                arg_name, arg_type, arg_doc = "", "", ""
                for line in docstring.strip().split("\n"):
                    line = line.lstrip()
                    match = re.search(r"(\w+)\s+\((.+?)\):(.*)", line)
                    if match:
                        arg_name, arg_type, arg_doc = match.groups()
                        arg_doc = arg_doc.strip()
                    else:
                        arg_doc += "\n" + line

                    if arg_name:
                        args_docs[arg_name] = {
                            "type": arg_type.split(", ")[0],
                            "description": arg_doc,
                        }
            elif ":param " in docstring:
                doc_directive, arg_type, arg_name, arg_doc = "", "", "", ""
                for line in docstring.split("\n"):
                    line = line.lstrip()
                    match = re.search(r":(param|type)(\s+\w+)?\s+(\w+):(.*)$", line)
                    if match:
                        doc_directive, arg_type, arg_name, arg_doc = match.groups()
                        arg_doc = arg_doc.lstrip()
                    else:
                        if line.startswith(":"):
                            arg_name = ""
                            continue

                        arg_doc += "\n" + line

                    if arg_name:
                        if doc_directive == "type":
                            args_docs[arg_name]["type"] = arg_doc
                        else:
                            args_docs[arg_name]["description"] = arg_doc
                            if arg_type:
                                args_docs[arg_name]["type"] = arg_type.strip()
            return args_docs

        args_docs = _parse_docstring(getattr(stage_cls, method).__doc__ or "")

        args_spec = inspect.getfullargspec(getattr(stage_cls, method))
        args_defaults = dict(
            zip(reversed(args_spec.args), reversed(args_spec.defaults or []))
        )

        for arg in args_spec.args[1:]:
            if arg not in args_docs:
                args_docs[arg] = {}
            if arg in args_defaults:
                args_docs[arg]["default"] = repr(args_defaults[arg])

        return [
            {
                "name": key,
                "type": val.get("type", "").strip(),
                "description": val.get("description"),
                "default": val.get("default"),
            }
            for key, val in args_docs.items()
            if "type" in val
        ]

    @staticmethod
    def return_file(ext: str, data: bytes, **kwargs) -> dict:
        """Make a dict to represent file in PipelineStage

        :param ext: extension name
        :type ext: str
        :param data: data
        :type data: bytes
        :return: dict representing the file
        :rtype: dict
        """
        file_dict = {"__file_ext__": ext, "data": data}
        file_dict.update(**kwargs)
        return file_dict

    @staticmethod
    def return_redirect(dest: str) -> dict:
        """Make a dict to represent redirection directive

        :param dest: destination url
        :type dest: str
        :return: dict representing the redirection
        :rtype: dict
        """
        return {"__redirect__": dest}

    @staticmethod
    def parse_lines(val) -> list:
        if isinstance(val, list):
            return val
        else:
            return [ele for ele in str(val).split("\n") if ele]

    @staticmethod
    async def parse_paths(val) -> list:
        files = []
        for pattern in PipelineStage.parse_lines(val):
            files.extend(storage.glob(pattern))
        return files

    @property
    def log(self):
        """Get logging method

        :return: logging method
        :rtype: Callable
        """
        return lambda *x: self._log(
            self.instance_name or self.__class__.__name__, "|", *x
        )

    @log.setter
    def log(self, val: Callable) -> None:
        """Setting logging method

        :param val: logging method
        :type val: Callable
        """
        self._log = val

    def log_exception(self, info, exc) -> None:
        self.log(info, type(exc).__name__, exc)
        self.log("\n".join(traceback.format_tb(exc.__traceback__)))

    def resolve(self, paragraph: Paragraph) -> ResolveReturn:
        """Map period, handling paragraph.

        :param paragraph: Paragraph to process
        :type paragraph: Paragraph
        :return: None if excluded from further processing;
            A Paragraph object (which may not match the one in the database),
            or iterable multiple objects for next stage.
        :rtype: Paragraph | Iterable[Paragraph] | None
        """
        return paragraph

    def summarize(self, result) -> Dict:
        """Reduce period, handling result from the last stage

        :param result: result from the last stage, None if the current stage
            is placed at the first place
        :type result: dict
        :return: Summarized reuslt, None for default.
        :rtype: dict | None
        """
        return result

    async def flow(
        self, paragraph: Paragraph
    ) -> Iterable[Tuple[ResolveReturn, "PipelineStage | None"]]:
        """Flow control

        :param paragraph: Paragraph to process
        :type paragraph: Paragraph
        :return: Iterator
        :rtype: Tuple
        :yield: a tuple in form of (<result/iterable multiple results>, next pipeline stage)
        :rtype: Iterator[Tuple]
        """
        if self.verbose:
            self.log("Processing")
            
        results = self.resolve(paragraph)
            
        if self.verbose:
            self.log("Resolved to", type(results).__name__)
        
        def _handle(result):
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], Paragraph)
                and isinstance(result[1], PipelineStage)
            ):
                return result
            else:
                return result, self.next
        
        if inspect.isasyncgen(results):
            async for result in results:
                yield _handle(result)
        elif isinstance(results, IterableClass):
            for result in results:
                yield _handle(result)
        else:
            if asyncio.iscoroutine(results):
                results = await results
            if results is not None:
                yield results, self.next


class DataSourceStage(PipelineStage):
    """PipelineStage for data sources"""

    mappings = {}

    def __init__(self, **params) -> None:
        """Initialize data source stage

        :param params: Parameters for data source
        :type params: dict
        """
        super().__init__()
        self.params = params

    @classmethod
    def get_spec(cls) -> dict[str, str]:
        """Overwrite the method for getting specifications

        :return: Name, docstring and argument info
        :rtype: dict
        """
        return {
            "name": cls.__name__,
            "doc": (cls.__doc__ or "").strip(),
            "args": PipelineStage._spec(cls, "apply_params"),
        }

    def before_fetch(self, instance):
        """Called before fetching data from data source

        :param instance: Data source instance
        :type instance: DataSourceStage
        """
        pass

    def apply_params(self, **params):
        """Apply parameters to data source

        :param params: Parameters to apply
        :type params: dict
        """
        pass

    async def fetch(self):
        """Fetch data from data source

        :return: Iterator of paragraphs
        :rtype: Iterable[Paragraph]
        """
        yield

    async def resolve(self, paragraph: Paragraph):
        """Update the parameters of the data source with
            the input paragraph

        :param paragraph: Paragraph object containing parameters for data source
        :type paragraph: Paragraph
        :return: an iterator
        :yield: Paragraphs from the data source
        :rtype: Paragraph
        """

        args = paragraph.as_dict()
        for k, mapped in self.mappings.items():
            if k in args:
                args[mapped] = args.pop(k)

        for k, v in self.params.items():
            if args.get(k) is None or args[k] == "":
                args[k] = v

        Pipeline.ensure_args(type(self), args)

        instance = type(self)(**args)
        instance.apply_params(**args)
        instance.params = args
        instance.log = self.log
        instance.gctx = self.gctx
        instance.next = self.next
        self.before_fetch(instance)
        async for item in instance.fetch():
            yield item


class Pipeline:
    """Pipeline"""

    ctx = {}

    @staticmethod
    def ensure_args(stage_type: Type, args: Dict) -> None:
        """Ensure arguments are in compliance with definition

        :param stage_type: the class to in compliance with
        :type stage_type: Type
        :param args: a dictionary containing arguments
        :type args: Dict
        """
        argnames = {_["name"]: _["type"] for _ in stage_type.get_spec()["args"]}

        toremove = []
        for k in args:
            if k not in argnames or args[k] is None:
                toremove.append(k)
                continue

            if isinstance(args[k], str) and args[k].startswith("CONST:"):
                args[k] = config.constants.get(args[k][6:], "")

            if k in argnames:
                atype = argnames[k]
                if atype == "int" and not isinstance(args[k], int):
                    args[k] = int(args[k])
                elif atype == "float" and not isinstance(args[k], float):
                    args[k] = float(args[k])
                elif atype in ("FILES", "LINES") and isinstance(args[k], list):
                    args[k] = [
                        _
                        for _ in [
                            (
                                line.get("text", "")
                                if isinstance(line, dict)
                                else str(line)
                            )
                            for line in args[k]
                        ]
                        if _
                    ]

        for k in toremove:
            del args[k]

    @staticmethod
    def instantiate(stage_name: str, args: Dict) -> PipelineStage:
        if args.pop("disabled", False):
            return PipelineStage()
        stage_type = Pipeline.ctx[stage_name]
        Pipeline.ensure_args(stage_type, args)
        try:
            return stage_type(**args)
        except TypeError as ex:
            raise ValueError(f"Error while instantiating {stage_name} with {args}")

    def __init__(
        self,
        stages: List[Union[Tuple[str, Dict], List, Dict, PipelineStage]],
        log: Callable = print,
        verbose=False,
    ) -> None:
        """Initialize the pipeline

        :param stages: pipeline stage info in one of the following forms:
                - Tuple[<PipelineStage name>, <parameters>]
                - List[<PipelineStage name>, <parameters>]
                - {$<PipelineStage name> : <parameters>}
        :type stages: List[Union[Tuple[str, Dict], List, Dict, PipelineStage]]
        :param log: Logging method, defaults to print
        :type log: Callable, optional
        :param verbose: Enable verbose logging, defaults to False
        :type verbose: bool, optional
        """

        self.stages = []
        self.log = log
        self.verbose = verbose
        self._gctx = {}

        counter = defaultdict(int)

        if stages:
            for stage in stages:
                if isinstance(stage, dict):
                    ((name, kwargs),) = stage.items()
                    if name.startswith("$"):
                        name = name[1:]
                    stage = (name, kwargs)

                if (
                    isinstance(stage, (tuple, list))
                    and len(stage) == 2
                    and Pipeline.ctx
                ):
                    name, kwargs = stage
                    assert name in Pipeline.ctx, f"Unknown stage: {name}"
                    counter[name] += 1
                    stage = Pipeline.instantiate(name, kwargs)
                    if stage is None:
                        continue
                    stage.instance_name = f"{name}{counter[name]}"

                assert isinstance(stage, PipelineStage), f"unknown format for {stage}"

                stage.log = self.log
                stage.verbose = verbose

                if self.stages:
                    self.stages[-1].next = stage
                stage.next = None
                self.stages.append(stage)

    @property
    def gctx(self) -> dict:
        return self._gctx

    @gctx.setter
    def gctx(self, val) -> None:
        self._gctx = val
        for stage in self.stages:
            stage.gctx = val


    async def summarize(self, result=None):
        """Summarize pipeline results by calling summarize on each stage

        :param result: Result from previous stage, defaults to None
        :type result: dict, optional
        :return: Final summarized result
        :rtype: dict
        """
        for stage in self.stages:
            result = stage.summarize(result) or result
            if asyncio.iscoroutine(result): result = await result

        return result
