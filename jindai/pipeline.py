"""Pipeline processing system for Jindai application.

This module provides:
- PipelineStage: Base class for pipeline processing stages
- DataSourceStage: Base class for data source stages
- Pipeline: Orchestrator for processing paragraphs through stages
"""

import asyncio
import inspect
import logging
import sys
import traceback
from collections import defaultdict
from collections.abc import Iterable as IterableClass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Tuple, Type, Union

import regex as re

from .config import config
from .helpers import inspect_function_signature
from .models import Paragraph
from .storage import storage

ResolveResultType = Paragraph | None
ResolveResultType |= Tuple[ResolveResultType, "PipelineStage"]
ResolveReturn = (
    ResolveResultType
    | Iterable[ResolveResultType]
    | Awaitable[ResolveResultType]
    | Awaitable[Iterable[ResolveResultType]]
)


class PipelineStage:
    """Base class for pipeline processing stages.

    Pipeline stages process paragraphs sequentially or concurrently.
    Stages should be stateless where possible to enable concurrent processing.

    Attributes:
        _log: Logging function.
        next: Next stage in the pipeline.
        gctx: Global context dictionary.
        verbose: Enable verbose logging.
        instance_name: Name for logging identification.
    """

    def __init__(self, name: str = "") -> None:
        """Initialize pipeline stage.

        Args:
            name: Instance name for logging (default: empty string).
        """
        self._log = lambda *x: logging.info(' '.join(map(str, x)))
        self.next = None
        self.gctx = {}
        self.verbose = False
        self.instance_name = name

    @classmethod
    def get_spec(cls) -> dict[str, str]:
        """Get specification info of the current stage.

        Returns:
            Dictionary with name, docstring, and argument info.
        """
        return {
            "name": cls.__name__,
            "doc": (cls.__doc__ or "").strip(),
            "args": PipelineStage._spec(cls),
        }

    @staticmethod
    def _spec(stage_cls: Type, method: str = "__init__") -> list:
        """Get argument info for a method of stage_cls.

        Args:
            stage_cls: A class to inspect.
            method: Method name to inspect (default: "__init__").

        Returns:
            List of argument specifications with name, type, description, and default.
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

        func = getattr(stage_cls, method)
        args_docs = _parse_docstring(func.__doc__ or "") or {
            argname: {"type": argtype}
            for argname, argtype in inspect_function_signature(func).items()
        }

        args_spec = inspect.getfullargspec(func)
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
        """Create a dict to represent a file in PipelineStage.

        Args:
            ext: File extension name.
            data: File data as bytes.
            **kwargs: Additional file metadata.

        Returns:
            Dictionary with file information.
        """
        file_dict = {"__file_ext__": ext, "data": data}
        file_dict.update(**kwargs)
        return file_dict

    @staticmethod
    def return_redirect(dest: str) -> dict:
        """Create a dict to represent a redirection directive.

        Args:
            dest: Destination URL.

        Returns:
            Dictionary with redirect information.
        """
        return {"__redirect__": dest}

    @staticmethod
    def parse_lines(val: Any) -> list:
        """Parse value into a list of lines.

        Args:
            val: String or list to parse.

        Returns:
            List of lines.
        """
        if isinstance(val, list):
            return val
        else:
            return [ele for ele in str(val).split("\n") if ele]

    @staticmethod
    async def parse_paths(val: Any) -> list:
        """Parse value into a list of file paths using glob patterns.

        Args:
            val: String or list of glob patterns.

        Returns:
            List of matching file paths.
        """
        files = []
        for pattern in PipelineStage.parse_lines(val):
            files.extend(storage.glob(pattern))
        return files

    @property
    def log(self) -> Callable:
        """Get logging method with instance name prefix.

        Returns:
            Logging function.
        """
        return lambda *x: self._log(
            self.instance_name or self.__class__.__name__, "|", *x
        )

    @log.setter
    def log(self, val: Callable) -> None:
        """Set the logging method.

        Args:
            val: Logging function to use.
        """
        self._log = val

    def log_exception(self, info: str, exc: Exception) -> None:
        """Log an exception with traceback.

        Args:
            info: Information about the error.
            exc: Exception instance.
        """
        self.log(info, type(exc).__name__, exc)
        self.log("\n".join(traceback.format_tb(exc.__traceback__)))

    def resolve(self, paragraph: Paragraph) -> ResolveReturn:
        """Process a paragraph and return result(s).

        Args:
            paragraph: Paragraph to process.

        Returns:
            None if excluded from further processing;
            A Paragraph object (which may not match the one in the database),
            or iterable multiple objects for next stage.
        """
        return paragraph

    def summarize(self, result: dict) -> Dict:
        """Reduce/aggregate results from the last stage.

        Args:
            result: Result from the last stage, None if current stage
                is placed at the first place.

        Returns:
            Summarized result, None for default.
        """
        return result

    async def flow(
        self, paragraph: Paragraph
    ) -> Iterable[Tuple[ResolveReturn, "PipelineStage | None"]]:
        """Flow control for pipeline processing.

        Args:
            paragraph: Paragraph to process.

        Yields:
            Tuples of (<result/iterable results>, next pipeline stage).
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
    """Base class for data source pipeline stages.

    Data source stages fetch data from external sources and
    yield paragraphs for further processing.
    """

    mappings = {}

    def __init__(self, **params) -> None:
        """Initialize data source stage.

        Args:
            **params: Parameters for data source.
        """
        super().__init__()
        self.params = params

    @classmethod
    def get_spec(cls) -> dict[str, str]:
        """Overwrite the method for getting specifications.

        Returns:
            Name, docstring and argument info.
        """
        return {
            "name": cls.__name__,
            "doc": (cls.__doc__ or "").strip(),
            "args": PipelineStage._spec(cls, "apply_params"),
        }

    def before_fetch(self, instance: "DataSourceStage") -> None:
        """Called before fetching data from data source.

        Args:
            instance: Data source instance.
        """
        pass

    def apply_params(self, **params) -> None:
        """Apply parameters to data source.

        Args:
            **params: Parameters to apply.
        """
        pass

    async def fetch(self):
        """Fetch data from data source.

        Yields:
            Paragraphs from the data source.
        """
        yield

    async def resolve(self, paragraph: Paragraph):
        """Update the parameters of the data source with the input paragraph.

        Args:
            paragraph: Paragraph object containing parameters for data source.

        Yields:
            Paragraphs from the data source.
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
    """Pipeline orchestrator for processing paragraphs through stages.

    Manages the execution flow of multiple pipeline stages,
    handling concurrent processing and result aggregation.
    """

    ctx = {}

    @staticmethod
    def ensure_args(stage_type: Type, args: Dict) -> None:
        """Ensure arguments are in compliance with stage definition.

        Args:
            stage_type: The class to validate against.
            args: Dictionary containing arguments.
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
        """Instantiate a pipeline stage by name.

        Args:
            stage_name: Name of the stage class.
            args: Arguments for stage initialization.

        Returns:
            PipelineStage instance.
        """
        if args is None:
            args = {}
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
        log: Callable = lambda *x: logging.info(' '.join(map(str, x))),
        verbose: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            stages: Pipeline stage info in one of the following forms:
                - Tuple[<PipelineStage name>, <parameters>]
                - List[<PipelineStage name>, <parameters>]
                - {<PipelineStage name> : <parameters>}
            log: Logging method (default: logging.info).
            verbose: Enable verbose logging (default: False).
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
        """Get global context dictionary.

        Returns:
            Global context dictionary.
        """
        return self._gctx

    @gctx.setter
    def gctx(self, val: dict) -> None:
        """Set global context dictionary.

        Args:
            val: Global context dictionary.
        """
        self._gctx = val
        for stage in self.stages:
            stage.gctx = val

    async def summarize(self, result: dict = None) -> dict | None:
        """Summarize pipeline results by calling summarize on each stage.

        Args:
            result: Result from previous stage (default: None).

        Returns:
            Final summarized result.
        """
        for stage in self.stages:
            result = stage.summarize(result) or result
            if asyncio.iscoroutine(result):
                result = await result

        return result
