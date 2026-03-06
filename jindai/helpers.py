"""Helper functions for Jindai application.

This module provides utility classes and functions for:
- Automatic model unloading to manage memory
- Word stemming for text processing
- Safe module importing with auto-install
- Context discovery for plugins
- Function signature inspection
- Safe expression evaluation
"""

import gc
import glob
import importlib
import inspect
import logging
import os
import subprocess
import sys
import threading
import time
from threading import Lock
from typing import (
    Callable,
    Dict,
    Type,
    List,
    Any,
    Callable,
    Dict,
    Literal,
    Union,
    get_args,
    get_origin,
)

from asteval import Interpreter
import jieba3
import nltk.stem.snowball
import numpy as np
import regex as re
import torch
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

jieba = jieba3.jieba3()


class AutoUnloadSentenceTransformer:
    """SentenceTransformer model automatic loading/unloading helper.

    Automatically unloads the model to release memory/GPU memory when idle
    for a specified timeout period (default 5 minutes). This helps manage
    memory usage in long-running applications.

    Attributes:
        model_name_or_path: Model name or path for SentenceTransformer.
        idle_timeout: Seconds of inactivity before unloading (default: 300).
    """

    def __init__(self, model_name_or_path: str, idle_timeout: int = 300) -> None:
        """Initialize the model manager.

        Args:
            model_name_or_path: Model name (e.g., all-MiniLM-L6-v2) or local model path,
                consistent with SentenceTransformer.
            idle_timeout: Idle timeout in seconds, default 300 seconds (5 minutes).
        """
        self.model_name_or_path = model_name_or_path
        self.idle_timeout = idle_timeout  # Idle timeout threshold
        self.model = None  # Core model object, initially empty (lazy loading)
        self.last_used_time = time.time()  # Last used timestamp
        self.lock = threading.Lock()  # Thread-safe lock
        self._monitor_thread = None  # Monitoring thread
        self._stop_monitor = False  # Monitoring thread stop flag

        # Start idle monitoring thread (daemon thread that exits with main thread)
        self._start_monitor()

    def _start_monitor(self) -> None:
        """Start the model idle monitoring background thread."""

        def monitor_loop():
            """Monitor model usage and unload when idle timeout is reached."""
            while not self._stop_monitor:
                time.sleep(10)  # Check every 10 seconds to reduce CPU usage
                with self.lock:
                    # Trigger unloading when both conditions are met:
                    # 1. Model is loaded 2. Current time - last used time > timeout threshold
                    if (
                        self.model is not None
                        and (time.time() - self.last_used_time) > self.idle_timeout
                    ):
                        self._unload_model()

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _load_model(self) -> None:
        """Load the model if not already loaded."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name_or_path)
        # Update last used time on each load/reuse
        self.last_used_time = time.time()

    def _unload_model(self) -> None:
        """Unload the model to release memory/GPU memory completely."""
        if self.model is not None:
            logging.info(
                f"[Auto-unload] Model idle for {self.idle_timeout} seconds, releasing resources..."
            )
            # Core: Delete the model object
            del self.model
            self.model = None

            # Force Python garbage collection to release memory
            gc.collect()

            # Critical step: Clear PyTorch GPU memory cache (required for GPU, otherwise GPU memory won't be released)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            logging.info(f"[Auto-unload] Model resources released")

    def encode(self, sentences, **kwargs) -> np.ndarray:
        """Wrap the native encode method for seamless usage.

        Args:
            sentences: Single sentence or list of sentences, consistent with native method.
            kwargs: Other encode parameters such as convert_to_tensor, normalize_embeddings, etc.

        Returns:
            Sentence embeddings as numpy array.
        """
        with self.lock:
            self._load_model()  # Load if not present, reuse and update time
            # Call native model method
            embeddings = self.model.encode(sentences, **kwargs)
            # Update last used time
            self.last_used_time = time.time()
            return embeddings

    def encode_batch(self, sentences, **kwargs):
        """Wrap the native encode_batch method for efficient batch encoding.

        Args:
            sentences: 2D list of sentences.
            kwargs: Other parameters.

        Returns:
            Batch sentence embeddings as numpy array.
        """
        with self.lock:
            self._load_model()
            embeddings = self.model.encode_batch(sentences, **kwargs)
            self.last_used_time = time.time()
            return embeddings

    def __del__(self) -> None:
        """Clean up when object is destroyed - unload model and stop monitoring thread."""
        self._stop_monitor = True
        with self.lock:
            self._unload_model()


from typing import Any

from nltk.stem.snowball import SnowballStemmer


class WordStemmer:
    """Word stemming utility for text processing.

    Provides stemming functionality for multiple languages using
    the Snowball stemmer from NLTK.
    """

    _language_stemmers = {"en": nltk.stem.snowball.SnowballStemmer("english")}

    _language_names = {
        # ISO-639-1 code to language name mapping
        "ar": "arabic",  # Arabic
        "da": "danish",  # Danish
        "nl": "dutch",  # Dutch
        "en": "english",  # English
        "fi": "finnish",  # Finnish
        "fr": "french",  # French
        "de": "german",  # German
        "hu": "hungarian",  # Hungarian
        "it": "italian",  # Italian
        "no": "norwegian",  # Norwegian (ISO 639-1 for Norwegian)
        "xx": "porter",  # Porter (custom code, no standard ISO-639-1)
        "pt": "portuguese",  # Portuguese
        "ro": "romanian",  # Romanian
        "ru": "russian",  # Russian
        "es": "spanish",  # Spanish
        "sv": "swedish",  # Swedish
    }

    @staticmethod
    def get_stemmer(lang: str) -> SnowballStemmer:
        """Get stemmer for a language.

        Args:
            lang: Language code (ISO-639-1) or language name.

        Returns:
            SnowballStemmer instance for the language.
        """
        stemmer = nltk.stem.snowball.SnowballStemmer
        lang = WordStemmer._language_names.get(lang, lang)
        if lang not in WordStemmer._language_stemmers:
            if lang not in stemmer.languages:
                return WordStemmer._language_stemmers["en"]
            stemmer = stemmer(lang)
            WordStemmer._language_stemmers[lang] = stemmer
        return WordStemmer._language_stemmers[lang]

    def stem_tokens(self, lang: str, tokens: list) -> list:
        """Stem a list of tokens.

        Args:
            lang: Language code for stemming.
            tokens: List of words to stem.

        Returns:
            List of stemmed words.
        """
        tokens = [WordStemmer.get_stemmer(lang).stem(_) for _ in tokens if _]
        return tokens

    def stem_from_params(self, word: str, lang: str = "en") -> dict[str, Any]:
        """Add stem() function for query processing.

        Args:
            word: Word to stem.
            lang: Language code (default: "en").

        Returns:
            Dictionary with stemmed keyword under 'keywords' key.
        """
        assert isinstance(lang, str) and isinstance(
            word, str
        ), f"Parameter type error for stem function: got {type(word)} and {type(lang)}"
        return {"keywords": self.stem_tokens(lang, [word])[0]}


_pip_lock = Lock()


def safe_import(module_name: str, package_name: str = ""):
    """Import a module and install it if not installed.

    Args:
        module_name: Name of the module to import.
        package_name: Name of the package to install. Defaults to module_name.

    Returns:
        Imported module object.
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        with _pip_lock:
            subprocess.call(
                [sys.executable, "-m", "pip", "install", package_name or module_name]
            )
    return importlib.import_module(module_name)


RE_DIGITS = re.compile(r"[\+\-]?\d+")


def get_context(directory: str, parent_class: Type, *sub_dirs: str) -> Dict:
    """Get context dictionary for given directory and parent class.

    Scans the directory for Python files and returns a dictionary of
    class names to class objects that inherit from the specified parent class.

    Args:
        directory: Directory path relative to the working directory.
        parent_class: Parent class of all defined classes to include.
        sub_dirs: Subdirectories to search (optional).

    Returns:
        Dictionary in form of {"ClassName": Class}.
    """

    def _prefix(sub_dir, name):
        """Prefixing module name for import."""
        dirpath = directory
        if sub_dir and sub_dir != ".":
            dirpath += os.sep + sub_dir
        return dirpath.replace(os.sep, ".") + "." + name

    if len(sub_dirs) == 0:
        sub_dirs = [""]
    modules = []
    for sub_dir in sub_dirs:
        modules += [
            _prefix(sub_dir, os.path.basename(f).split(".")[0])
            for f in glob.glob(os.path.join(directory, sub_dir, "*.py"))
        ] + [
            _prefix(sub_dir, f.split(os.path.sep)[-2])
            for f in glob.glob(os.path.join(directory, sub_dir, "*/__init__.py"))
        ]
    ctx = {}
    for module_name in modules:
        try:
            logging.info("Loading", module_name)
            module = importlib.import_module(module_name)
            for k in module.__dict__:
                if (
                    k != parent_class.__name__
                    and not k.startswith("_")
                    and isinstance(module.__dict__[k], type)
                    and issubclass(module.__dict__[k], parent_class)
                ):
                    ctx[k] = module.__dict__[k]
        except Exception as exception:
            logging.error("Error while importing", module_name, ":", exception)

    return ctx


def inspect_function_signature(func: Callable) -> Dict[str, str]:
    """Extract parameter names and types from a function signature.

    This function uses Python's type hints to determine parameter types.
    It supports:
    - Basic types: int, float, str, bool
    - Optional types: Optional[T] -> T
    - Union types: Union[A, B] -> str (complex unions default to str)
    - List types: List[T] -> {isArray: True, itemType: T}
    - Literal types: Literal['a', 'b'] -> {options: ['a', 'b']}
    - Pydantic models: Extracts all fields recursively
    - Callable types: Returns 'Callable'
    - Any type: Returns 'Any'

    Args:
        func: Function to inspect.

    Returns:
        Dictionary mapping parameter names to type information.
        Type information can be:
        - Simple string for basic types (e.g., 'int', 'str')
        - Dict with 'isArray' and 'itemType' for lists
        - Dict with 'options' for Literal types
        - Dict with field names for Pydantic models
    """
    # Get function signature information
    func_signature = inspect.signature(func)
    params_info = {}

    def parse_type(type_obj) -> Union[str, Dict]:
        """Recursively parse type objects into JSON-serializable format."""
        origin = get_origin(type_obj)
        args = get_args(type_obj)

        # 1. Handle Optional or Union
        if origin is Union:
            # Filter out NoneType, get actual types
            actual_args = [arg for arg in args if arg is not type(None)]
            if len(actual_args) == 1:
                return parse_type(actual_args[0])
            # For complex unions, return a descriptive format
            return {"union": [parse_type(arg) for arg in actual_args]}

        # 2. Handle Literal (enum options)
        if origin is Literal:
            return {"options": list(args)}

        # 3. Handle Pydantic models (QueryFilters, etc.)
        if inspect.isclass(type_obj) and issubclass(type_obj, BaseModel):
            # Recursively get all fields in the model
            return {
                name: parse_type(field.annotation)
                for name, field in type_obj.model_fields.items()
            }

        # 4. Handle Callable types
        if origin is Callable or type_obj is Callable:
            return "Callable"

        # 5. Handle Any type
        if type_obj is Any:
            return "Any"

        # 6. Handle basic list types List[int], etc.
        if origin is list or origin is List:
            return {
                "isArray": True,
                "itemType": parse_type(args[0]) if args else "str",
            }

        # 7. Handle basic type mapping
        mapping = {int: "int", float: "float", bool: "bool", str: "str"}
        if type_obj in mapping:
            return mapping[type_obj]

        # 8. Handle class names with __name__
        if hasattr(type_obj, "__name__"):
            name = type_obj.__name__
            # Convert to lowercase for consistency
            return name.lower() if name != "Type" else name

        # 9. Handle generic types (e.g., Dict[str, int])
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))
            if origin_name in ("dict", "Dict"):
                return {
                    "type": "dict",
                    "keyType": parse_type(args[0]) if len(args) > 0 else "str",
                    "valueType": parse_type(args[1]) if len(args) > 1 else "Any",
                }
            return str(origin_name)

        return "str"

    for param_name, param in func_signature.parameters.items():
        if param_name in ["self", "cls"]:
            continue

        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
            # No type hint - use default based on default value if available
            if param.default is inspect.Parameter.empty:
                params_info[param_name] = "str"
            else:
                # Infer type from default value
                default_type = type(param.default).__name__.lower()
                params_info[param_name] = default_type
        else:
            params_info[param_name] = parse_type(param_type)

    return params_info


def aeval(expr: str, context: Union[Dict[str, Any], Any]) -> Any:
    """Evaluate an expression in a safe context.

    Args:
        expr: Expression to evaluate.
        context: Context object for evaluation (dict or object with as_dict() method).

    Returns:
        Evaluation result of any type.
    """
    if not isinstance(context, dict):
        context = context.as_dict()
    ee = Interpreter(context)
    result = ee(expr)
    # Return result as is, since asteval can return various types
    return result
