"""Helper functions"""

import gc
import glob
import importlib
import os
import subprocess
import sys
import threading
import time
from threading import Lock
from typing import Dict, Type

import jieba3
import nltk.stem.snowball
import numpy as np
import regex as re
import torch
from sentence_transformers import SentenceTransformer

jieba = jieba3.jieba3()


class AutoUnloadSentenceTransformer:
    """
    SentenceTransformer model automatic loading/unloading helper class.

    Automatically unload the model to release memory/GPU memory when idle
    for a specified timeout period (default 5 minutes).
    """

    def __init__(self, model_name_or_path: str, idle_timeout: int = 300) -> None:
        """
        Initialize the model manager.

        Args:
            model_name_or_path: Model name (e.g., all-MiniLM-L6-v2) or local model path,
                consistent with SentenceTransformer
            idle_timeout: Idle timeout in seconds, default 300 seconds (5 minutes)
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
            """Monitor model usage and unload when idle timeout is reached"""
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
        """Load the model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name_or_path)
        # Update last used time on each load/reuse
        self.last_used_time = time.time()

    def _unload_model(self) -> None:
        """Unload the model to release memory/GPU memory completely."""
        if self.model is not None:
            print(
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

            print(f"[Auto-unload] Model resources released")

    def encode(self, sentences, **kwargs) -> np.ndarray:
        """
        Wrap the native encode method for seamless usage.

        Args:
            sentences: Single sentence or list of sentences, consistent with native method
            kwargs: Other encode parameters such as convert_to_tensor, normalize_embeddings, etc.

        Returns:
            Sentence embeddings
        """
        with self.lock:
            self._load_model()  # Load if not present, reuse and update time
            # Call native model method
            embeddings = self.model.encode(sentences, **kwargs)
            # Update last used time
            self.last_used_time = time.time()
            return embeddings

    def encode_batch(self, sentences, **kwargs):
        """
        Wrap the native encode_batch method for efficient batch encoding.

        Args:
            sentences: 2D list of sentences
            kwargs: Other parameters

        Returns:
            Batch sentence embeddings
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
    """
    Stemming words
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
        """Get stemmer for language"""
        stemmer = nltk.stem.snowball.SnowballStemmer
        lang = WordStemmer._language_names.get(lang, lang)
        if lang not in WordStemmer._language_stemmers:
            if lang not in stemmer.languages:
                return WordStemmer._language_stemmers["en"]
            stemmer = stemmer(lang)
            WordStemmer._language_stemmers[lang] = stemmer
        return WordStemmer._language_stemmers[lang]

    def stem_tokens(self, lang, tokens):
        """
        Stem words

        Args:
            tokens: list of words

        Returns:
            stemmed words
        """
        tokens = [WordStemmer.get_stemmer(lang).stem(_) for _ in tokens]
        return tokens

    def stem_from_params(self, word, lang="en") -> dict[str, Any]:
        """
        Add stem() function for query

        Args:
            word: word to stem
            lang: language code

        Returns:
            dictionary with stemmed keywords
        """
        assert isinstance(lang, str) and isinstance(word, str), (
            f"Parameter type error for stem function: got {type(word)} and {type(lang)}"
        )
        return {"keywords": self.stem_tokens(lang, [word])[0]}


_pip_lock = Lock()


def safe_import(module_name: str, package_name: str = ""):
    """
    Import a module and install it if not installed.

    Args:
        module_name: Name of the module to import
        package_name: Name of the package to import the module from.
            Defaults to module name if not specified.

    Returns:
        Imported module object
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
    """
    Get context for given directory and parent class.

    Args:
        directory: Directory path relative to the working directory
        parent_class: Parent class of all defined classes to include
        sub_dirs: Subdirectories to search (optional)

    Returns:
        Dictionary in form of {"ClassName": Class}
    """

    def _prefix(sub_dir, name):
        """Prefixing module name"""
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
            print("Loading", module_name)
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
            print("Error while importing", module_name, ":", exception)

    return ctx
