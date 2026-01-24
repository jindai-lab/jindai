"""Helper functions"""

import gc
import glob
import importlib
import os
import regex as re
import subprocess
import sys
import threading
import time
from threading import Lock
from typing import Dict, Type

import jieba3
import nltk.stem.snowball
import torch
from sentence_transformers import SentenceTransformer

jieba = jieba3.jieba3()


class AutoUnloadSentenceTransformer:
    """
    SentenceTransformer 模型自动加载/卸载辅助类
    核心机制：超过指定空闲时间（默认5分钟）无调用，自动卸载模型释放内存/显存
    """

    def __init__(self, model_name_or_path: str, idle_timeout: int = 300):
        """
        初始化模型管理器
        :param model_name_or_path: 模型名称(如all-MiniLM-L6-v2)或本地模型路径，与SentenceTransformer一致
        :param idle_timeout: 空闲超时时间，单位秒，默认300秒=5分钟
        """
        self.model_name_or_path = model_name_or_path
        self.idle_timeout = idle_timeout  # 空闲超时阈值
        self.model = None  # 核心模型对象，初始为空（惰性加载）
        self.last_used_time = time.time()  # 最后一次使用时间戳
        self.lock = threading.Lock()  # 线程安全锁
        self._monitor_thread = None  # 监控线程
        self._stop_monitor = False  # 监控线程停止标识

        # 启动空闲监控线程（后台守护线程，随主线程退出）
        self._start_monitor()

    def _start_monitor(self):
        """启动模型空闲监控后台线程"""

        def monitor_loop():
            while not self._stop_monitor:
                time.sleep(10)  # 每10秒检测一次，降低CPU占用
                with self.lock:
                    # 满足2个条件触发卸载：1.模型已加载 2.当前时间 - 最后使用时间 > 超时阈值
                    if (
                        self.model is not None
                        and (time.time() - self.last_used_time) > self.idle_timeout
                    ):
                        self._unload_model()

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _load_model(self):
        """加载模型"""
        if self.model is None:
            self.model = SentenceTransformer(
                self.model_name_or_path, local_files_only=True
            )
        # 每次加载/复用，都更新最后使用时间
        self.last_used_time = time.time()

    def _unload_model(self):
        """卸载模型，彻底释放内存+显存"""
        if self.model is not None:
            print(f"[自动卸载] 模型空闲超{self.idle_timeout}秒，开始释放资源...")
            # 核心：删除模型对象
            del self.model
            self.model = None

            # 强制触发Python垃圾回收，释放内存
            gc.collect()

            # 关键步骤：清空torch显存缓存（GPU环境必须，否则显存不释放）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            print(f"[自动卸载] 模型资源释放完成")

    def encode(self, sentences, **kwargs):
        """
        封装原生encode方法，无缝调用
        :param sentences: 单句/句子列表，与原生一致
        :param kwargs: encode的其他参数，如convert_to_tensor、normalize_embeddings等
        :return: 句子向量，与原生一致
        """
        with self.lock:
            self._load_model()  # 无则加载，有则复用并更新时间
            # 调用原生模型方法
            embeddings = self.model.encode(sentences, **kwargs)
            # 更新最后使用时间
            self.last_used_time = time.time()
            return embeddings

    def encode_batch(self, sentences, **kwargs):
        """
        封装原生encode_batch方法，批量编码更高效
        :param sentences: 二维句子列表，与原生一致
        :param kwargs: 其他参数
        :return: 批量句子向量，与原生一致
        """
        with self.lock:
            self._load_model()
            embeddings = self.model.encode_batch(sentences, **kwargs)
            self.last_used_time = time.time()
            return embeddings

    def __del__(self):
        """对象销毁时，主动卸载模型+停止监控线程"""
        self._stop_monitor = True
        with self.lock:
            self._unload_model()


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
    def get_stemmer(lang):
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
        """Stem words

        :param tokens: list of words
        :type tokens: list
        :return: stemmed words
        :rtype: list
        """
        tokens = [WordStemmer.get_stemmer(lang).stem(_) for _ in tokens]
        return tokens

    def stem_from_params(self, word, lang="en"):
        """Add stem() function for query"""
        assert isinstance(lang, str) and isinstance(
            word, str
        ), f"Parameter type error for stem function: got {type(word)} and {type(lang)}"
        return {"keywords": self.stem_tokens(lang, [word])[0]}


_pip_lock = Lock()


def safe_import(module_name, package_name=""):
    """
    Import a module and if it's not installed install it.

    @param module_name - The name of the module to import.
    @param package_name - The name of the package to import the module from. Defaults to the module name if not specified.

    @return The imported module object.
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
    """Get context for given directory and parent class

    :param directory: directory path relative to the working directory
    :type directory: str
    :param parent_class: parent class of all defined classes to include
    :type parent_class: Type
    :return: a directory in form of {"ClassName": Class}
    :rtype: Dict
    """

    def _prefix(sub_dir, name):
        """Prefixing module name"""
        dirpath = directory
        if sub_dir != ".":
            dirpath += os.sep + sub_dir
        return dirpath.replace(os.sep, ".") + "." + name

    if len(sub_dirs) == 0:
        sub_dirs = ["."]
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
