"""Helper functions"""
import datetime
import glob
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from functools import wraps
from threading import Lock
from typing import IO, Dict, List, Type
from uuid import UUID

import iso639
import numpy as np
import requests
import werkzeug.wrappers.response
from flask import Flask, Response, abort, jsonify, request
from flask.json.provider import JSONProvider as JSONProvideBase
from PIL.Image import Image
from werkzeug.exceptions import HTTPException

from .config import instance as config
from .models import UserInfo
from .storage import instance as storage


class JSONProvider(JSONProvideBase):

    def dumps(self, obj, **kwargs):
        return json.dumps(obj, **kwargs, cls=JSONEncoder)
    
    def loads(self, sbuf, **kwargs):
        return json.loads(sbuf, **kwargs)


class WordStemmer:
    """
    Stemming words
    """

    _language_stemmers = {}

    @staticmethod
    def get_stemmer(lang):
        """Get stemmer for language"""
        safe_import("nltk")
        stemmer = safe_import("nltk.stem.snowball").SnowballStemmer
        if lang not in WordStemmer._language_stemmers:
            lang = language_iso639.get(lang, lang).lower()
            if lang not in stemmer.languages:
                return WordStemmer.get_stemmer("en")
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


def rest(login=True, cache=False, role="", mapping=None):
    """
    Decorator for REST API endpoints. Decorates a function to be used as a WSGI application and returns a JSON response.

    @param login - If True ( default ) login the user with the given role
    @param cache - If True cache the response in the cache directory. This is useful if you want to make sure a user is logged in before accessing the endpoint.
    @param role - The role to check login status against. Defaults to''
    @param mapping - A dictionary of key / value pairs to be used as keyword arguments.

    @return A JSON response to the endpoint or an error if something went wrong
    """
    # Set mapping to a new mapping.
    if mapping is None:
        mapping = {}

    def do_rest(func):
        """
        Decorator to wrap REST calls. Decorated function will check login role and return 403 if user is not logged in.

        @param func - function to be wrapped. This is the function that will be called.

        @return a response to the function or an exception if something went wrong
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            """
            Wraps the function and returns a response. If an exception is raised it is logged and the response is returned to the client.


            @return The response to the client or a JSON object with error
            """
            try:
                erred = False
                # If login is not logged in return 403.
                if login and not logined(role):
                    return f"Forbidden. Client: {request.remote_addr}", 403
                if request.content_type == "application/json" and request.json:
                    for key, val in request.json.items():
                        kwargs[mapping.get(key, key)] = val
                elif request.method == 'GET':
                    for key, val in request.args.items():
                        kwargs[mapping.get(key, key)] = val

                request.lang = request.headers.get("X-Preferred-Language", "")
                result = func(*args, **kwargs)
                if isinstance(
                    result, (tuple, Response, werkzeug.wrappers.response.Response)
                ):
                    return result

                resp = jsonify(result)
            except HTTPException as hx:
                raise hx
            except Exception as ex:
                erred = True
                resp = jsonify(
                    {
                        "__exception__": type(ex).__name__ + ": " + str(ex),
                        "__tracestack__": traceback.format_tb(ex.__traceback__),
                    }
                )

            resp.headers.add("Access-Control-Allow-Origin", "*")
            # Add Cache Control header to the response.
            if cache and not erred:
                resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp

        return wrapped

    return do_rest


def logined(role="", detailed=False):
    """
    Check if user logged in and return user object.
    This is used to handle requests that have a token in their request headers.

    @param role - Role to check. Check only if is logined if empty.

    @return User object or None if not logged in or token
    """
    token = request.headers.get(
            "X-Authentication-Token",
            request.cookies.get("token", request.args.get("_token", "")),
        )

    # Returns the user who owns the token.
    with SessionLocal() as session:
        user = session.query(UserInfo).filter(UserInfo.token == token).first()
    if user:
        if role == "" or role in user.roles:
            return user.username if not detailed else user
    
    inet_addr = request.headers.get("X-Real-IP") or request.remote_addr

    # Check if IP address exists in automatic login mapping.
    if inet_addr in config.allowed_ips:
        return config.allowed_ips[inet_addr]

    return None


def serve_proxy(server, path):
    """Serve from remote server

    :param server: server host
    :type server: str
    :param path: path
    :type path: str
    :return: response from remote server
    :rtype: Response
    """
    resp = requests.get(f"http://{server}/{path}", timeout=1000)
    return Response(resp.content, headers=dict(resp.headers))


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


JSONEncoderCls = json.JSONEncoder


class JSONEncoder(json.JSONEncoder):
    """JSONEncoder for api use"""

    def __init__(self, **kwargs):
        """Initialize the JSON Encoder"""
        kwargs["ensure_ascii"] = False
        super().__init__(**kwargs)

    def default(self, o):
        """Encode the object o

        :param o: the object
        :type o: Any
        :return: str or JSON-compatible objects
        :rtype: Any
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int32):
            return o.tolist()
        if isinstance(o, Image):
            return str(o)
        if isinstance(o, datetime.datetime):
            return o.isoformat() + "Z"
        if isinstance(o, UUID):
            return str(o)
        
        return JSONEncoderCls.default(self, o)


# ISO639 language codes
language_iso639 = {
    lang.part1: lang.name for lang in iso639.ALL_LANGUAGES if lang.part1 and lang.part1 != "zh"
}
language_iso639.update(zhs="Chinese Simplified", zht="Chinese Traditional")
