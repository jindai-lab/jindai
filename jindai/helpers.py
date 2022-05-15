"""Helper functions"""
import datetime
import glob
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from functools import wraps
from typing import IO, Dict, Type, Union

import iso639
import numpy as np
import requests
import werkzeug.wrappers.response
from bson import ObjectId
from flask import Response, jsonify, request, send_file, stream_with_context
from PIL import Image
from PyMongoWrapper import MongoOperand
from PyMongoWrapper.dbo import create_dbo_json_decoder, create_dbo_json_encoder

from .config import instance as config
from .dbquery import parser
from .models import Token
from .storage import safe_open


def _me(param=''):
    """Add me() Function for query

    :param param: a string in query, defaults to ''
    :type param: str, optional
    :return: A string in form of "{param:}{logined user}"
    :rtype: _type_
    """
    param = str(param)
    if param:
        param += ':'
    return param + logined()


class WordStemmer:
    """
    Stemming words
    """

    _language_stemmers = {}

    @staticmethod
    def get_stemmer(lang):
        """Get stemmer for language"""
        safe_import('nltk')
        stemmer = safe_import('nltk.stem.snowball').SnowballStemmer
        if lang not in WordStemmer._language_stemmers:
            lang = language_iso639.get(lang, lang).lower()
            if lang not in stemmer.languages:
                return WordStemmer.get_stemmer('en')
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
        tokens = [WordStemmer.get_stemmer(
            lang).stem(_) for _ in tokens]
        return tokens

    def stem_from_params(self, param=''):
        """Add stem() function for query

        :param param: '<word:lang>', or [word, lang], or {'$and': [{'keywords': word}, {'keywords': lang}]}
        :type param: str, optional
        :return: stemmed word
        :rtype: str
        """
        if isinstance(param, MongoOperand):
            param = param()
        word, lang = str(param), 'en'
        if isinstance(param, (tuple, list)) and len(param) == 2:
            word, lang = param
        elif isinstance(param, dict) and '$and' in param and len(param['$and']) == 2:
            word, lang = param['$and']
            if isinstance(word, dict) and isinstance(lang, dict) and len(word) == 1 and len(lang) == 1:
                (word,), (lang,) = word.values(), lang.values()
            else:
                word, lang = '', ''
        elif isinstance(param, dict) and len(param) == 2 and 'lang' in param and 'keywords' in param:
            word, lang = param['keywords'], param['lang']
        assert isinstance(lang, str) and isinstance(
            word, str), 'Parameter type error for stem function'
        if len(word) == 2 and len(word) < len(lang):
            lang, word = word, lang
        return {'keywords': self.stem_tokens(lang, [word])[0]}


parser.functions['me'] = _me
parser.functions['stem'] = WordStemmer().stem_from_params


def safe_import(module_name, package_name=''):
    """Safe import module

    :param module_name: module name
    :type module_name: str
    :param package_name: package name for pip, defaults to ''
    :type package_name: str, optional
    :return: the imported module
    :rtype: Module
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.call([sys.executable, '-m', 'pip',
                        'install', package_name or module_name])
    return importlib.import_module(module_name)


def rest(login=True, cache=False, role='', mapping=None):
    """Rest API

    :param login: require logged in, defaults to True
    :type login: bool, optional
    :param cache: use cache, defaults to False
    :type cache: bool, optional
    :param role: check user role, defaults to ''
    :type role: str, optional
    :param mapping: mapping json request body keys, defaults to None
    :type mapping: dict, optional
    :return: Flask response
    :rtype: Response
    """
    if mapping is None:
        mapping = {}

    def do_rest(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                if login and not logined(role):
                    raise Exception(
                        f'Forbidden. Client: {request.remote_addr}')
                if request.json:
                    for key, val in request.json.items():
                        kwargs[mapping.get(key, key)] = val
                request.lang = request.headers.get('X-Preferred-Language', '')
                result = func(*args, **kwargs)
                if isinstance(result, (tuple, Response, werkzeug.wrappers.response.Response)):
                    return result

                resp = jsonify({'result': result})
            except Exception as ex:
                resp = jsonify(
                    {'__exception__': str(ex), '__tracestack__': traceback.format_tb(ex.__traceback__)})

            resp.headers.add("Access-Control-Allow-Origin", "*")
            if cache:
                resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp
        return wrapped

    return do_rest


def logined(role=''):
    """Check user login and role

    :param role: str, defaults to ''
    :type role: str, optional
    :return: User name if passed, None if not
    :rtype: Union[str, None]
    """
    token = Token.check(request.headers.get(
        'X-Authentication-Token', request.cookies.get('token')))

    if token and (not role or role in token.roles):
        return token.user

    if request.remote_addr in config.allowed_ips:
        return config.allowed_ips[request.remote_addr]

    return None


def serve_file(path_or_io: Union[str, IO], ext: str = '', file_size: int = 0) -> Response:
    """Serve static file or buffer

    Args:
        p (Union[str, IO]): file name or buffer
        ext (str, optional): extension name
        file_size (int, optional): file size

    Returns:
        Response: a flask response object
    """
    if isinstance(path_or_io, str):
        input_file = open(path_or_io, 'rb')
        ext = path_or_io.rsplit('.', 1)[-1]
        file_size = os.stat(path_or_io).st_size
    else:
        input_file = path_or_io

    mimetype = {
        'html': 'text/html',
                'htm': 'text/html',
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'gif': 'image/gif',
                'json': 'application/json',
                'css': 'text/css',
                'js': 'application/javascript',
                'mp4': 'video/mp4'
    }.get(ext, 'text/plain')

    start, length = 0, 1 << 20
    range_header = request.headers.get('Range')
    if file_size and file_size > 10 << 20:
        if range_header:
            # example: 0-1000 or 1250-
            matched_nums = re.search('([0-9]+)-([0-9]*)', range_header)
            num_groups = matched_nums.groups()
            byte1, byte2 = 0, None
            if num_groups[0]:
                byte1 = int(num_groups[0])
            if num_groups[1]:
                byte2 = int(num_groups[1])
            if byte1 < file_size:
                start = byte1
            if byte2:
                length = byte2 + 1 - byte1
            else:
                length = file_size - start
        else:
            length = file_size

        def _generate_chunks():
            coming_length = length
            input_file.seek(start)
            while coming_length > 0:
                chunk = input_file.read(min(coming_length, 1 << 20))
                coming_length -= len(chunk)
                yield chunk

        resp = Response(stream_with_context(_generate_chunks()), 206,
                        content_type=mimetype, direct_passthrough=True)
        resp.headers.add(
            'Content-Range', f'bytes {start}-{start+length-1}/{file_size}')
        return resp

    return send_file(input_file, mimetype=mimetype, conditional=True)


def serve_proxy(server, path):
    """Serve from remote server

    :param server: server host
    :type server: str
    :param path: path
    :type path: str
    :return: response from remote server
    :rtype: Response
    """
    resp = requests.get(f'http://{server}/{path}',
                        headers={'Host': 'localhost:8080'})
    return Response(resp.content, headers=dict(resp.headers))


RE_DIGITS = re.compile(r'[\+\-]?\d+')


def execute_query_expr(parsed, test_val):
    """Check according to parsed query expression

    :param parsed: parsed QueryExpr
    :type parsed: dict
    :param test_val: input object
    :type test_val: Union[Dict, List]
    """

    def _opr(k):
        oprname = {
            'lte': 'le',
            'gte': 'ge',
            '': 'eq'
        }.get(k, k)
        return '__' + oprname + '__'

    def _getattr(obj, key, default=None):
        if obj is None:
            return obj

        if '.' in key:
            for key_seg in key.split('.'):
                obj = _getattr(obj, key_seg, default)
            return obj

        if isinstance(obj, dict):
            return obj.get(key, default)

        if isinstance(obj, list) and RE_DIGITS.match(key):
            return obj[int(key)] if 0 <= int(key) < len(obj) else default

        return getattr(obj, key, default)

    def _test_inputs(inputs, val, relation='eq'):
        oprname = _opr(relation)

        if oprname == '__in__':
            return inputs in val

        if oprname == '__size__':
            return len(inputs) == val

        if oprname == '__first__':
            for i in inputs:
                return i
            return None

        if oprname == '__last__':
            i = None
            for i in inputs:
                pass
            return i

        if isinstance(inputs, list):
            arr_result = False
            for input_val in inputs:
                arr_result = arr_result or _getfunc(input_val, oprname)(val)
                if arr_result:
                    break
        else:
            arr_result = _getfunc(inputs, oprname)(val)

        return arr_result

    def _getfunc(obj, func_name):
        func = _getattr(obj, func_name)
        if func:
            return func
        else:
            return lambda: None

    def _append_result(res):
        if result is None:
            return res
        return result and res

    result = None
    assert isinstance(
        parsed, dict), 'QueryExpr should be parsed first and well-formed.'

    for key, val in parsed.items():
        if key.startswith('$'):
            if key == '$and':
                arr_result = True
                for element in val:
                    arr_result = arr_result and execute_query_expr(
                        element, test_val)
            elif key == '$or':
                arr_result = False
                for element in val:
                    arr_result = arr_result or execute_query_expr(
                        element, test_val)
                    if arr_result:
                        break
            elif key == '$not':
                arr_result = not execute_query_expr(val, test_val)
            elif key == '$regex':
                arr_result = re.search(val, test_val) is not None
            elif key == '$options':
                continue
            else:
                arr_result = _test_inputs(test_val, val, key[1:])

            result = _append_result(arr_result)

        elif not isinstance(val, dict) or not [1 for v_ in val if v_.startswith('$')]:
            result = _append_result(_test_inputs(_getattr(test_val, key), val))
        else:
            result = _append_result(execute_query_expr(val, _getattr(
                test_val, key)))
        if result is False:
            return result

    return result


def get_context(directory: str, parent_class: Type) -> Dict:
    """Get context for given directory and parent class

    :param directory: directory path relative to the working directory
    :type directory: str
    :param parent_class: parent class of all defined classes to include
    :type parent_class: Type
    :return: a directory in form of {"ClassName": Class}
    :rtype: Dict
    """

    def _prefix(name):
        """Prefixing module name"""
        return directory.replace(os.sep, '.') + '.' + name

    modules = [
        _prefix(os.path.basename(f).split('.')[0])
        for f in glob.glob(os.path.join(directory, "*.py"))
    ] + [
        _prefix(f.split(os.path.sep)[-2])
        for f in glob.glob(os.path.join(directory, '*/__init__.py'))
    ]
    ctx = {}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            for k in module.__dict__:
                if k != parent_class.__name__ and not k.startswith('_') and \
                        isinstance(module.__dict__[k], type) and \
                        issubclass(module.__dict__[k], parent_class):
                    ctx[k] = module.__dict__[k]
        except Exception as exception:
            print('Error while importing', module_name, ':', exception)

    return ctx


JSONEncoderCls = create_dbo_json_encoder(json.JSONEncoder)


class JSONEncoder(json.JSONEncoder):
    """JSONEncoder for api use
    """

    def __init__(self, **kwargs):
        """Initialize the JSON Encoder
        """
        kwargs['ensure_ascii'] = False
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
        if isinstance(o, Image.Image):
            return str(o)
        if isinstance(o, datetime.datetime):
            return o.isoformat() + "Z"

        return JSONEncoderCls.default(self, o)


"""JSONDecoder for api use"""
JSONDecoder = create_dbo_json_decoder(json.JSONDecoder)


"""ISO639 language codes"""
language_iso639 = {
    lang.pt1: lang.name for lang in iso639.iter_langs() if lang.pt1 and lang.pt1 != 'zh'
}
language_iso639.update(chs='Chinese Simplified', cht='Chinese Traditional')
