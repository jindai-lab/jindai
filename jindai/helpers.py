"""辅助函数"""
import glob
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import datetime
from PIL import Image
import time
import traceback
from functools import wraps
from typing import IO, Dict, Type, Union

import numpy as np
import requests
import iso639
import werkzeug.wrappers.response
from bson import ObjectId
from flask import Response, jsonify, request, send_file, stream_with_context
from PyMongoWrapper.dbo import create_dbo_json_decoder, create_dbo_json_encoder

from .config import instance as config
from .models import Token
from .dbquery import parser
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


parser.functions['me'] = _me


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
                result = func(*args, **kwargs)
                if isinstance(result, (tuple, Response, werkzeug.wrappers.response.Response)):
                    return result

                resp = jsonify({'result': result})
            except Exception as ex:
                resp = jsonify(
                    {'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)})

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


def execute_query_expr(parsed, inputs):
    """Check according to parsed query expression

    :param parsed: parsed QueryExpr
    :type parsed: dict
    :param inputs: input object
    :type inputs: Union[Dict, List]
    """

    def _opr(k):
        oprname = {
            'lte': 'le',
            'gte': 'ge',
            '': 'eq'
        }.get(k, k)
        return '__' + oprname + '__'

    def _getattr(obj, k, default=None):
        if '.' in k:
            for key_seg in k.split('.'):
                obj = _getattr(obj, key_seg, default)
            return obj

        if isinstance(obj, dict):
            return obj.get(k, default)

        if isinstance(obj, list) and RE_DIGITS.match(k):
            return obj[int(k)] if 0 <= int(k) < len(obj) else default

        return getattr(obj, k, default)

    def _hasattr(obj, k):
        if '.' in k:
            for key_seg in k.split('.')[:-1]:
                obj = _getattr(obj, key_seg)
            k = k.split('.')[-1]

        if isinstance(obj, dict):
            return k in obj

        if isinstance(obj, list) and RE_DIGITS.match(k):
            return 0 <= int(k) < len(obj)

        return hasattr(obj, k)

    def _test_inputs(inputs, val, k='eq'):
        oprname = _opr(k)

        if oprname == '__in__':
            return inputs in val

        if oprname == '__size__':
            return len(inputs) == val

        if isinstance(inputs, list):
            arr_result = False
            for input_val in inputs:
                arr_result = arr_result or _getattr(input_val, oprname)(val)
                if arr_result:
                    break
        else:
            arr_result = _getattr(inputs, oprname)(val)

        return arr_result is True

    result = True
    assert isinstance(
        parsed, dict), 'QueryExpr should be parsed first and well-formed.'

    for key, val in parsed.items():
        if key.startswith('$'):
            if key == '$and':
                arr_result = True
                for element in val:
                    arr_result = arr_result and execute_query_expr(
                        element, inputs)
            elif key == '$or':
                arr_result = False
                for element in val:
                    arr_result = arr_result or execute_query_expr(
                        element, inputs)
                    if arr_result:
                        break
            elif key == '$not':
                arr_result = not execute_query_expr(val, inputs)
            elif key == '$regex':
                arr_result = re.search(val, inputs) is not None
            elif key == '$options':
                continue
            else:
                arr_result = _test_inputs(inputs, val, key[1:])

            result = result and arr_result

        elif not isinstance(val, dict) or not [1 for v_ in val if v_.startswith('$')]:

            result = result and _test_inputs(_getattr(inputs, key), val)
        else:
            result = result and execute_query_expr(val, _getattr(
                inputs, key) if _hasattr(inputs, key) else None)

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
