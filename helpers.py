import importlib
import os
import re
import sys
import pickle
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import IO, Union

import requests
import werkzeug.wrappers.response
from flask import Response, jsonify, request, send_file, stream_with_context
from PyMongoWrapper import F

import config
from models import (MongoJSONEncoder, Paragraph, Token, User, get_context,
                    parser)
from storage import safe_open


def _me(p=''):
    p = str(p)
    if p:
        p += ':'
    return p + logined()


parser.functions['me'] = _me


def safe_import(module_name, package_name=''):
    try:
        importlib.import_module(module_name)
    except ImportError:
        import subprocess
        subprocess.call([sys.executable, '-m', 'pip',
                        'install', package_name or module_name])
    return importlib.import_module(module_name)


def rest(login=True, cache=False, role=''):
    def do_rest(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                if login and not logined(role):
                    raise Exception(
                        f'Forbidden. Client: {request.remote_addr}')
                if request.json:
                    kwargs.update(**request.json)
                f = func(*args, **kwargs)
                if isinstance(f, (tuple, Response, werkzeug.wrappers.response.Response)):
                    return f
                resp = jsonify({'result': f})
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
    t = Token.check(request.headers.get(
        'X-Authentication-Token', request.cookies.get('token')))
    if t and (not role or role in t.roles):
        return t.user
    if request.remote_addr in config.allowed_ips:
        return config.allowed_ips[request.remote_addr]


def tmap(func, iterable, n=10):
    tpe = ThreadPoolExecutor(n)
    return tpe.map(func, iterable)


def serve_file(p: Union[str, IO], ext: str = '', file_size: int = 0) -> Response:
    """Serve static file or buffer

    Args:
        p (Union[str, IO]): file name or buffer
        ext (str, optional): extension name
        file_size (int, optional): file size

    Returns:
        Response: a flask response object
    """
    if isinstance(p, str):
        f = open(p, 'rb')
        ext = p.rsplit('.', 1)[-1]
        file_size = os.stat(p).st_size
    else:
        f = p

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
            m = re.search('([0-9]+)-([0-9]*)', range_header)
            g = m.groups()
            byte1, byte2 = 0, None
            if g[0]:
                byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])
            if byte1 < file_size:
                start = byte1
            if byte2:
                length = byte2 + 1 - byte1
            else:
                length = file_size - start
        else:
            length = file_size

        def _generate_chunks():
            l = length
            f.seek(start)
            while l > 0:
                chunk = f.read(min(l, 1 << 20))
                l -= len(chunk)
                yield chunk

        rv = Response(stream_with_context(_generate_chunks()), 206,
                      content_type=mimetype, direct_passthrough=True)
        rv.headers.add(
            'Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
        return rv
    else:
        return send_file(f, mimetype=mimetype, conditional=True)


def serve_proxy(server, path):
    resp = requests.get(f'http://{server}/{path}',
                        headers={'Host': 'localhost:8080'})
    return Response(resp.content, headers=dict(resp.headers))


def logs_view(task):
    """Provide log stream of given TaskDBO
    """

    def generate():
        """Generate log text from task object of the TaskDBO

        Yields:
            str: log text
        """
        while task._task is None:
            time.sleep(1)

        while task._task.alive:
            yield from task._task.log_fetch()
            time.sleep(0.1)

        yield from task._task.log_fetch()
        yield 'returned: ' + str(type(task._task.returned)) + '\n'

        yield 'finished.\n'

    return Response(stream_with_context(generate()), status=200,
                    mimetype="text/plain",
                    content_type="text/event-stream"
                    )


RE_DIGITS = re.compile(r'[\+\-]?\d+')


def execute_query_expr(parsed, inputs):

    def _opr(k):
        oprname = {
            'lte': 'le',
            'gte': 'ge',
            '': 'eq'
        }.get(k, k)
        return '__' + oprname + '__'

    def _getattr(o, k, default=None):
        if '.' in k:
            for k_ in k.split('.'):
                o = _getattr(o, k_, default)
            return o

        if isinstance(o, dict):
            return o.get(k, default)
        elif isinstance(o, list) and RE_DIGITS.match(k):
            return o[int(k)] if 0 <= int(k) < len(o) else default
        else:
            return getattr(o, k, default)

    def _hasattr(o, k):
        if '.' in k:
            for k_ in k.split('.')[:-1]:
                o = _getattr(o, k_)
            k = k.split('.')[-1]

        if isinstance(o, dict):
            return k in o
        elif isinstance(o, list) and RE_DIGITS.match(k):
            return 0 <= int(k) < len(o)
        else:
            return hasattr(o, k)

    def _test_inputs(inputs, v, k='eq'):
        oprname = _opr(k)
        if oprname == '__in__':
            return inputs in v
        elif oprname == '__size__':
            return len(inputs) == v

        if isinstance(inputs, list):
            rr = False
            for v_ in inputs:
                rr = rr or _getattr(v_, oprname)(v)
                if rr:
                    break
        else:
            rr = _getattr(inputs, oprname)(v)
        return rr is True
    r = True
    assert isinstance(
        parsed, dict), 'QueryExpr should be parsed first and well-formed.'
    for k, v in parsed.items():
        if k.startswith('$'):
            if k == '$and':
                rr = True
                for v_ in v:
                    rr = rr and execute_query_expr(v_, inputs)
            elif k == '$or':
                rr = False
                for v_ in v:
                    rr = rr or execute_query_expr(v_, inputs)
                    if rr:
                        break
            elif k == '$not':
                rr = not execute_query_expr(v, inputs)
            elif k == '$regex':
                rr = re.search(v, inputs) is not None
            elif k == '$options':
                continue
            else:
                rr = _test_inputs(inputs, v, k[1:])
            r = r and rr
        elif not isinstance(v, dict) or not [1 for v_ in v if v_.startswith('$')]:
            r = r and _test_inputs(_getattr(inputs, k), v)
        else:
            r = r and execute_query_expr(v, _getattr(
                inputs, k) if _hasattr(inputs, k) else None)
    return r


with safe_open('models_data/language_iso639', 'rb') as flang:
    language_iso639 = pickle.load(flang)
