import os
from functools import wraps
from flask import Response, request, send_file, stream_with_context, jsonify, abort
import traceback
from PyMongoWrapper import F
from models import Token, User
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Any, Callable, List, Dict, Iterable, Tuple, Union


def rest(login=True, cache=False, user_role=''):
    def do_rest(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                if (login and not logined()) or \
                    (user_role and not User.first((F.roles == user_role) & (F.username == logined()))):
                    raise Exception('Forbidden.')
                if request.json:
                    kwargs.update(**request.json)
                f = func(*args, **kwargs)
                if isinstance(f, (tuple, Response)): return f
                resp = jsonify({'result': f})
            except Exception as ex:
                resp = jsonify({'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)})
            
            resp.headers.add("Access-Control-Allow-Origin", "*")
            if cache:
                resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp
        return wrapped

    return do_rest


def logined():
    t = Token.check(request.headers.get('X-Authentication-Token', request.cookies.get('token')))
    if t:
        return t.user


def tmap(func, iterable, n=10):
    tpe = ThreadPoolExecutor(n)
    return tpe.map(func, iterable)


def serve_file(p: Union[str, IO], ext: str = '', file_size: int = 0) -> Response:
    """Serve static file or buffer

    Args:
        p (Union[str, IO]): file name or buffer
        ext (str, optional): extension name. Defaults to '' for auto.
        file_size (int, optional): file size. Defaults to 0 for auto.

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
        start = 0
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
            with f:
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
