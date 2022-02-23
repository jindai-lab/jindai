import importlib
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import IO, Union

import werkzeug.wrappers.response
from flask import Response, jsonify, request, send_file, stream_with_context
from PyMongoWrapper import F

import config
from models import MongoJSONEncoder, Paragraph, Token, User, get_context, parser
from plugin import Plugin


def _me(p=''):
    p = str(p)
    if p: 
        p += ':'
    return p + logined()


parser.functions['me'] = _me


class PluginManager:

    def __init__(self, app) -> None:
        self.plugins = []
        self.pages = {}
        self.callbacks = defaultdict(list)

        @app.route('/api/plugins/styles.css')
        def plugins_style():
            """Returns css from all enabled plugins

            Returns:
                Response: css document
            """
            css = '\n'.join([p.run_callback('css')
                            for p in self.callbacks['css']])
            return Response(css, mimetype='text/css')

        @app.route('/api/plugins/pages', methods=["GET", "POST"])
        @rest()
        def plugin_pages():
            """Returns names for special pages in every plugins
            """
            return list(self.pages.keys())

        # load plugins

        import plugins as _plugins
        pls = getattr(config, 'plugins', ['*'])
        if pls == ['*']:
            pls = list(get_context('plugins', Plugin).values())

        for pl in pls:
            if isinstance(pl, tuple) and len(pl) == 2:
                pl, kwargs = pl
            else:
                kwargs = {}

            if isinstance(pl, str):
                if '.' in pl:
                    plpkg, plname = pl.rsplit('.', 1)
                    pkg = __import__('plugins.' + plpkg)
                    for seg in pl.split('.'):
                        pkg = getattr(pkg, seg)
                    pl = pkg
                else:
                    pl = getattr(_plugins, pl)

            try:
                pl = pl(app, **kwargs)

                for name in pl.get_pages():
                    self.pages[name] = pl

                for name in pl.get_callbacks():
                    self.callbacks[name].append(pl)

                self.plugins.append(pl)
                print('Registered plugin:', type(pl).__name__)
            except Exception as ex:
                print('Error while registering plugin:', pl, ex)
                continue


def safe_import(module_name, package_name=''):
    try:
        importlib.import_module(module_name)
    except ImportError:
        import subprocess
        subprocess.call([sys.executable, '-m', 'pip', 'install', package_name or module_name])
    return importlib.import_module(module_name)


def rest(login=True, cache=False, user_role=''):
    def do_rest(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                if (login and not logined()) or \
                    (user_role and not User.first((F.roles == user_role) & (F.username == logined()))):
                    raise Exception(f'Forbidden. Client: {request.remote_addr}')
                if request.json:
                    kwargs.update(**request.json)
                f = func(*args, **kwargs)
                if isinstance(f, (tuple, Response, werkzeug.wrappers.response.Response)): return f
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
    if request.remote_addr in config.allowed_ips:
        return request.remote_addr


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
                print(l)
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
            yield from task._task.fetch_log()
            time.sleep(0.1)

        yield from task._task.fetch_log()
        yield 'returned: ' + str(type(task._task.returned)) + '\n'

        yield 'finished.\n'

    return Response(stream_with_context(generate()), status=200,
                    mimetype="text/plain",
                    content_type="text/event-stream"
                    )
