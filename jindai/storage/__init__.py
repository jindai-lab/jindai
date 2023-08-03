"""File/Network storage access"""
import json
import os
import re
import sys
import urllib
import zipfile
from io import BytesIO
from queue import deque
from typing import IO, Union

import requests
from waitress import serve
from flask import Flask, Response, jsonify, request, stream_with_context
from jindai.config import instance as config

from .osfile import OSFileSystemManager
from .storage import StorageManager
from .webmanager import DataSchemeManager, WebManager


class StorageProxyManager(StorageManager):
    """Access thorugh storage server"""

    def __init__(self, proxy) -> None:
        super().__init__()
        self._webm = WebManager(seekable=True)
        self._base = proxy.rstrip('/')

    def _proxied_url(self, path):
        parsed = urllib.parse.urlparse(path)

        target = '/'.join([_ for _ in (
            self._base,
            parsed.scheme or 'file',
            parsed.netloc,
            parsed.path.lstrip('/')) if _])

        if parsed.fragment:
            target += '__hash/' + parsed.fragment

        return target

    def _get_json(self, path, action, **params):
        params['action'] = action
        url = self._proxied_url(path) + '?' + urllib.parse.urlencode(params)
        try:
            text = requests.get(url).content
            return json.loads(text)
        except:
            return

    def __getattribute__(self, __name: str):
        if __name in ('statdir', 'listdir', 'exists', 'stat', 'mkdir', 'expand_path', 'truncate_path'):
            return lambda p: self._get_json(p, __name)
        return super().__getattribute__(__name)

    def write(self, path, data, **params):
        return self._webm.write(self._proxied_url(path), data=data, **params)

    def readbuf(self, path, **params):
        return self._webm.readbuf(self._proxied_url(path), **params)

    def read(self, path, **params):
        return self._webm.read(self._proxied_url(path), **params)

    def walk(self, start_path, name_pattern=''):
        yield from self._get_json(start_path, 'walk', name_pattern=name_pattern) or []

    def search(self, path, name_pattern):
        yield from self._get_json(path, 'search', name_pattern=name_pattern) or []
        
    def move(self, path, dst):
        return self._get_json(path, 'move', dst=dst) or []
        

class Storage:

    def __init__(self) -> None:
        """Initialize storage
        """

        self._schema = {
            'http': WebManager,
            'https': WebManager,
            'data': DataSchemeManager,
            'file': OSFileSystemManager,
        }

        self.storage = config.storage
        if not self.storage:
            self.storage = {}

        if isinstance(self.storage, str):
            self.storage = {
                k: [self.storage] for k in self._schema if k not in ('http', 'https', 'data')
            }

        for key, val in self.storage.items():
            if key == 'default':
                continue
            if isinstance(val, str):
                self.storage[key] = [val]
        
        self._fragment_handlers = {}

    def _get_managers(self, path) -> StorageManager:
        """Get storage manager according to scheme"""
        scheme = path
        if '://' in path:
            scheme = path.split('://', 1)[0]
        if not scheme:
            scheme = 'file'

        if scheme in self.storage:
            if scheme in self._schema:
                yield self._schema[scheme](self.storage[scheme])
            for server in self.storage[scheme]:
                if '://' in server:
                    yield StorageProxyManager(server)
        elif scheme in self._schema:
            yield self._schema[scheme]()

    def register_fragment_handler(self, frag_name, func):
        """Register fragment handler

        Args:
            frag_name (str): fragment name
            func (function): handler function
        """
        self._fragment_handlers[frag_name] = func

    def register_scheme(self, scheme: str, mgr):
        """Register scheme

        Args:
            scheme (str): scheme name
            mgr (any func/type that accepts one argument and yieldsStorageManager): manager instance
        """
        if isinstance(mgr, StorageManager):
            mgr = lambda *_: mgr
        self._schema[scheme.lower()] = mgr
        
    def open(self, path: str, mode='rb', **params):
        """Open the path for read/write

        :param path: file system path or URLs.
                Relative and absolute paths are supported.
                Paths are matched in the following priority:
                    - URLs starting with http:// or https:// 
                    may use `proxies`, `referer`, and `headers` in `params`
                    - Data URLs, i.e. data:...
                    readonly, `mode` should be 'rb'
                    - <custom scheme>://<item id>
                    `mode` should be 'rb' or 'wb'
                    - <file path>#zip/<zip path>
                    read a zipped file inside a tar specified by <file path>
                    - <file path>#pdf/<page>
                    read PNG image from a PDF at page <page>
                    - (ordinary file system paths, with '/' replaced with `os.path.sep`

        :type path: str
        :param mode: mode for read or write (binary only), defaults to 'rb'
        :type mode: str, optional
        :return: opened file/IO buffer
        :rtype: IO
        """
        parsed = urllib.parse.urlparse(path)
        buf = None
        exc = None
        
        for mgr in self._get_managers(parsed.scheme):
            try:
                if not isinstance(mgr, StorageProxyManager):
                    dst = path.split('#')[0]
                else:
                    dst = path
                    parsed = urllib.parse.urlparse(
                        path.replace('#', '__hash/'))
                buf = getattr(mgr, 'readbuf' if mode ==
                              'rb' else 'writebuf')(dst, **params)
                break
            except OSError as ex:
                exc = ex
                
        if buf is None:
            raise exc or OSError('Unable to open: ' + path)

        if parsed.fragment:
            handler_name, *fragments = parsed.fragment.split('/')
            if handler_name in self._fragment_handlers:
                assert mode == 'rb', 'Fragment handlers can only be applied in read mode'
                buf = self._fragment_handlers[handler_name](buf, *fragments)

        return buf

    def _query_until(self, action, path, default, **params):
        """Query multiple proxies/managers until found a result, or return the default

        Args:
            action (str): action name
            path (str): full path
            default (Any): default value
        """
        for mgr in self._get_managers(path):
            try:
                assert hasattr(mgr, action)
                val = getattr(mgr, action)(path, **params)
            except OSError:
                val = None
            except AssertionError:
                val = None
            if val:
                return val
        return default

    def exists(self, path):
        """Check if path exists
        :param path: file system path or URLs.
        :type path: str
        :return: True if existent, otherwise False
        """
        return self._query_until('exists', path, False)

    def statdir(self, path: str):
        """Get stat of a directory

        :param path: file system path or URLs.
        :type path: str
        :return: stat of the directory
        """
        return self._query_until('statdir', path, [])

    def listdir(self, path: str):
        """Get file list of a directory

        :param path: file system path or URLs.
        :type path: str
        :return: stat of the directory
        """
        return self._query_until('listdir', path, [])

    def stat(self, path: str):
        """Get stat of a file

        :param path: file system path or URLs.
        :type path: str
        :return: stat of the directory
        """
        return self._query_until('stat', path, {})

    def search(self, path: str, name_pattern: str):
        """Find a file
        """
        return self._query_until('search', path, [], name_pattern=name_pattern)

    def mkdir(self, path: str, new_folder: str):
        """Make directory in path"""
        return self._query_until('mkdir', path, False, new_folder=new_folder)

    @staticmethod
    def get_mimetype(ext):
        """Get mimetype from extension name"""
        return {
            'html': 'text/html',
                    'htm': 'text/html',
                    'png': 'image/png',
                    'jpg': 'image/jpeg',
                    'gif': 'image/gif',
                    'json': 'application/json',
                    'css': 'text/css',
                    'js': 'application/javascript',
                    'mp4': 'video/mp4'
        }.get(ext, 'application/octet-stream')

    @staticmethod
    def get_schemed_path(scheme, path):
        """Get schemed path

        Args:
            scheme (scheme): scheme name
            path (path): path without scheme

        Returns:
            str: full schemed path
        """
        path = path.replace('__hash/', '#')
        if scheme == 'file': path = '/' + path
        path = f'{scheme}://{path}'
        ext = path.rsplit('.', 1)[-1]
        return path, ext.lower() if len(ext) <= 4 else ''

    def serve_file(self, path_or_io: Union[str, IO], ext: str = '', file_size: int = 0) -> Response:
        """Serve static file or buffer

        Args:
            p (Union[str, IO]): file name or buffer
            ext (str, optional): extension name
            file_size (int, optional): file size

        Returns:
            Response: a flask response object
        """
        if isinstance(path_or_io, str):
            input_file = self.open(path_or_io, 'rb')
            ext = path_or_io.rsplit('.', 1)[-1]
            file_size = self.stat(path_or_io)['size']
        else:
            input_file = path_or_io
            ext = getattr(input_file, 'name', '').rsplit(
                '.', 1)[-1][:4].lower()

        mimetype = self.get_mimetype(ext)

        if file_size <= 0:
            file_size = getattr(path_or_io, 'st_size', 0)

        if file_size <= 0:
            input_file = BytesIO(input_file.read())
            file_size = len(input_file.getvalue())

        start, length = 0, file_size
        range_header = request.headers.get('Range')
        if range_header:
            # example: 0-1000 or 1250-
            matched_nums = re.search('([0-9]+)-([0-9]*)', range_header)
            num_groups = matched_nums.groups()
            if num_groups[0]:
                start = min(int(num_groups[0]), file_size - 1)
            if num_groups[1]:
                end = min(int(num_groups[1]), file_size - 1)
            else:
                end = file_size - 1
            length = end - start + 1

        def _generate_chunks(coming_length):
            input_file.seek(start)
            while coming_length > 0:
                chunk = input_file.read(min(coming_length, 1 << 20))
                coming_length -= len(chunk)
                yield chunk

        if range_header:
            resp = Response(stream_with_context(_generate_chunks(length)), 206,
                            content_type=mimetype, direct_passthrough=True)
            resp.headers.add(
                'Content-Range', f'bytes {start}-{start+length-1}/{file_size}')
            resp.headers.add('Content-Length', length)
        else:
            resp = Response(input_file.read(), 200,
                            content_type=mimetype, direct_passthrough=True)
        resp.headers.add('Accept-Ranges', 'bytes')
        resp.headers.add('Access-Control-Allow-Origin', '*')
        resp.headers.add('Access-Control-Allow-Headers', 'Accept')
        return resp

    def expand_path(self, path):
        """Expand path to full os file path

        Args:
            path (str): path

        Returns:
            str: os file path
        """
        expanded = ''
        for mgr in self._get_managers(path):
            try:
                val = mgr.expand_path(path)
            except OSError:
                val = None
            if mgr.exists(val):
                return val
            elif val and not expanded:
                expanded = val
        return expanded or path

    def truncate_path(self, path):
        """Truncate path to relative path

        Args:
            path (str): path

        Returns:
            str: path relative to storage bases
        """
        return self._query_until('truncate_path', path, path)

    def move(self, src, dst):
        """Move file from src to dst

        Args:
            src (str): source path
            dst (str): destination path

        Returns:
            bool: success flag
        """
        return self._query_until('move', src, False, dst=dst)

    def globs(self, patterns: Union[list, str, tuple], expand_zips=True):
        """Get expanded paths according to wildcards patterns

        :param patterns: patterns for looking up files.
            Wildcards (*/?) supported for file system paths.
            Brackets ({num1-num2}) supported for URLs.
        :type patterns: Union[list, str]
        :yield: full/absolute path
        :rtype: str
        """

        if isinstance(patterns, str):
            patterns = patterns.split('\n')

        if isinstance(patterns, tuple):
            patterns = list(patterns)

        patterns = deque(patterns)
        while patterns:
            pattern = patterns.popleft()
            
            if '://' not in pattern:
                pattern = 'file://' + pattern

            iterate = re.search(r'\{(\d+\-\d+)\}', pattern)
            if iterate:
                start, end = map(int, iterate.group(1).split('-'))
                for i in range(start, end+1):
                    patterns.append(pattern.replace(iterate.group(0), str(i)))
                continue

            if '*' in pattern:
                segs = pattern.split('/')
                for i, seg in enumerate(segs):
                    if '*' in seg:
                        parents = segs[:i]
                        break
                parent = '/'.join(parents) + '/'
                for mgr in self._get_managers(parent):
                    try:
                        for pattern in mgr.search(parent, pattern):
                            patterns.append(pattern)
                    except OSError:
                        continue
                continue
            
            if expand_zips and pattern.endswith(('.zip', '.epub')):
                try:
                    with zipfile.ZipFile(self.open(pattern, 'rb')) as zfile:
                        for item in zfile.filelist:
                            yield pattern + '#zip/' + item.filename
                except zipfile.BadZipFile:
                    print('Bad zip file', pattern)
            else:
                yield pattern

    def fspath(self, base_file, *segs):
        """Get full os file path in relative to base_file

        Args:
            base_file (str): base file.

        Returns:
            str: full os file path
        """
        return os.path.join(os.path.dirname(os.path.abspath(base_file)), *segs)
    
    def default_path(self, name):
        default_storage = config.storage.get('default', 'file')
        if '://' not in default_storage:
            default_storage += '://'
        elif not default_storage.endswith('/'):
            default_storage += '/'
        return f'{default_storage}{name}'
    
    def serve(self, host='0.0.0.0', port=8371, debug=False):
        """Start storage server

        Args:
            host (str, optional): Host name. Defaults to '0.0.0.0'.
            port (int, optional): Port number. Defaults to 8371.

        Returns:
            Flask: Flask app
        """
        storage_app = Flask(__name__)
        storage_app.secret_key = config.secret_key

        @storage_app.route('/<scheme>/<path:path>', methods=['GET'])
        @storage_app.route('/<scheme>', methods=['GET'])
        def get_item(scheme, path='/'):
            if scheme == 'file':
                path = '/' + path.lstrip('/')
            path, ext = Storage.get_schemed_path(scheme, path)

            # handle with storage queries
            params = request.args.to_dict()
            action = params.pop('action', '')
            if action:
                result = getattr(self, action, lambda *x, **kws: None)(path, **params)
                if action in ('walk', 'search'):
                    result = list(result or [])
                return jsonify(result)

            resp = self.serve_file(path, ext)
            resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp

        @storage_app.route('/<scheme>/<path:path>', methods=['PUT', 'POST'])
        def put_item(scheme, path):
            path, _ = Storage.get_schemed_path(scheme, path)
            if not request.data:
                return jsonify({'__exception__': 'No data, ignored.'})
            with self.open(path, 'wb') as fout:
                fout.write(request.data)
            return jsonify({'result': True})        

        storage_app.debug = debug

        if storage_app.debug or '-d' in sys.argv:
            storage_app.run(host=host, port=port, debug=True)
        else:
            serve(storage_app, host=host, port=port, threads=8,
                  connection_limit=512, backlog=2048)

        return storage_app


instance = Storage()
