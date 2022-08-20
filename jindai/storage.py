"""File/Network storage access"""

from ast import Bytes
from cgitb import handler
from collections import defaultdict
import enum
from fileinput import filename
import glob
import itertools
import os
import re
import tempfile
import time
from io import BytesIO
from threading import Lock
from typing import Iterable, List, Tuple, Type, Union
import zipfile
from unrar import rarfile
import h5py
import numpy as np
import requests
import urllib
from pdf2image import convert_from_path as _pdf_convert
from flask import Flask, Response, request, send_file
from fnmatch import fnmatch
from smb.SMBHandler import SMBHandler

from .config import instance as config

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

storage_app = Flask(__name__)
storage_app.secret_key = config.secret_key


class StorageManager:
    """Storage manager with predefined functions
    """

    def stat(self, path: str) -> dict:
        """Stat file/folder

        Args:
            path (str): Path for stat

        Returns:
            dict: Stat result containing name, fullpath, ctime, mtime, size, type
        """
        return {
            'name': path.rstrip('/').rsplit('/')[-1],
            'fullpath': path,
            'ctime': time.time(),
            'mtime': time.time(),
            'size': 0,
            'type': 'folder|file'
        }

    def listdir(self, path: str) -> list:
        return []

    def statdir(self, path: str) -> list:
        return [self.stat(self.join(path, f)) for f in self.listdir(path)]

    def exists(self, path: str) -> bool:
        return False

    def read(self, path: str, **params) -> BytesIO:
        return BytesIO()

    def writebuf(self, path: str, **params) -> BytesIO:
        return BytesIO()

    def unlink(self, path: str) -> bool:
        return False

    def join(self, base_path: str, *path_segs: str) -> str:
        return '/'.join((base_path, *path_segs)).replace('..', '')

    def walk(self, base_path: str) -> Iterable[Tuple[str, List, List]]:
        dirs, files = [], []
        for f in self.listdir(base_path):
            if self.stat(f)['type'] == 'folder':
                dirs.append(f)
            else:
                files.append(f)
        yield base_path, dirs, files
        for d in dirs:
            yield from self.walk(self.join(base_path, d))

    def search(self, base_path: str, name_pattern: str) -> Iterable[str]:
        for path, dirs, files in self.walk(base_path):
            for f in dirs + files:
                if fnmatch(self.join(path, f), name_pattern):
                    yield self.join(path, f)


class OSFileSystemManager(StorageManager):
    """Storage manager for local file system"""

    def __init__(self, base: str, allowed_locations=None) -> None:
        if not base.endswith(os.path.sep):
            base += os.path.sep
        base = base.replace(os.path.sep, '/')
        self.base = base

        if allowed_locations is None:
            allowed_locations = [
                base,
                tempfile.gettempdir()
            ]
        self.allowed_locations = allowed_locations

    def expand_path(self, path: str) -> str:
        """Expand path to local storage path

        :param path: the path to expand
        :type path: Union[Tuple[str], str]
        :return: str
        :rtype: full/absolute path
        """

        if isinstance(path, tuple):
            path = os.path.sep.join([str(x) for x in path])

        if path.startswith('file://'):
            path = path[7:]

        if '://' in path:
            return path
        if '#' in path:
            path, _ = path.split('#', 1)

        path = path.replace('/', os.path.sep)

        if not path.startswith(tuple(self.allowed_locations)):
            if path.startswith((os.path.altsep or os.path.sep, os.path.sep)):
                path = path[1:]
            if re.match(r'[A-Za-z]:\\', path):
                path = path[3:]
            for parent in self.allowed_locations:
                tmpath = os.path.join(parent, path)
                if os.path.exists(tmpath):
                    return tmpath

        return os.path.join(self.base, path).replace(os.path.sep, '/')

    def truncate_path(self, path):
        """Truncate path if it belongs to the base directory.

        :param path: file path
        :type path: str
        :param base: base directory path, defaults to None
        :type base: str, optional

        :return: truncated path
        :rtype: str
        """
        path = path.replace(os.path.sep, '/')
        if path.startswith(self.base):
            return path[len(self.base):]
        return path

    def stat(self, path: str) -> list:
        path = self.expand_path(path)
        file_stat = os.stat(path)
        return {
            'name': os.path.basename(path),
            'fullpath': self.truncate_path(path),
            'ctime': file_stat.st_ctime,
            'mtime': file_stat.st_mtime,
            'size': file_stat.st_size,
            'type': 'folder' if os.path.isdir(path) else 'file'
        }

    def listdir(self, path: str) -> list:
        path = self.expand_path(path)
        return [f for f in os.listdir(path) if not f.startswith(('.', '@'))]

    def exists(self, path: str) -> bool:
        path = self.expand_path(path)
        return os.path.exists(path)

    def read(self, path: str, **params) -> BytesIO:
        path = self.expand_path(path)
        buf = open(path, 'rb')
        buf.filename = path
        return buf

    def writebuf(self, path: str, **params) -> BytesIO:
        path = self.expand_path(path)
        return open(path, 'wb')

    def unlink(self, path: str) -> bool:
        path = self.expand_path(path)
        try:
            os.unlink(path)
        except OSError:
            return False
        return True

    def join(self, path: str, *path_segs: str) -> str:
        path = self.expand_path(path)
        return 'file:///' + os.path.sep.join((path, *path_segs)).replace('..', '').replace(os.path.sep, '/').lstrip('/')


class Hdf5Manager(StorageManager):
    """HDF5 Manager"""

    _lock = Lock()

    class _Hdf5WriteBuffer(BytesIO):
        """Buffer object for writing Hdf5
        """

        def __init__(self, manager, item_id, initial_bytes=b''):
            """Initialize the buffer object

            :param item_id: item id
            :type item_id: str
            :param initial_bytes: bytes for initialization, defaults to b''
            :type initial_bytes: bytes, optional
            """
            self.item_id = item_id
            self.manager = manager
            super().__init__(initial_bytes)

        def close(self):
            with self.manager:
                self.manager.write(self.getvalue(), self.item_id)
            super().close()

    def __init__(self, storage_base: str, *external_storage: str) -> None:
        self.files = []
        for storage_parent in [storage_base, *external_storage]:
            self.files += glob.glob(os.path.join(storage_parent, '*.h5'))
        self.base = os.path.join(storage_base, 'blocks.h5')
        if self.base in self.files:
            self.files.remove(self.base)

        self.files = [h5py.File(g, 'r') for g in self.files]

        self._writable_file = None

        if not os.path.exists(self.base):
            self._writable_file = h5py.File(self.base, 'w')
            self._writable_file.close()
            self._writable_file = None

    def stat(self, path: str) -> dict:
        data = self.get(path)
        if data:
            return {
                'name': path,
                'fullpath': path,
                'ctime': time.time(),
                'mtime': time.time(),
                'size': len(data),
                'type': 'file'
            }

    def __enter__(self, *_):
        """Enter `with` block

        :return: self
        :rtype: Hdf5Manager
        """
        self._lock.acquire()
        if self._writable_file and self._writable_file.mode != 'r+':
            self._writable_file.close()
            self._writable_file = None

        if not self._writable_file:
            self._writable_file = h5py.File(self.base, 'r+')

        return self

    def __exit__(self, *_):
        """Exit `with` block
        """
        self._writable_file.flush()
        self._lock.release()

    def write(self, src, item_id):
        """Write data from src to hdf5 file with specific item id

        :param src: bytes or io object to read bytes from
        :type src: IO | bytes
        :param item_id: item id
        :type item_id: str
        :return: True if success
        :rtype: bool
        """

        assert self._writable_file, 'Please use `with` statement.'

        if isinstance(src, bytes):
            src = BytesIO(src)

        k = f'data/{item_id}'
        if k in self._writable_file:
            del self._writable_file[k]

        self._writable_file[k] = np.frombuffer(
            src.read(), dtype='uint8')

        return True

    def writebuf(self, path: str, **params) -> BytesIO:
        parsed = urllib.parse.urlparse(path)
        return self._Hdf5WriteBuffer(self, parsed.netloc)

    def get(self, item_id: str) -> bytes:
        """Read data from hdf5 file

        :param item_id: id string
        :type item_id: str
        :raises OSError: Not found
        :return: data
        :rtype: BytesIO
        """
        if not self._writable_file:
            self._writable_file = h5py.File(self.base, 'r')

        key = f'data/{item_id}'
        for block_file in [self._writable_file, *self.files]:
            if key in block_file:
                return block_file[key][:].tobytes()

    def read(self, path: str, **params):
        path = urllib.parse.urlparse(path).netloc
        buf = self.get(path)
        if buf:
            return BytesIO(buf)
        else:
            raise OSError(f"No matched ID: {path}")

    def listdir(self, path: str) -> list:
        return itertools.chain(*[list(s['data']) for s in self.files + [self._writable_file]])

    def walk(self, base_path: str) -> Iterable[Tuple[str, List, List]]:
        if base_path.rstrip('/') == '':
            yield '/', [], self.listdir('/')

    def exists(self, path: str) -> bool:
        return self.get(path) is not None

    def join(self, base_path: str, *path_segs: str) -> str:
        return path_segs[-1]


class WebManager(StorageManager):

    class _RequestBuffer(BytesIO):
        """Buffer object for post url requests
        """

        def __init__(self, url, method='POST', **params):
            """Initialize a buffer object for post requests

            :param url: url
            :type url: str
            :param method: method for the request, defaults to 'POST'
            :type method: str, optional
            """
            self.req = WebManager._build_request(url, method, **params)
            self.resp = None
            super().__init__()

        def close(self):
            self.req.data = self.getvalue()
            super().close()
            self.resp = requests.Session().send(self.req.prepare())
            if self.resp.status_code != 200:
                raise OSError(f'{self.resp.status_code} {self.resp.reason}')

    @staticmethod
    def _build_request(url: str, method='GET', referer: str = '',
                       headers=None, data=None, **_):
        """Build a request

        :param url: target url
        :type url: str
        :param method: request  method, defaults to 'GET'
        :type method: str, optional
        :param referer: referer url, defaults to ''
        :type referer: str, optional
        :param headers: headers, defaults to None
        :type headers: dict, optional
        :param data: request body data, defaults to None
        :type data: bytes, optional
        :return: request context
        :rtype: requests.Request
        """

        if headers is None:
            headers = {}

        headers.update(**{
            "user-agent": "Mozilla/5.1 (Windows NT 6.0) Gecko/20180101 Firefox/23.5.1"
        })

        if isinstance(url, tuple):
            url, referer = url

        if referer is not None:
            headers["referer"] = referer.encode('utf-8')

        return requests.Request(url=url, method=method, headers=headers, cookies={}, data=data)

    def __init__(self, attempts=3, verify=False, timeout=30):
        self.attempts = attempts
        self.verify = verify
        self.timeout = timeout

    def writebuf(self, path: str, **params) -> BytesIO:
        return WebManager._RequestBuffer(path, **params)

    def read(self, path: str, proxies={}, **params) -> BytesIO:
        buf = None
        req = WebManager._build_request(path, **params).prepare()
        for _ in range(self.attempts):
            try:
                code = -1
                try:
                    resp = requests.Session().send(req, verify=self.verify,
                                                   timeout=self.timeout, proxies=proxies)
                    buf = resp.content
                    code = resp.status_code
                except requests.exceptions.ProxyError:
                    buf = None
                if code != -1:
                    break
            except Exception:
                time.sleep(1)

        return BytesIO(buf)

    def exists(self, path: str) -> bool:
        return True

    def join(self, base_path: str, *path_segs: str) -> str:
        return urllib.parse.urljoin(base_path, '/'.join(path_segs))

    def listdir(self, path: str) -> list:
        return []

    def stat(self, path: str) -> dict:
        d = super().stat(path)
        d['type'] = 'file'
        return d


class SMBManager(StorageManager):
    """SMB Client
    """

    class _SMBWriteBuffer(BytesIO):

        def __init__(self, conn, service, path):
            super().__init__()
            self.conn = conn
            self.service = service
            self.path = path

        def close(self):
            self.conn.storeFile(self.service, self.path, self)

    def __init__(self) -> None:
        self._connections = {}
        self._last_active = {}
        self._ttl = 60
        self._opener = urllib.request.build_opener(SMBHandler)

    def _smb_split(self, path):
        # path in the form of smb://__netloc__/__service__/__path__
        return path[6:].split('/', 2)

    def _get_connection(self, path):
        from smb.SMBConnection import SMBConnection
        netloc, service, path = self._smb_split(path)
        if netloc not in self._connections or self._last_active.get(netloc, 0) < time.time() - self._ttl:
            parsed = urllib.parse.urlparse(path)
            self._connection[netloc] = SMBConnection(
                parsed.username, parsed.password, '', '')

        self._last_active[netloc] = time.time()
        return self._connection[netloc], service, path

    def statdir(self, path):
        path = path.rstrip('/')
        conn, service, path = self._get_connection(path)
        return [f for f in conn.listPath(service, path + '/') if not f.filename.startswith(('.', '@'))]

    def listdir(self, path):
        return [f.filename for f in self.statdir(path)]

    def stat(self, path):
        path = path.rstrip('/')
        parent, filename = path.rsplit('/', 1)
        matched = [
            {
                'name': f.filename,
                'fullpath': path + '/' + f.filename,
                'ctime': f.create_time,
                'mtime': f.last_write_time,
                'size': f.alloc_size,
                'type': 'folder' if f.isDirectory else 'file'
            }
            for f in self.statdir(parent)
            if f.filename == filename
        ]
        if matched:
            return matched[0]

    def exists(self, path: str) -> bool:
        return self.stat(path) is not None

    def walk(self, start_path):
        start_path = start_path.strip('/') + '/'
        service, path = start_path.split('/', 1)
        file_dirs = self.conn.listPath(service, path)

        files = [f.filename for f in file_dirs if not f.isDirectory]
        dirs = [f.filename for f in file_dirs if f.isDirectory and f.filename !=
                '.' and f.filename != '..']
        yield start_path, dirs, files

        for d in dirs:
            yield from self.walk('/'.join([start_path.rstrip('/'), d]))

    def read(self, path, **params):
        return self._opener.open(path)

    def writebuf(self, path: str, **params):
        return SMBManager._SMBWriteBuffer(*self._get_connection(path))


def _pdf_image(file: str, page: int, **_) -> BytesIO:
    """Get PNG file from PDF

    :param file: PDF path
    :type file: str
    :param page: page index, starting form 0
    :type page: int
    :return: PNG bytes in a BytesIO
    :rtype: BytesIO
    """
    buf = BytesIO()
    page = int(page)

    img, = _pdf_convert(file, 120, first_page=page+1,
                        last_page=page+1, fmt='png') or [None]
    if img:
        img.save(buf, format='png')
        buf.seek(0)

    return buf


class DataSchemeManager(StorageManager):

    def exists(self, path: str) -> bool:
        return True

    def read(self, path, **params):
        with urllib.request.urlopen(path) as response:
            data = response.read()
        return BytesIO(data)


SCHEMES = {
    'smb': SMBManager(),
    'http': WebManager(),
    'https': WebManager(),
    'data': DataSchemeManager(),
    'hdf5': Hdf5Manager(config.storage, *(config.external_storage or [])),
    '': OSFileSystemManager(config.storage)
}


def _get_manager(path) -> StorageManager:
    if path in SCHEMES:
        return SCHEMES[path]

    if '://' in path and not path.startswith('file://'):
        scheme = path.split('://', 1)[0]
        return SCHEMES[scheme]

    return SCHEMES['']


expand_path = SCHEMES[''].expand_path
truncate_path = SCHEMES[''].truncate_path


def expand_patterns(patterns: Union[list, str, tuple], allowed_locations=None):
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

    patterns.reverse()
    while patterns:
        pattern = patterns.pop()

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
            parent = '/'.join(parents)
            yield from _get_manager(pattern).search(parent, pattern)
            continue

        path = pattern
        if path.endswith('.zip') or path.endswith('.epub'):
            with zipfile.ZipFile(path, 'r') as zfile:
                for item in zfile.filelist:
                    yield path + '#zip/' + item.filename
        else:
            yield path


def _fh_zip(buf, *inner_path):
    """Handle zip file"""

    class _ZipWriteBuffer(BytesIO):
        """Buffer object for writing zip files
        """

        def __init__(self, path, zfile):
            """Initialize a buffer object for writing zip files

            :param path: path for zip file
            :type path: str
            :param zfile: path inside the zip file
            :type zfile: str
            """
            self.path = path
            self.zfile = zfile
            super().__init__()

        def close(self):
            super().close()
            with zipfile.ZipFile(self.path, 'a') as zip_file:
                zip_file.writestr(self.zfile, self.getvalue())

    zpath = '/'.join(inner_path)
    with zipfile.ZipFile(buf, 'r') as zip_file:
        return zip_file.open(zpath)


def _fh_pdf(buf, page):
    if hasattr(buf, 'filename'):
        filename = buf.filename
        temp = False
    else:
        filename = tempfile.mktemp(suffix='.pdf')
        temp = True
        with open(filename, 'wb') as f:
            f.write(buf.read())
    buf = _pdf_image(filename, int(page))
    if temp:
        os.unlink(filename)
    return buf


def _fh_thumbnail(buf, width, height=''):
    if not height:
        height = width
    width, height = int(width), int(height)
    from PIL import Image
    im = Image.open(buf)
    im.thumbnail((width, height))
    buf = BytesIO()
    im.save(buf, 'JPEG')
    buf.seek(0)
    return buf


_fragment_handlers = {
    fh[4:]: func for fh, func in globals().items() if fh.startswith('_fh_')
}


def safe_open(path: str, mode='rb', **params):
    """Open the path for read/write

    :param path: file system path or URLs.
            Relative and absolute paths are supported.
            Paths are matched in the following priority:
                - URLs starting with http:// or https:// 
                  may use `proxies`, `referer`, and `headers` in `params`
                - Data URLs, i.e. data:...
                  readonly, `mode` should be 'rb'
                - hdf5://<item id>
                  `mode` should be 'rb' or 'wb'
                - <file path>#zip/<zip path>
                  read a zipped file inside a tar specified by <file path>
                - <file path>#pdf/<page>
                  read PNG image from a PDF at page <page>
                - (ordinary file system paths, with '/' replaced with `os.path.sep`

    :type path: str
    :param mode: mode for read or write, defaults to 'rb'
    :type mode: str, optional
    :return: opened file/IO buffer
    :rtype: IO
    """
    parsed = urllib.parse.urlparse(path.replace('__hash/', '#'))

    if parsed.scheme not in ('http', 'https') and config.storage_proxy:
        target = '/'.join([_ for _ in (config.storage_proxy.rstrip('/'),
                          parsed.scheme or 'file', parsed.netloc, parsed.path.lstrip('/')) if _])
        if parsed.fragment:
            target += '__hash/' + parsed.fragment

        if mode == 'rb':
            return safe_open(target, 'rb')
        else:
            return safe_open(target, 'wb')

    mgr = _get_manager(parsed.scheme)
    buf = getattr(mgr, 'read' if mode == 'rb' else 'writebuf')(path, **params)

    if not buf:
        raise OSError('Unable to open: ' + path)

    if parsed.fragment:
        handler_name, *fragments = parsed.fragment.split('/')
        if handler_name in _fragment_handlers:
            assert mode == 'rb', 'Fragment handlers can only be applied in read mode'
            buf = _fragment_handlers[handler_name](buf, *fragments)

    return buf


def safe_statdir(path: str):
    """Get stat of a directory

    :param path: file system path or URLs.
    :type path: str
    :return: stat of the directory
    """
    parsed = urllib.parse.urlparse(path.replace('__hash/', '#'))
    return _get_manager(parsed.scheme).statdir(path)


def safe_find(path: str, pattern: str):
    """Find a file
    """
    parsed = urllib.parse.urlparse(path.replace('__hash/', '#'))
    return _get_manager(parsed.scheme).search(path, pattern)


def _get_mimetype(ext):
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


def _get_schemed_path(scheme, path):
    path = path.replace('__hash/', '#')
    path = f'{scheme}://{path}'
    if '#' in path:
        filepath = path.split('#')[0]
    else:
        filepath = path
    ext = filepath.rsplit('.', 1)[-1]
    return path, ext.lower() if len(ext) <= 4 else ''


@storage_app.route('/<scheme>/<path:path>', methods=['GET'])
def get_item(scheme, path):
    path, ext = _get_schemed_path(scheme, path)
    try:
        buf = safe_open(path, 'rb')
    except OSError:
        return 'Not Found: ' + path, 404

    mimetype = _get_mimetype(ext)

    resp = send_file(buf, mimetype)
    resp.headers.add("Cache-Control", "public,max-age=86400")
    return resp


@storage_app.route('/<scheme>/<path:path>', methods=['PUT', 'POST'])
def put_item(scheme, path):
    path, ext = _get_schemed_path(scheme, path)
    if not request.data:
        return 'no data', 501
    with safe_open(path, 'wb') as fo:
        fo.write(request.data)
    return f'OK {path}'


storage_app.debug = config.debug


def serve_storage(port, host='0.0.0.0'):
    import sys
    if '-d' not in sys.argv:
        from waitress import serve
        serve(storage_app, host='0.0.0.0', port=port, threads=8)
    else:
        storage_app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    serve_storage(config.port)
