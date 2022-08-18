"""File/Network storage access"""

from cgitb import handler
import enum
from fileinput import filename
import glob
import os
import re
import tempfile
import time
from io import BytesIO
from threading import Lock
from typing import Tuple, Union
import zipfile
from unrar import rarfile
import h5py
import numpy as np
import requests
from urllib import request
from pdf2image import convert_from_path as _pdf_convert
from flask import Flask, Response, request, send_file
import urllib
from smb.SMBHandler import SMBHandler

from .config import instance as config

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

storage_app = Flask(__name__)
storage_app.secret_key = config.secret_key


class Hdf5Manager:
    """HDF5 Manager"""

    files = []
    for storage_parent in [config.storage] + (config.external_storage.get('.', [])):
        files += [h5py.File(g, 'r') for g in glob.glob(os.path.join(
            storage_parent, '*.h5')) if not g.endswith(os.path.sep + 'blocks.h5')]
    base = os.path.join(config.storage, 'blocks.h5')

    writable_file = None
    _lock = Lock()

    def __enter__(self, *_):
        """Enter `with` block

        :return: self
        :rtype: Hdf5Manager
        """
        Hdf5Manager._lock.acquire()
        if Hdf5Manager.writable_file and Hdf5Manager.writable_file.mode != 'r+':
            Hdf5Manager.writable_file.close()
            Hdf5Manager.writable_file = None

        if not Hdf5Manager.writable_file:
            Hdf5Manager.writable_file = h5py.File(Hdf5Manager.base, 'r+')

        return self

    def __exit__(self, *_):
        """Exit `with` block
        """
        Hdf5Manager.writable_file.flush()
        Hdf5Manager._lock.release()

    @staticmethod
    def write(src, item_id):
        """Write data from src to h5df file with specific item id

        :param src: bytes or io object to read bytes from
        :type src: IO | bytes
        :param item_id: item id
        :type item_id: str
        :return: True if success
        :rtype: bool
        """

        assert Hdf5Manager.writable_file, 'Please use `with` statement.'

        if isinstance(src, bytes):
            src = BytesIO(src)

        k = f'data/{item_id}'
        if k in Hdf5Manager.writable_file:
            del Hdf5Manager.writable_file[k]

        Hdf5Manager.writable_file[k] = np.frombuffer(src.read(), dtype='uint8')

        return True

    @staticmethod
    def read(item_id: str) -> bytes:
        """Read data from h5df file

        :param item_id: id string
        :type item_id: str
        :raises OSError: Not found
        :return: data
        :rtype: BytesIO
        """

        if not os.path.exists(Hdf5Manager.base):
            Hdf5Manager.writable_file = h5py.File(Hdf5Manager.base, 'w')
            Hdf5Manager.writable_file.close()
            Hdf5Manager.writable_file = None

        if not Hdf5Manager.writable_file:
            Hdf5Manager.writable_file = h5py.File(Hdf5Manager.base, 'r')

        key = f'data/{item_id}'
        for block_file in [Hdf5Manager.writable_file] + Hdf5Manager.files:
            if key in block_file:
                return BytesIO(block_file[key][:].tobytes())

        raise OSError(f"No matched ID: {item_id}")

    def delete(self, item_id: str) -> None:
        """Delete item key

        :param item_id: item id
        :type item_id: str
        """
        key = f'data/{item_id}'
        for block_file in [Hdf5Manager.writable_file] + Hdf5Manager.files:
            if key in block_file:
                del block_file[key]


class _Hdf5WriteBuffer(BytesIO):
    """Buffer object for writing Hdf5
    """

    def __init__(self, item_id, initial_bytes=b''):
        """Initialize the buffer object

        :param item_id: item id
        :type item_id: str
        :param initial_bytes: bytes for initialization, defaults to b''
        :type initial_bytes: bytes, optional
        """
        self.item_id = item_id
        super().__init__(initial_bytes)

    def close(self):
        """close
        """
        with Hdf5Manager():
            Hdf5Manager.write(self.getvalue(), self.item_id)
        super().close()


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
        with zipfile.ZipFile(self.path) as zip_file:
            zip_file.writestr(self.zfile, self.getvalue())


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
        self.req = _build_request(url, method, **params)
        super().__init__()

    def close(self):
        self.req.data = self.getvalue()
        super().close()
        requests.Session().send(self.req.prepare())


class _SMBClient:
    """SMB Client
    """

    def __init__(self, smb_string) -> None:
        self.connection_path = smb_string

        parsed = urllib.parse.urlparse(smb_string)

        from smb.SMBConnection import SMBConnection
        self.conn = SMBConnection(parsed.username, parsed.password, '', '')
        self.conn.connect(parsed.hostname)

    def listdir(self, path):
        return self.conn.listPath(self.connection_path, path)

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

    def mkdir(self, path):
        service, path = path.split('/', 1)
        self.conn.createDirectory(service, path)

    def open(self, path, mode='rb'):
        service, path = path.split('/', 1)
        if mode == 'rb':
            smb_opener = urllib.request.build_opener(SMBHandler)
            fh = smb_opener.open(self.connection_path + '/' + service + path)
            return fh
        elif mode == 'wb':
            return _SMBWriteBuffer(self.conn, service, path)


class _SMBWriteBuffer(BytesIO):

    def __init__(self, conn, service, path):
        super().__init__()
        self.conn = conn
        self.service = service
        self.path = path

    def close(self):
        self.conn.storeFile(self.service, self.path, self)


def safe_statdir(path):
    assert '/..' not in path and '\\..' not in path, 'Illegal path: found `..`'
    path = expand_path(path)

    def _file_detail(path):
        """Get detailed status of local file"""

        file_stat = os.stat(path)
        return {
            'name': os.path.basename(path),
            'fullpath': path[len(config.storage):],
            'ctime': file_stat.st_ctime,
            'mtime': file_stat.st_mtime,
            'size': file_stat.st_size,
            'type': 'folder' if os.path.isdir(path) else 'file'
        }

    def _listsmb(path):
        return [
            {
                'name': f.filename,
                'fullpath': path + '/' + f.filename,
                'ctime': f.create_time,
                'mtime': f.last_write_time,
                'size': f.alloc_size,
                'type': 'folder' if f.isDirectory else 'file'
            }
            for f in client.listdir(path)
        ]

    parsed = urllib.parse.urlparse(path)
    if parsed.scheme == '':
        if os.path.isfile(path):
            return [_file_detail(path)]

        if os.path.exists(path):
            return [
                _file_detail(os.path.join(path, x)) for x in os.listdir(path)
                if not x.startswith(('@', '.'))
            ]

    elif parsed.scheme == 'smb':
        client = _SMBClient('smb://' + parsed.netloc)
        path = parsed.path.lstrip('/')
        if '/' not in path:
            path += '/'

        # try to handle the path as a directory
        try:
            return _listsmb(path)
        except:
            pass

        # assume it is a file, let's check the parent directory
        parent, filename = path.rsplit('/', 1)
        results = [f for f in client.listdir(parent) if f.filename == filename]
        return results


def safe_find(root_path, pattern):
    root_path = expand_path(root_path)
    parsed = urllib.parse.urlparse(root_path)
    conds = re.compile('.*'.join([re.escape(cond)
                                  for cond in pattern.split() if cond]), flags=re.I)

    if parsed.scheme == '':
        target = os
    elif parsed.scheme == 'smb':
        target = _SMBClient('smb://' + parsed.netloc)
        root_path = parsed.path

    for pwd, _, files in target.walk(root_path):
        for f in files:
            if conds.search(f):
                yield os.path.join(pwd, f)


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


def _try_download(url, attempts: int = 3, proxies=None, verify=False, timeout=60,
                  **params) -> Union[bytes, None]:
    """Try download from url

    :param url: url
    :type url: str
    :param attempts: max times of attempts, defaults to 3
    :type attempts: int, optional
    :param proxies: proxies info in requests format, defaults to None
    :type proxies: dict, optional
    :param verify: verify ssl, defaults to False
    :type verify: bool, optional
    :param timeout: seconds to timeout, defaults to 60
    :type timeout: int, optional
    :return: response content or None if failed
    :rtype: Union[bytes, None]
    """

    buf = None
    req = _build_request(url, **params).prepare()
    for _ in range(attempts):
        try:
            code = -1
            try:
                resp = requests.Session().send(req, proxies=proxies, verify=verify, timeout=timeout)
                buf = resp.content
                code = resp.status_code
            except requests.exceptions.ProxyError:
                buf = None
            if code != -1:
                break
        except Exception:
            time.sleep(1)

    return buf


def expand_path(path: Union[Tuple[str], str], allowed_locations=None):
    """Expand path to local storage path

    :param path: the path to expand
    :type path: Union[Tuple[str], str]
    :return: str
    :rtype: full/absolute path
    """

    if isinstance(path, tuple):
        path = os.path.sep.join([str(x) for x in path])

    if '://' in path:
        return path
    if '#' in path:
        path, _ = path.split('#', 1)

    path = path.replace('/', os.path.sep)

    if allowed_locations is None:
        allowed_locations = [
            config.storage,
            tempfile.gettempdir()
        ]

    if allowed_locations and not path.startswith(tuple(allowed_locations)):
        if path.startswith((os.path.altsep or os.path.sep, os.path.sep)):
            path = path[1:]
        if re.match(r'[A-Za-z]:\\', path):
            path = path[3:]
        for parent in allowed_locations:
            tmpath = os.path.join(parent, path)
            if os.path.exists(tmpath):
                return tmpath

    return os.path.join(config.storage, path)


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
        if pattern.startswith('https://') or pattern.startswith('http://'):
            urls = []
            iterate = re.search(r'\{(\d+\-\d+)\}', pattern)
            if iterate:
                start, end = map(int, iterate.group(1).split('-'))
                for i in range(start, end+1):
                    urls.append(pattern.replace(iterate.group(0), str(i)))
            else:
                urls = [pattern]
            yield from urls
        elif pattern.startswith('smb://'):
            from fnmatch import fnmatch
            parsed = urllib.parse.urlparse(pattern)
            segs = parsed.path.split('/')
            parents, wildcarded = segs, []
            for i, seg in enumerate(segs):
                if '*' in seg:
                    parents, wildcarded = segs[:i], segs[i:]
                    break
            parents = '/'.join(parents)
            if not wildcarded:
                yield [pattern]
            else:
                client = _SMBClient('smb://' + parsed.netloc)
                fnpattern = pattern[len('smb://' + parsed.netloc):]
                print(parents)
                for p, _, files in client.walk(parents):
                    for f in files:
                        print(p, f)
                        fn = '/' + p + f
                        if fnmatch(fn, fnpattern):
                            yield 'smb://' + parsed.netloc + fn
        else:
            pattern = expand_path(pattern, allowed_locations)
            for path in glob.glob(pattern):
                if path.endswith('.zip') or path.endswith('.epub'):
                    with zipfile.ZipFile(path, 'r') as rar_file:
                        for rar_item in rar_file.filelist:
                            yield path + '#zip/' + rar_item.filename
                elif path.endswith('.rar'):
                    with rarfile.RarFile(path, 'r') as rar_file:
                        for rar_item in rar_file.filelist:
                            yield path + '#rar/' + rar_item.filename
                elif os.path.isdir(path):
                    patterns.append(path + '/*')
                else:
                    yield path


def _open_path(parsed, mode, **params):

    if parsed.scheme == '':
        fpath = expand_path(parsed.path)
        if '://' in fpath:
            return _open_path(fpath, mode, **params)
        else:
            parsed = urllib.parse.urlparse('file:// ' + fpath)

    # always process http/https/data/smb URLs locally

    if parsed.scheme in ('http', 'https'):
        assert mode in ('rb', 'wb')
        if mode == 'rb':
            return BytesIO(_try_download(parsed.geturl(), **params))

        return _RequestBuffer(parsed.geturl(), **params)

    if parsed.scheme == 'data':
        assert mode == 'rb'
        with request.urlopen(parsed.geturl()) as response:
            data = response.read()
        return BytesIO(data)

    if parsed.scheme == 'smb':
        assert mode in ('rb', 'wb')
        client = _SMBClient('smb://' + parsed.netloc)
        return client.open(parsed.path, mode)

    # for other schemes, use storage proxy if existent

    if config.storage_proxy:
        target = '/'.join([_ for _ in (config.storage_proxy.rstrip('/'),
                          parsed.scheme, parsed.netloc, parsed.path) if _])
        if mode == 'rb':
            return BytesIO(_try_download(target))
        else:
            return _RequestBuffer(target)

    # handling with customized schemes

    if parsed.scheme == 'hdf5':
        assert mode in ('rb', 'wb')
        item_id = parsed.netloc
        if mode == 'rb':
            buf = Hdf5Manager.read(item_id)
            return buf
        else:
            return _Hdf5WriteBuffer(item_id)

    if parsed.scheme == 'file':
        buf = open(parsed.path, mode)
        buf.st_size = os.stat(fpath).st_size
        buf.filename = parsed.path
        return buf

    raise OSError('Unsupported scheme: ' + parsed.scheme)


def _fh_zip(buf, *inner_path):
    """Handle zip file"""
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
        with open('filename', 'wb') as f:
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

    buf = _open_path(parsed, mode, **params)
    if not buf:
        raise OSError('Unable to open: ' + path)

    if parsed.fragment:
        assert mode == 'rb', 'Fragment handlers can only be applied in read mode'
        handler_name, *fragments = parsed.fragment.split('/')
        if handler_name in _fragment_handlers:
            buf = _fragment_handlers[handler_name](buf, *fragments)

    return buf


def truncate_path(path, base=None):
    """Truncate path if it belongs to the base directory.

    :param path: file path
    :type path: str
    :param base: base directory path, defaults to None
    :type base: str, optional

    :return: truncated path
    :rtype: str
    """
    if base is None:
        base = config.storage
    if not base.endswith(os.path.sep):
        base += os.path.sep
    if path.startswith(base):
        return path[len(base):].replace('\\', '/')
    return path


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


def _get_schemed_path(scheme, path, ext):
    if ext:
        ext = '.' + ext
    path = f'{scheme}://{path}{ext}'
    return path


@storage_app.route('/<scheme>/<path:path>.<ext>', methods=['GET'])
@storage_app.route('/<scheme>/<path:path>', methods=['GET'])
def get_item(scheme, path, ext=''):
    path = _get_schemed_path(scheme, path, ext)
    try:
        buf = safe_open(path, 'rb')
    except OSError:
        return 'Not Found: ' + path, 404

    mimetype = _get_mimetype(ext[1:])

    resp = send_file(buf, mimetype)
    resp.headers.add("Cache-Control", "public,max-age=86400")
    return resp


@storage_app.route('/<scheme>/<path:path>.<ext>', methods=['PUT', 'POST'])
@storage_app.route('/<scheme>/<path:path>', methods=['PUT', 'POST'])
def put_item(scheme, path, ext=''):
    path = _get_schemed_path(scheme, path, ext)
    with safe_open(path, 'wb') as fo:
        fo.write(request.data)
    return 'OK'


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
