"""File/Network storage access"""

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

from .config import instance as config


class Hdf5Manager:
    """HDF5 Manager"""
    
    files = []
    for storage_parent in [config.storage] + (config.external_storage or []):
        files += [h5py.File(g, 'r') for g in glob.glob(os.path.join(storage_parent, '*.h5'))]
    base = os.path.join(config.storage, 'blocks.h5')
    if base in files:
        files.remove(base)

    writable_file = None
    write_counter = 0
    _lock = Lock()

    def __enter__(self, *_):
        """Enter `with` block

        :return: self
        :rtype: Hdf5Manager
        """
        Hdf5Manager._lock.acquire()
        if not Hdf5Manager.writable_file or Hdf5Manager.writable_file.mode != 'r+':
            if Hdf5Manager.writable_file:
                Hdf5Manager.writable_file.close()
            Hdf5Manager.writable_file = h5py.File(Hdf5Manager.base, 'r+')
        Hdf5Manager.write_counter = 0
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
        super().close()
        self.req.data = self.getvalue()
        requests.Session().send(self.req.prepare())


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
            os.path.join(tempfile.gettempdir(), tempfile.gettempprefix()),
            config.storage
        ]

    if allowed_locations and not path.startswith(tuple(allowed_locations)):
        if path.startswith((os.path.altsep or os.path.sep, os.path.sep)):
            path = path[1:]
        path = os.path.join(config.storage, path)

    return path


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
                - <file path>#pdf/png:<page>
                  read PNG image from a PDF at page <page>
                - (ordinary file system paths, with '/' replaced with `os.path.sep`

    :type path: str
    :param mode: mode for read or write, defaults to 'rb'
    :type mode: str, optional
    :return: opened file/IO buffer
    :rtype: IO
    """

    if path.startswith(('http://', 'https://')):
        assert mode in ('rb', 'wb')
        if mode == 'rb':
            return BytesIO(_try_download(path, **params))

        return _RequestBuffer(path, **params)

    if path.startswith('data:'):
        assert mode == 'rb'
        with request.urlopen(path) as response:
            data = response.read()
        return BytesIO(data)

    if path.startswith('hdf5://'):
        assert mode in ('rb', 'wb')
        item_id = path.split('://', 1)[1]
        if mode == 'rb':
            if config.hdf5_proxy:
                return BytesIO(_try_download(config.hdf5_proxy + item_id))
            return Hdf5Manager.read(item_id)

        return _Hdf5WriteBuffer(item_id)

    fpath = expand_path(path)

    if '#zip/' in path:
        assert mode in ('rb', 'wb')
        _, zpath = path.split('#zip/', 1)
        if mode == 'rb':
            with zipfile.ZipFile(fpath) as zip_file:
                return zip_file.open(zpath)
        else:
            return _ZipWriteBuffer(path, zpath)
        
    elif '#rar/' in path:
        assert mode == 'rb'
        _, zpath = path.split('#rar/', 1)
        with rarfile.RarFile(fpath) as rar_file:
            return rar_file.open(zpath)
        
    elif '#pdf/png:' in path:
        assert mode == 'rb'
        _, page = path.split('#pdf/png:', 1)
        return _pdf_image(fpath, int(page), **params)

    else:
        buf = open(fpath, mode)
        buf.st_size = os.stat(fpath).st_size
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
