"""File/Network storage access"""

import enum
import glob
import io
import itertools
import json
import os
import ssl
import re
import sys
import tempfile
import time
import urllib
import zipfile
import requests_cache
from fnmatch import fnmatch
from io import BytesIO, IOBase
from threading import Lock
from typing import IO, Iterable, List, Tuple, Union

import h5py
import numpy as np
import requests
from flask import (Flask, Response, jsonify, request, 
                   stream_with_context)
from pdf2image import convert_from_path as _pdf_convert
from smb.SMBHandler import SMBHandler

from .config import instance as config


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
            'size': -1,
            'type': 'folder|file'
        }

    def listdir(self, path: str) -> list:
        """List files/folders in path

        Args:
            path (str): Path in full form

        Returns:
            list: List of names in path
        """
        return []

    def statdir(self, path: str) -> list:
        """Stat folder and return list of stat results

        Args:
            path (str): Path in full form

        Returns:
            list: Stat info for each file/folder in folder
        """
        return [self.stat(self.join(path, f)) for f in self.listdir(path)]

    def exists(self, path: str) -> bool:
        """Check if path exists

        Args:
            path (str): Path in full form

        Returns:
            bool: True if path exists
        """
        return False
    
    def read(self, path: str, **params) -> bytes:
        """Get data bytes"""
        return b''

    def readbuf(self, path: str, **params) -> BytesIO:
        """Get read buffer for path

        Args:
            path (str): Path in full form

        Returns:
            BytesIO: Readable buffer
        """
        return BytesIO(self.read(path, **params))
    
    def writebuf(self, path, **params):
        """Get write buffer for path
        """
        return WriteBuffer(self, path, **params)
        
    def write(self, path, data : bytes) -> bool:
        """Write bytes data"""
        return True

    def unlink(self, path: str) -> bool:
        """Unlink a file/folder

        Args:
            path (str): Path in full form

        Returns:
            bool: True if succeeded
        """
        return False

    def join(self, base_path: str, *path_segs: str) -> str:
        """Join segements of a path

        Args:
            base_path (str): Base paths
            path_segs (Tuple[str]): Segments

        Returns:
            str: Full path
        """
        segs = []
        for s in path_segs:
            if s == '..' and segs:
                segs.pop()
            else:
                segs.append(s)
        return '/'.join((base_path.rstrip('/'), *[seg.strip('/') for seg in segs]))

    def walk(self, base_path: str, match_pattern: str = '') -> Tuple[str, List, List]:
        """Walk through a folder and return files and folders

        Args:
            base_path (str): Base path
            match_pattern (str, optional): Only continue when matching pattern, used for `search`

        Yields:
            Tuple[str, List, List]: Path, folders, files
        """
        base_path = base_path.rstrip('/') + '/'
        pattern_segs = '/'.join(match_pattern.split('/')[:base_path.count('/')+1])
        dirs, files = [], []

        for f in self.listdir(base_path):
            if self.stat(self.join(base_path, f))['type'] == 'folder':
                if not match_pattern or fnmatch(self.join(base_path, f), pattern_segs):
                    dirs.append(f)
            else:
                if not match_pattern or fnmatch(self.join(base_path, f), match_pattern):
                    files.append(f)

        yield base_path, dirs, files
        for d in dirs:
            yield from self.walk(self.join(base_path, d), match_pattern)

    def search(self, base_path: str, name_pattern: str) -> str:
        """Search for name_pattern in base_path

        Args:
            base_path (str): Searching start
            name_pattern (str): Name pattern to search for

        Yields:
            str: Matched paths
        """

        for path, dirs, files in self.walk(base_path, name_pattern):
            for f in dirs + files:
                fpath = self.join(path, f)
                if fnmatch(fpath, name_pattern):
                    yield fpath

    def mkdir(self, path: str, new_folder: str) -> bool:
        """Make a new directory in path"""
        return False
    
    def move(self, path: str, new_path: str) -> bool:
        """Move path name"""
        return False
    

class WriteBuffer(BytesIO):
    
    def __init__(self, writer: StorageManager, path : str, **write_params: dict) -> None:
        super().__init__()
        self._writer = writer
        self._params = write_params or {}
        self._params['path'] = path
        
    def close(self):
        self._writer.write(data=self.getvalue(), **self._params)
        super().close()


class OSFileSystemManager(StorageManager):
    """Storage manager for local file system"""

    def __init__(self, base: str) -> None:
        allowed_locations = []
        if isinstance(base, list):
            base, *allowed_locations = base

        allowed_locations = [
            loc + os.path.sep
            for loc in allowed_locations
            if not loc.endswith(os.path.sep)
        ]
        
        if not base.endswith(os.path.sep):
            base += os.path.sep
        self.base = base

        allowed_locations += [
            base,
            tempfile.gettempdir()
        ]
        self.allowed_locations = allowed_locations

    def expand_path(self, path: Union[Tuple[str], str]) -> str:
        """Expand path to local (OS-specific) storage path

        :param path: the path to expand
        :type path: Union[Tuple[str], str]
        :return: str
        :rtype: full/absolute path
        """
        suffix_path = ''

        if isinstance(path, tuple):
            path = os.path.sep.join([str(x) for x in path])

        if path.startswith('file://'):
            path = path[7:]

        if '#' in path:
            path, suffix_path = path.split('#', 1)
            suffix_path = '#' + suffix_path

        if '*' in path:
            segs = path.split('/')
            for i, seg in  enumerate(segs):
                if '*' in seg: break
            parent_path = '/'.join(segs[:i]) + '/'
            suffix_path = '/'.join(segs[i:])
            path = parent_path

        path = path.replace('/', os.path.sep) # path is now os-specific

        if not path.startswith(tuple(self.allowed_locations)):
            if path.startswith(os.path.sep):
                path = path[1:]
            for parent in self.allowed_locations:
                if path.startswith(parent + os.path.sep):
                    return path + suffix_path
                
                tmpath = os.path.join(parent, path)
                if os.path.exists(tmpath):
                    return tmpath + suffix_path
            
            return os.path.join(self.base, path) + suffix_path
        else:
            return path + suffix_path

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
        if path.startswith('file://'): path = path[7:]
        for base in self.allowed_locations:
            if path.startswith(base):
                return path[len(base):]
        return path

    def stat(self, path: str) -> list:
        path = self.expand_path(path)
        if '#' in path:
            return super().stat(path)
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

    def readbuf(self, path: str, **params) -> BytesIO:
        path = self.expand_path(path)
        buf = open(path, 'rb')
        return buf
    
    def read(self, path, **params) -> bytes:
        with self.readbuf(path, **params) as fi:
            return fi.read()

    def writebuf(self, path: str, **params) -> BytesIO:
        path = self.expand_path(path)
        return open(path, 'wb')
    
    def write(self, path, data: bytes) -> bool:
        with self.writebuf(path) as fo:
            fo.write(data)

    def unlink(self, path: str) -> bool:
        path = self.expand_path(path)
        try:
            os.unlink(path)
        except OSError:
            return False
        return True

    def join(self, path: str, *path_segs: str) -> str:
        if path.startswith('file://'):
            path = path[7:]
        return 'file://' + os.path.join(path, *path_segs).replace(os.path.sep, '/')

    def mkdir(self, path: str, new_folder: str) -> bool:
        path = self.expand_path(path)
        os.makedirs(os.path.join(path, new_folder), exist_ok=True)
        return True

    def walk(self, base_path, name_pattern=''):
        base_path = self.join(self.expand_path(base_path))
        name_pattern = self.join(self.expand_path(name_pattern)) if name_pattern else ''
        yield from super().walk(base_path, name_pattern)

    def search(self, base_path, name_pattern = ''):
        base_path = self.join(self.expand_path(base_path))
        name_pattern = self.join(self.expand_path(name_pattern)) if name_pattern else ''
        yield from super().search(base_path, name_pattern)


class Hdf5Manager(StorageManager):
    """HDF5 Manager"""

    _lock = Lock()
    
    def __init__(self, storage_base: str) -> None:
        files = []
        if isinstance(storage_base, list):
            storage_base, *external_storage = storage_base
        else:
            external_storage = []
        
        for storage_parent in [storage_base, *external_storage]:
            files += glob.glob(os.path.join(storage_parent, '*.h5'))
        self.base = os.path.join(storage_base, 'blocks.h5')
        if self.base in files:
            files.remove(self.base)

        self.files = []
        for g in files:
            try:
                self.files.append(h5py.File(g, 'r'))
            except OSError:
                print('OSError while loading from', g)
            
        self._writable_file = None

        if not os.path.exists(self.base):
            try:
                self._writable_file = h5py.File(self.base, 'w')
            except:
                self.base = tempfile.mktemp('.h5')
                self._writable_file = h5py.File(self.base, 'w')
                
            self._writable_file.close()
            self._writable_file = None

    def stat(self, path: str) -> dict:
        parsed = urllib.parse.urlparse(path)
        data = self.read(parsed.netloc)
        if data:
            return {
                'name': path,
                'fullpath': path,
                'ctime': time.time(),
                'mtime': time.time(),
                'size': len(data),
                'type': 'file'
            }
            
    def _get_item_id(self, path:str) -> str:
        if '://' in path:
            path = path.split('://', 1)[1].split('/')[0]
        return path

    def write(self, path, data):
        """Write data from src to hdf5 file with specific item id

        :param src: bytes or io object to read bytes from
        :type src: IO | bytes
        :param item_id: item id
        :type item_id: str
        :return: True if success
        :rtype: bool
        """
        path = self._get_item_id(path)
        
        self._lock.acquire()
        if self._writable_file and self._writable_file.mode != 'r+':
            self._writable_file.close()
            self._writable_file = None

        if not self._writable_file:
            self._writable_file = h5py.File(self.base, 'r+')
        
        if isinstance(data, bytes):
            data = BytesIO(data)

        k = f'data/{path}'
        if k in self._writable_file:
            del self._writable_file[k]

        self._writable_file[k] = np.frombuffer(
            data.read(), dtype='uint8')

        self._writable_file.flush()
        self._lock.release()
        
        return True

    def read(self, path: str) -> bytes:
        """Read data from hdf5 file

        :param path: path string
        :type path: str
        :raises OSError: Not found
        :return: data
        :rtype: BytesIO
        """
        path = self._get_item_id(path)
        
        if not self._writable_file:
            self._writable_file = h5py.File(self.base, 'r')

        key = f'data/{path}'
        for block_file in [self._writable_file, *self.files]:
            if key in block_file:
                return block_file[key][:].tobytes()

        raise OSError(f"No matched ID: {path}")

    def listdir(self, path: str) -> list:
        return itertools.chain(*[list(s['data']) for s in self.files + [self._writable_file]])

    def walk(self, base_path: str, name_pattern='') -> Iterable[Tuple[str, List, List]]:
        if base_path.rstrip('/') == '':
            yield '/', [], self.listdir('/')

    def exists(self, path: str) -> bool:
        return self.read(path) is not None

    def join(self, base_path: str, *path_segs: str) -> str:
        return path_segs[-1]


class WebManager(StorageManager):

    class _ResponseStream(io.IOBase):

        def __init__(self, req, attempts=3, proxies=None, verify=False, timeout=60):
            super().__init__()
            self._pos = 0
            self._seekable = True
            self.req = req
            self.attempts = attempts
            self.verify = verify
            self.timeout = timeout
            self.proxies = proxies
            
            with self._urlopen() as f:
                self.content_length = int(f.headers.get('content-length', -1))
                if self.content_length < 0:
                    self._seekable = False
                self.st_size = self.content_length

        def seek(self, offset, whence=0):
            if not self.seekable():
                raise io.UnsupportedOperation
            if whence == 0:
                self._pos = 0
            elif whence == 1:
                pass
            elif whence == 2:
                self._pos = self.content_length
            self._pos += offset
            return self._pos

        def seekable(self, *args, **kwargs):
            return self._seekable

        def readable(self, *args, **kwargs):
            return not self.closed

        def writable(self, *args, **kwargs):
            return False

        def read(self, amt=-1):
            if self._pos >= self.content_length:
                return b""
            if amt < 0:
                end = self.content_length - 1
            else:
                end = min(self._pos + amt - 1, self.content_length - 1)
            byte_range = (self._pos, end)
            self._pos = end + 1

            with self._urlopen(byte_range) as f:
                return f.content

        def readall(self):
            return self.read(-1)

        def tell(self):
            return self._pos

        def _urlopen(self, byte_range=None):
            
            if byte_range:
                self.req.headers['Range'] = '{}-{}'.format(*byte_range)
            else:
                self.req.headers.pop('Range', '')
            
            ex = None

            for _ in range(self.attempts):
                with WebManager._session() as s:
                    # try:
                    return s.send(self.req, stream=True, proxies=self.proxies, verify=self.verify, timeout=self.timeout)
                    # except requests.exceptions.ConnectionError as e:
                        # ex = e
                        # time.sleep(1)
                    
            if ex:
                print('Read from', self.req.url, 'failed with exception', type(ex).__name__, ex)
                
    @staticmethod
    def _build_proxies(proxies):
        if proxies and not isinstance(proxies, dict):
            return {
                'http': proxies,
                'https': proxies
            }
        else:
            return proxies or {}
        
    @staticmethod
    def _session():
        return requests.session()
        
    def _build_request(self, url: str, method='GET', referer: str = '',
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

        if data and type(data) == bytes:
            headers['content-type'] = 'application/octet-stream'
        
        return requests.Request(url=url, method=method, headers=headers, 
                                data=data)

    def __init__(self, attempts=3, verify=False, timeout=30, seekable=False):
        """
        Args:
            attempts (int, optional): Maximal retries when fetching data from server. Defaults to 3.
            verify (bool, optional): Verify SSL Certs. Defaults to False.
            timeout (int, optional): Timeout in seconds. Defaults to 30.
        """
        self.attempts = attempts
        self.verify = verify
        self.timeout = timeout
        self.seekable = seekable

    def write(self, path: str, data: bytes, method = 'POST', proxy = None, **params) -> BytesIO:
        req = self._build_request(path, method, data=data, **params).prepare()
        with self._session() as s:
            resp = s.send(req, proxies=WebManager._build_proxies(proxy), verify=self.verify, timeout=self.timeout)
        if resp.status_code != 200:
            raise OSError(f'HTTP {resp.status_code}')
        return True
    
    def readbuf(self, path: str, proxy=None, **params) -> IOBase:
        req = self._build_request(path, **params).prepare()
        return WebManager._ResponseStream(req, self.attempts, proxies=WebManager._build_proxies(proxy), verify=self.verify, timeout=self.timeout)
        
    def read(self, path: str, proxy=None, **params) -> BytesIO:
        req = self._build_request(path, **params).prepare()
        with self._session() as s:        
            return BytesIO(s.send(
                req, proxies=WebManager._build_proxies(proxy),
                verify=self.verify, timeout=self.timeout
            ).content)
        
    def exists(self, path: str) -> bool:
        return requests.get(path).status_code == 200

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

    def walk(self, start_path, name_pattern=''):
        start_path = start_path.strip('/') + '/'
        service, path = start_path.split('/', 1)
        file_dirs = self.conn.listPath(service, path)

        files = [f.filename for f in file_dirs if not f.isDirectory]
        dirs = [f.filename for f in file_dirs if f.isDirectory and not f.filename.startswith(
            ('.', '@'))]
        yield start_path, dirs, files

        for d in dirs:
            yield from self.walk('/'.join([start_path.rstrip('/'), d]))

    def readbuf(self, path, **params):
        return self._opener.open(path)
    
    def read(self, path, **params):
        return self.readbuf(path, **params).read()

    def write(self, path, data, **params):
        conn, service, path = self._get_connection(path)
        conn.storeFile(service, path, BytesIO(data))
        
    def mkdir(self, path: str, new_folder: str) -> bool:
        conn, service, path = self._get_connection(path)
        return conn.createDirectory(service, path.rstrip('/') + '/' + new_folder)


class DataSchemeManager(StorageManager):
    """Handle with data: scheme
    """

    def exists(self, path: str) -> bool:
        return True

    def read(self, path, **params):
        with urllib.request.urlopen(path) as response:
            data = response.read()
        return data
    
    
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
       
    
def fragment_handlers():

    def handle_zip(buf, *inner_path):
        """Handle zip file"""
        zpath = '/'.join(inner_path)
        with zipfile.ZipFile(buf, 'r') as zip_file:
            return zip_file.open(zpath)

    def handle_pdf(buf, page, *_):
        """Get PNG data from PDF

        :param file: PDF path
        :type file: str
        :param page: page index, starting form 0
        :type page: int
        :return: PNG bytes in a BytesIO
        :rtype: BytesIO
        """
        
        def _pdf_image(file: str, page: int, **_) -> BytesIO:
            buf = BytesIO()
            page = int(page)

            img, = _pdf_convert(file, 120, first_page=page+1,
                                last_page=page+1, fmt='png') or [None]
            if img:
                img.save(buf, format='png')
                buf.seek(0)

            return buf

        filename = getattr(buf, 'name', '')
        temp = not not filename
        if temp:
            filename = tempfile.mktemp(suffix='.pdf')
            with open(filename, 'wb') as f:
                f.write(buf.read())
        
        buf = _pdf_image(filename, int(page))
        
        if temp:
            os.unlink(filename)
        
        return buf

    def handle_thumbnail(buf, width, height='', *_):
        """Get thumbnail for image"""
        from PIL import Image

        if not height:
            height = width
        width, height = int(width), int(height)

        im = Image.open(buf)
        im.thumbnail((width, height))

        buf = BytesIO()
        im.save(buf, 'JPEG')
        buf.seek(0)

        return buf
    
    return locals().items()


class Storage:
    
    def __init__(self) -> None:
        """Initialize storage
        """

        self._schema = {
            'smb': SMBManager(),
            'http': WebManager(),
            'https': WebManager(),
            'data': DataSchemeManager(),
            'hdf5': Hdf5Manager(config.storage),
            '': OSFileSystemManager(config.storage),
        }
        
        self.storage_proxy = config.storage_proxy
        if not self.storage_proxy:
            self.storage_proxy = {}
            
        if isinstance(self.storage_proxy, str):
            self.storage_proxy = {
                k: [config.storage_proxy] for k in ('hdf5', 'smb', 'file')
            }
            
        for k, v in self.storage_proxy.items():
            if isinstance(v, str):
                self.storage_proxy[k] = [v]

        self._fragment_handlers = {
            fh[7:]: func for fh, func in fragment_handlers() if fh.startswith('handle_')
        }

    def _get_managers(self, path) -> StorageManager:
        """Get storage manager according to scheme"""
        
        scheme = ''        
        if path in self._schema:
            scheme = path
        elif '://' in path:
            tmp = path.split('://', 1)[0]
            if tmp in self._schema:
                scheme = tmp
            
        if scheme in self.storage_proxy:
            for server in self.storage_proxy[scheme]:
                if server == 'local':
                    yield self._schema[scheme]
                else:
                    yield StorageProxyManager(server)
        else:
            yield self._schema[scheme]

    def register_fragment_handler(self, frag_name, func):
        """Register fragment handler

        Args:
            frag_name (str): fragment name
            func (function): handler function
        """
        self._fragment_handlers[frag_name] = func

    def register_scheme(self, scheme: str, mgr: StorageManager):
        """Register scheme

        Args:
            scheme (str): scheme name
            mgr (StorageManager): manager instance
        """
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
                    - hdf5://<item id>
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
        parsed = urllib.parse.urlparse(path.replace('__hash/', '#'))
        buf = None
        if '#' in path: path = path[:path.find('#')]

        for mgr in self._get_managers(parsed.scheme):
            try:
                buf = getattr(mgr, 'readbuf' if mode ==
                        'rb' else 'writebuf')(path, **params)
            except OSError as ex:
                print(ex)

        if not buf:
            raise OSError('Unable to open: ' + path)

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
            val = getattr(mgr, action)(path, **params)
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
        path = path.replace('__hash/', '#')
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
            ext = getattr(input_file, 'name', '').rsplit('.', 1)[-1][:4].lower()
            
        mimetype = self.get_mimetype(ext)
       
        if not file_size:
            file_size = getattr(path_or_io, 'st_size', 0)
        
        if not file_size:
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
            resp = Response(input_file.read(), 200, content_type=mimetype, direct_passthrough=True)
        resp.headers.add('Accept-Ranges', 'bytes')
        return resp
    
    def expand_path(self, path):
        return self._query_until('expand_path', path, self._schema[''].expand_path(path))
    
    def truncate_path(self, path):
        return self._query_until('truncate_path', path, self._schema[''].truncate_path(path))
    
    def expand_patterns(self, patterns: Union[list, str, tuple]):
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
                for mgr in self._get_managers(parent):
                    for pattern in mgr.search(parent, pattern):
                        patterns.append(pattern)

            if pattern.endswith('.zip') or pattern.endswith('.epub'):
                with zipfile.ZipFile(self.open(pattern, 'rb')) as zfile:
                    for item in zfile.filelist:
                        yield pattern + '#zip/' + item.filename
            else:
                yield pattern
                
    def fspath(self, base_file='test', *segs):
        return os.path.join(os.path.dirname(os.path.abspath(base_file)), *segs)
  
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
            
            resp = self.serve_file(path, ext, self.stat(path)['size'])
            resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp

        @storage_app.route('/<scheme>/<path:path>', methods=['PUT', 'POST'])
        def put_item(scheme, path):
            path, ext = Storage.get_schemed_path(scheme, path)
            if not request.data:
                return f'No data, ignored.'
            with self.open(path, 'wb') as fo:
                fo.write(request.data)
            return f'OK {path}'

        storage_app.debug = debug

        if storage_app.debug or '-d' in sys.argv:
            storage_app.run(host=host, port=port, debug=True)
        else:
            from waitress import serve
            serve(storage_app, host=host, port=port, threads=8, connection_limit=512, backlog=2048)

        return storage_app
  

instance = Storage()
