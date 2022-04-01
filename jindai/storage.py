"""数据存储访问"""

import glob
import os
import re
import tempfile
import time
from io import BytesIO
from threading import Lock
from typing import Tuple, Union
import zipfile
import h5py
import numpy as np
import requests
from pdf2image import convert_from_path as _pdf_convert

from .config import instance as config


class Hdf5Manager:
    """HDF5 Manager"""

    files = [h5py.File(g, 'r') for g in glob.glob(os.path.join(
        config.storage, '*.h5')) if not g.endswith('blocks.h5')]
    base = os.path.join(config.storage, 'blocks.h5')
    writable_file = None
    write_counter = 0
    _lock = Lock()

    def __enter__(self, *_):
        Hdf5Manager._lock.acquire()
        if not Hdf5Manager.writable_file or Hdf5Manager.writable_file.mode != 'r+':
            if Hdf5Manager.writable_file:
                Hdf5Manager.writable_file.close()
            Hdf5Manager.writable_file = h5py.File(Hdf5Manager.base, 'r+')
        Hdf5Manager.write_counter = 0
        return self

    def __exit__(self, *_):
        """Exit with block
        """
        Hdf5Manager.writable_file.flush()
        Hdf5Manager._lock.release()

    @staticmethod
    def write(src, item_id):
        """Write data from src to h5df file with specific item id

        Args:
            src (IO | bytes): bytes or io object to read bytes from
            item_id (str): item id

        Returns:
            bool: True if success
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

        Args:
            item_id (str): id string

        Raises:
            OSError: Not found

        Returns:
            bytes: data
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

        raise OSError(f"No matched ID found: {item_id}")

    def delete(self, item_id: str) -> None:
        """Delete item id

        Args:
            item_id (str): ID string
        """
        key = f'data/{item_id}'
        for block_file in [Hdf5Manager.writable_file] + Hdf5Manager.files:
            if key in block_file:
                del block_file[key]


class _Hdf5WriteBuffer(BytesIO):

    def __init__(self, item_id, initial_bytes=b''):
        self.item_id = item_id
        super().__init__(initial_bytes)

    def close(self):
        with Hdf5Manager():
            Hdf5Manager.write(self.getvalue(), self.item_id)
        super().close()


class _ZipWriteBuffer(BytesIO):

    def __init__(self, path, zfile):
        self.path = path
        self.zfile = zfile
        super().__init__()

    def close(self):
        super().close()
        with zipfile.ZipFile(self.path) as zip_file:
            zip_file.writestr(self.zfile, self.getvalue())


class _RequestBuffer(BytesIO):

    def __init__(self, url, method='POST', **params):
        self.req = _build_request(url, method, **params)
        super().__init__()

    def close(self):
        super().close()
        self.req.data = self.getvalue()
        requests.Session().send(self.req.prepare())


def _pdf_image(file, page, **_):
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

    Args:
        url (str): url
        referer (str, optional): referer url
        attempts (int, optional): max attempts

    Returns:
        Union[bytes, None]: response content or None if failed
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


def expand_path(path: Union[Tuple[str], str]):
    """
    扩展本地文件系统路径
    """
    if isinstance(path, tuple):
        path = os.path.sep.join([str(x) for x in path])

    if '://' in path:
        return path
    if '#' in path:
        path, _ = path.split('#', 1)

    path = path.replace('/', os.path.sep)

    allowed_locations = [
        os.path.join(tempfile.gettempdir(), tempfile.gettempprefix()),
        config.storage
    ]

    if not path.startswith(tuple(allowed_locations)):
        if path.startswith((os.path.altsep or os.path.sep, os.path.sep)):
            path = path[1:]
        path = os.path.join(config.storage, path)

    return path


def expand_patterns(patterns: Union[list, str]):
    """
    读取文件（包括压缩包内容）或网址，其中文件名可以使用 */? 通配符，网址可以使用 {num1-num2} 形式给定迭代范围
    Returns:
        Tuple[IO, str]: IO 为内容，str 为文件名或网址
    """

    if isinstance(patterns, str):
        patterns = patterns.split('\n')

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
            pattern = expand_path(pattern)
            for path in glob.glob(pattern):
                if path.endswith('.zip') or path.endswith('.epub'):
                    with zipfile.ZipFile(path, 'r') as zip_file:
                        for zipped_item in zip_file.filelist:
                            yield path + '#' + zipped_item.filename
                elif os.path.isdir(path):
                    patterns.append(path + '/*')
                else:
                    yield path


def safe_open(path: str, mode='rb', **params):
    """
    打开各种格式的文件。
    Args:
        path (str | tuple[str]): 可以是一个 tuple，也可以是字符串，为 tuple 时自动拼成字符串
            除了常用的绝对和相对路径之外，还支持如下格式（按匹配优先级顺序）：
            以 http:// 或 https:// 开头：时 params 支持 proxies, headers，mode 必须为 rb；
            hdf5:// 开头：使用 hdf5Manager 读取或写入，mode 必须为 rb 或 wb；
            包含 #zip/ ：将 # 前的作为 ZIP 压缩包文件路径，zip/ 后的作为包内的路径；
            包含 #pdf/png ：将 # 前的作为 PDF 文件路径，pdf/png: 后的作为页号（从 0 开始），读取的是图像文件；
            其余视为文件系统路径，其时 / 自动替换为对应平台的 / 或 \\。

        mode (str): r, rb, w, wb, a, rb+ 等，取决于 path 的类型

        allowed_locations (list): 允许的文件系统路径，默认包含临时文件（包含默认临时文件前缀）和配置文件中所设置的 storage 路径
            若不在允许的列表中，则将其视为一个相对于 storage 的路径
    """

    if path.startswith(('http://', 'https://')):
        assert mode in ('rb', 'wb')
        if mode == 'rb':
            return BytesIO(_try_download(path, **params))

        return _RequestBuffer(path, **params)

    if path.startswith('hdf5://'):
        assert mode in ('rb', 'wb')
        item_id = path.split('://', 1)[1]
        if mode == 'rb':
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

    elif '#pdf/png:' in path:
        assert mode == 'rb'
        _, page = path.split('#pdf/png:', 1)
        return _pdf_image(fpath, int(page), **params)

    else:
        buf = open(fpath, mode)
        buf.st_size = os.stat(fpath).st_size
        return buf


def truncate_path(path, base=None):
    """Truncate path into base directory."""
    if base is None:
        base = config.storage
    if not base.endswith(os.path.sep):
        base += os.path.sep
    if path.startswith(base):
        return path[len(base):]
    return path
