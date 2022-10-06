"""HDF5 manager"""
import glob
import itertools
import os
import re
import tempfile
import time
import urllib
from threading import Lock
from typing import Iterable, List, Tuple

import h5py
import numpy as np

from jindai.storage import StorageManager
from jindai import storage, config, Plugin


class Hdf5Manager(StorageManager):
    """HDF5 Manager"""

    _lock = Lock()

    def __init__(self, storage_base: str, quota='40G') -> None:
        files = []
        self.quota = self.parse_size(quota)
        if isinstance(storage_base, list):
            storage_base, *external_storage = storage_base
        else:
            external_storage = []

        self.storage_base = storage_base

        for storage_parent in [storage_base, *external_storage]:
            files += glob.glob(os.path.join(storage_parent, '*.h5'))

        self._num = 0
        self._next_file()

        if self._filename in files:
            files.remove(self._filename)

        self.files = []
        for g in files:
            try:
                self.files.append(h5py.File(g, 'r'))
            except OSError:
                self.dprint('OSError while loading from', g)

        self._writable_file = None

        if not os.path.exists(self._filename):
            try:
                self._writable_file = h5py.File(self._filename, 'w')
            except:
                self._filename = tempfile.mktemp('.h5')
                self._writable_file = h5py.File(self._filename, 'w')

            self._writable_file.close()
            self._writable_file = None

        self._written_size = os.stat(self._filename).st_size

    def _next_file(self):
        while True:
            self._num += 1
            self._filename = os.path.join(
                self.storage_base, f'blocks{self._num}.h5')
            if os.path.exists(self._filename) and os.stat(self._filename).st_size > self.quota * 0.99:
                continue
            break

    def parse_size(self, size: str) -> int:
        m = re.match(r'(\d+)([gGmMkK])?B?', size)
        if m:
            size = int(m.group(1))
            if m.group(2):
                size <<= {
                    'g': 30,
                    'm': 20,
                    'k': 10
                }.get(m.group(2).lower(), 1)
        return size

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

    def _get_item_id(self, path: str) -> str:
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

        if not isinstance(data, bytes):
            data = data.read()
        data = np.frombuffer(data, dtype='uint8')
                    
        self._lock.acquire()

        if self._written_size + len(data) > self.quota:
            if self._writable_file:
                self._writable_file.close()
                self._writable_file = None
            self._next_file()
            self._written_size = 0

        if self._writable_file and self._writable_file.mode != 'r+':
            self._writable_file.close()
            self._writable_file = None

        if not self._writable_file:
            self._writable_file = h5py.File(self._filename, 'r+')

        k = f'data/{path}'
        if k in self._writable_file:
            del self._writable_file[k]

        self._writable_file[k] = data

        self._writable_file.flush()
        self._written_size += len(data)

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
            self._writable_file = h5py.File(self._filename, 'r')

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



class Hdf5ManagerPlugin(Plugin):
    
    def __init__(self, pmanager, storage_base=None, **conf) -> None:
        super().__init__(pmanager, **conf)
        storage.register_scheme('hdf5', Hdf5Manager(storage_base or config.storage))
