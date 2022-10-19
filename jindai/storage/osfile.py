"""Local OS File Storage"""
import os
import shutil
import tempfile
from io import BytesIO
from typing import Tuple, Union

from .storage import StorageManager


class OSFileSystemManager(StorageManager):
    """Storage manager for local file system"""

    def __init__(self, base: Union[str, list] = None) -> None:
        allowed_locations = []
        if isinstance(base, list):
            base, *allowed_locations = [b for b in base if '://' not in b]

        allowed_locations = [
            (loc + os.path.sep) if not loc.endswith(os.path.sep) else loc
            for loc in allowed_locations
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
            for i, seg in enumerate(segs):
                if '*' in seg:
                    break
            parent_path = '/'.join(segs[:i]) + '/'
            suffix_path = '/'.join(segs[i:])
            path = parent_path

        path = path.replace('/', os.path.sep)  # path is now os-specific

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
        if path.startswith('file://'):
            path = path[7:]
        path = path.replace(os.path.sep, '/')
        for base in self.allowed_locations:
            if path.startswith(base.replace(os.path.sep, '/')):
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

    def move(self, src: str, dst: str) -> bool:
        src, dst = self.expand_path(src), self.expand_path(dst)
        return shutil.move(src, dst)

    def readbuf(self, path: str, **params) -> BytesIO:
        path = self.expand_path(path)
        buf = open(path, 'rb')
        return buf

    def read(self, path, **params) -> bytes:
        with self.readbuf(path, **params) as finp:
            return finp.read()

    def writebuf(self, path: str, **params) -> BytesIO:
        path = self.expand_path(path)
        return open(path, 'wb')

    def write(self, path, data: bytes) -> bool:
        with self.writebuf(path) as fout:
            fout.write(data)

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
        name_pattern = self.join(self.expand_path(
            name_pattern)) if name_pattern else ''
        yield from super().walk(base_path, name_pattern)

    def search(self, base_path, name_pattern=''):
        base_path = self.join(self.expand_path(base_path))
        name_pattern = self.join(self.expand_path(
            name_pattern)) if name_pattern else ''
        yield from super().search(base_path, name_pattern)
