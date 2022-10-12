"""Base storage classes"""
import sys
import time
from io import BytesIO
from typing import Tuple, List
from fnmatch import fnmatch


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

    def write(self, path, data: bytes) -> bool:
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
        pattern_segs = '/'.join(match_pattern.split('/')
                                [:base_path.count('/')+1])
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

    def move(self, path: str, dst: str) -> bool:
        """Move path name"""
        return False

    def dprint(self, *args):
        """Print debug info"""
        print(*args, file=sys.stderr)
        
    def __init__(self, *_) -> None:
        pass


class WriteBuffer(BytesIO):

    def __init__(self, writer: StorageManager, path: str, **write_params: dict) -> None:
        super().__init__()
        self._writer = writer
        self._params = write_params or {}
        self._params['path'] = path

    def close(self):
        self._writer.write(data=self.getvalue(), **self._params)
        super().close()
