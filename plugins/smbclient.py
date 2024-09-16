"""SMB Storage"""

import urllib
import time
from io import BytesIO
from smb.SMBConnection import SMBConnection
from smb.SMBHandler import SMBHandler

from jindai.storage import StorageManager
from jindai import storage, Plugin


class SMBManager(StorageManager):
    """SMB Client
    """

    def __init__(self, *_) -> None:
        self._connections = {}
        self._last_active = {}
        self._ttl = 60
        self._opener = urllib.request.build_opener(SMBHandler)

    def _get_connection(self, path):
        parsed = urllib.parse.urlparse(path)
        if parsed.netloc not in self._connections or self._last_active.get(parsed.netloc, 0) < time.time() - self._ttl:
            conn = SMBConnection(parsed.username, parsed.password, '', '')
            conn.connect(parsed.hostname)
            self._connections[parsed.netloc] = conn

        self._last_active[parsed.netloc] = time.time()
        segs = parsed.path.lstrip('/').split('/', 1)
        return self._connections[parsed.netloc], segs[0], segs[1]

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

    def walk(self, base_path, match_pattern=''):
        conn, service, path = self._get_connection(base_path)
        file_dirs = conn.listPath(service, path)

        files = [f.filename for f in file_dirs if not f.isDirectory]
        dirs = [f.filename for f in file_dirs if f.isDirectory and not f.filename.startswith(
            ('.', '@'))]
        yield base_path, dirs, files

        for dir in dirs:
            yield from self.walk('/'.join([base_path.rstrip('/'), dir]))

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


class SMBClientPlugin(Plugin):
    
    def __init__(self, pmanager, **conf) -> None:
        super().__init__(pmanager, **conf)
        storage.register_scheme('smb', SMBManager)
