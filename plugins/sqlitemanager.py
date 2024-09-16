from datetime import datetime
from itertools import chain
import threading
from time import sleep, time
import sqlite3
import glob
import os

from jindai import storage, config, Plugin
from jindai.storage import StorageManager


class _WrappedCursor:
    
    def __init__(self, conn) -> None:
        self._conn = conn
        self._cursor = conn.cursor()
        
    def __enter__(self, *_):
        return self._cursor
    
    def __exit__(self, *_):
        self._cursor.close()
        self._conn.commit()
        self._conn.close()


class SqliteSingleAccessor:
    
    def __init__(self, file) -> None:
        self._file = file
        self._last_write = 0
        with self.cursor() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS data (id TEXT PRIMARY KEY, bytes BLOB, updated TIMESTAMP)""")
        
    def cursor(self):
        _conn = sqlite3.connect(self._file)
        return _WrappedCursor(_conn)
        
    def write(self, key, buf):
        self._last_write = time()
        with self.cursor() as cursor:
            cursor.execute("""INSERT OR REPLACE INTO data VALUES (?, ?, ?)""", (key, buf, datetime.utcnow()))
        
    def read(self, key):
        with self.cursor() as cursor:
            for (buf,) in cursor.execute("""SELECT bytes FROM data WHERE id = ?""", (key,)):
                return buf
            
    def delete(self, key):
        with self.cursor() as cursor:
            cursor.execute("DELETE FROM data WHERE id = ?", (key,))
        
    def keys(self):
        with self.cursor() as cursor:
            for (key,) in cursor.execute("""SELECT id FROM data"""):
                yield key
                
    def stat(self, key):
        with self.cursor() as cursor:
            for (data, ctime) in cursor.execute("""SELECT bytes, updated FROM data WHERE id = ?""", (key,)):
                return {
                    'name': key,
                    'fullpath': key,
                    'ctime': ctime,
                    'mtime': ctime,
                    'size': len(data),
                    'type': 'file'
                }
        
        
class SqliteManager(StorageManager):
    
    def __init__(self, storage_base) -> None:
        super().__init__()
        self.dbs = []
        for base in storage_base:
            if '://' not in base:
                base = base
                self.dbs += [SqliteSingleAccessor(f) for f in glob.glob(os.path.join(base, 'sblobs*.db'))]
        if not storage_base:
            print('Please specify local storage base in `file` section first.')
            return
        if not self.dbs:
            self.dbs = [SqliteSingleAccessor(os.path.join(storage_base[0], 'sblobs1.db'))]

    def _get_item_id(self, path: str) -> str:
        if '://' in path:
            path = path.split('://', 1)[1].split('/')[0]
        return path

    def read(self, path: str, **params) -> bytes:
        path = self._get_item_id(path)
        for d in self.dbs:
            buf = d.read(path)
            if buf:
                return buf
        
        raise OSError(f"No item named {path}")
    
    def write(self, path, data: bytes) -> bool:
        path = self._get_item_id(path)
        self.dbs[-1].write(path, data)
        return True
    
    def listdir(self, path: str) -> list:
        path = self._get_item_id(path)
        if path != '':
            return []
        return list(chain(*[d.keys() for d in self.dbs]))
    
    def stat(self, path: str) -> dict:
        path = self._get_item_id(path)
        for d in self.dbs:
            data = d.stat(path)
            if data:
                return data

    def exists(self, path: str) -> bool:
        return self.stat(path) is not None
        
    def unlink(self, path: str) -> bool:
        path = self._get_item_id(path)
        for d in self.dbs:
            flag = d.delete(path)
            if flag:
                return True
        return False
    

class SqliteManagerPlugin(Plugin):
    
    def __init__(self, pmanager, **conf) -> None:
        super().__init__(pmanager, **conf)
        mgr = SqliteManager(config.storage['file'])
        storage.register_scheme('sqlite', lambda *_: mgr)
