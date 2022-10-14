from collections import defaultdict
import threading
import time
from typing import Any
from bson import ObjectId


class DictObject(dict):

    def __init__(self, data : dict = None):
        super().__init__(data or {})
        if '_id' in self:
            self['_id'] = ObjectId(self['_id'])
        
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        if name == 'id' and '_id' in self:
            return self['_id']
        elif name in self.__dict__:
            return object.__getattribute__(self, name)
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'id':
            name = '_id'
        if name == '_id':
            value = ObjectId(value)
        self[name] = value
        
        
class CacheDict:
    
    def __init__(self, ttl=300) -> None:
        self.ttl = ttl
        self._orig = defaultdict(DictObject)
        self._lock = threading.Lock()
        
    def __getitem__(self, key):
        self._orig[key].access = time.time()
        return self._orig[key].value
    
    def __setitem__(self, key, value):
        self.clear()
        self._orig[key].value = value
        self._orig[key].access = time.time()
    
    def clear(self):
        to_pop = []
        for key, cached in self._orig.items():
            if time.time() - cached.access > self.ttl:
                to_pop.append(key)
        if to_pop:
            with self._lock:
                for key in to_pop:
                    self._orig.pop(key, None)
