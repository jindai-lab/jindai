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
        