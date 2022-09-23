from typing import Any


class DictObject(dict):

    def __init__(self, data : dict = None):
        super().__init__(data or {})

    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        elif name in self.__dict__:
            return object.__getattribute__(self, name)
        
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
        