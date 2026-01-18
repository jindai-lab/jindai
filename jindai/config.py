"""Config file"""
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

import yaml


class DictObject(dict):

    def __init__(self, data : Dict | None = None):
        super().__init__(data or {})
        if '_id' in self:
            self['_id'] = UUID(self['id'])
        
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        if name == 'id' and '_id' in self:
            return self['_id']
        elif name in self.__dict__:
            return object.__getattribute__(self, name)
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'id':
            value = UUID(value)
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


if '-c' in sys.argv:
    config_arg = sys.argv.index('-c')
    config_file = sys.argv[config_arg + 1]
    sys.argv.pop(config_arg + 1)
    sys.argv.pop(config_arg)
    os.environ['CONFIG_FILE'] = config_file


class ConfigObject(DictObject):
    """Accessing config file"""

    def __init__(self, filename=None):
        """Load config file

        :param config_file: Path for config file,
            None to load from env variable CONFIG_FILE, and config.yaml in pwd
        :type config_file: str, optional
        """

        orig = {
            'mongo': 'localhost:27017',
            'mongoDbName': 'hamster',
            'rootpath': '',
            'storage': 'storage',
            'file_serve': {},
            'secret_key': '!!seckey',
            'concurrent': 3,
            'plugins': ['*'],
            'allowed_ips': {},
            'ui_proxy': '',
            'port': 8370,
            'debug': False,
            'constants': {}
        }
        filename = filename or os.environ.get('CONFIG_FILE', 'config.yaml')

        if not os.path.exists(filename):
            print('Config file not found:', filename)
        else:
            with open(filename, 'r', encoding='utf-8') as fin:
                orig.update(**yaml.safe_load(fin))

        self._filename = filename

        if orig['rootpath'] == '':
            orig['rootpath'] = str(
                Path(os.path.abspath(__file__)).parent.parent.absolute())

        super().__init__(orig)

    def save(self, filename: str = '') -> None:
        """Save config file"""
        filename = filename or self._filename
        with open(filename, 'w', encoding='utf-8') as fout:
            yaml.dump(self, fout)


instance = ConfigObject()