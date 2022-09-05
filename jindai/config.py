"""Config file"""
import os
import sys
from pathlib import Path
from typing import Any
import yaml


if '-c' in sys.argv:
    config_arg = sys.argv.index('-c')
    config_file = sys.argv[config_arg + 1]
    sys.argv.pop(config_arg + 1)
    sys.argv.pop(config_arg)
    os.environ['CONFIG_FILE'] = config_file


class ConfigObject:
    """Accessing config file"""

    def __init__(self, config_file=None):
        """Load config file

        :param config_file: Path for config file,
            None to load from env variable CONFIG_FILE, and config.yaml in pwd
        :type config_file: str, optional
        """
        self._orig = {
            'mongo': 'localhost:27017',
            'mongoDbName': 'hamster',
            'rootpath': '',
            'storage': 'storage',
            'external_storage': {},
            'file_serve': {},
            'secret_key': '!!seckey',
            'concurrent': 3,
            'plugins': ['*'],
            'allowed_ips': {},
            'ui_proxy': '',
            'port': 8370,
        }
        config_file = os.environ.get('CONFIG_FILE', 'config.yaml')
        
        if not os.path.exists(config_file):
            print('Config file not found:', config_file)
            exit(255)

        with open(config_file, 'r', encoding='utf-8') as fin:
            self._orig.update(**yaml.safe_load(fin))

        self._filename = config_file

        if self._orig['rootpath'] == '':
            self._orig['rootpath'] = str(
                Path(os.path.abspath(__file__)).parent.parent.absolute())
        if not isinstance(self._orig['storage'], list):
            self._orig['storage'] = [self._orig['storage']]
        self._orig['storage'] = [os.path.join(self._orig['rootpath'], p) for p in self._orig['storage']]

    def __getattr__(self, __name: str):
        if __name in self.__dict__:
            return object.__getattribute__(self, __name)
        return self._orig.get(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name.startswith('_') or __name in self.__dict__:
            object.__setattr__(self, __name, __value)
        else:
            self._orig[__name] = __value

    def save(self, filename: str = '') -> None:
        """Save config file"""
        filename = filename or self._filename
        with open(filename, 'w', encoding='utf-8') as fout:
            yaml.dump(self._orig, fout)


instance = ConfigObject()
