"""Config file"""
import os
import sys
from pathlib import Path
from typing import Any
import yaml


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
            'file_serve': {},
            'secret_key': '!!seckey',
            'concurrent': 3,
            'plugins': ['*'],
            'allowed_ips': {},
            'ui_proxy': ''
        }
        if config_file is None:
            if '-c' in sys.argv:
                config_arg = sys.argv.index('-c')
                config_file = sys.argv[config_arg + 1]
                sys.argv.pop(config_arg + 1)
                sys.argv.pop(config_arg)
            else:
                for config_file in [
                    os.environ.get('CONFIG_FILE', ''),
                    'config.yaml'
                ]:
                    if os.path.exists(config_file) and os.path.isfile(config_file):
                        break

        with open(config_file, 'r', encoding='utf-8') as fin:
            self._orig.update(**yaml.safe_load(fin))

        self._filename = config_file

        if self._orig['rootpath'] == '':
            self._orig['rootpath'] = str(
                Path(os.path.abspath(__file__)).parent.parent.absolute())
        if not self._orig['storage'].startswith(('/', r'\\')):
            self._orig['storage'] = os.path.join(
                self._orig['rootpath'], self._orig['storage'])

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
