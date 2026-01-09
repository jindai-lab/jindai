"""Config file"""
import os
import sys
from pathlib import Path
import yaml
from .common import DictObject


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
print(instance)