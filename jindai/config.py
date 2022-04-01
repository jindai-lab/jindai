import os
from pathlib import Path
import sys
import yaml


class ConfigObject:

    def __init__(self, config_file=None):
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
            for config_file in [
                os.environ.get('CONFIG_FILE', ''),
                'config.yaml'
            ]:
                if os.path.exists(config_file) and os.path.isfile(config_file):
                    break

        with open(config_file, 'r') as fin:
            self._orig.update(**yaml.safe_load(fin))
        if self._orig['rootpath'] == '':
            self._orig['rootpath'] = str(Path(os.path.abspath(__file__)).parent.parent.absolute())
        if not self._orig['storage'].startswith('/'):
            self._orig['storage'] = os.path.join(self._orig['rootpath'], self._orig['storage'])
    
    def __getattr__(self, attr):
        return self._orig.get(attr)


sys.modules[__name__] = ConfigObject()
