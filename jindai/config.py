"""配置文件"""
import os
from pathlib import Path
import yaml


class ConfigObject:
    """访问配置文件"""

    def __init__(self, config_file=None):
        """加载配置文件

        :param config_file: 配置文件路径，默认设为 None 时从 CONFIG_FILE 环境变量读取，若不存在则从当前工作文件夹 config.yaml
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
            for config_file in [
                os.environ.get('CONFIG_FILE', ''),
                'config.yaml'
            ]:
                if os.path.exists(config_file) and os.path.isfile(config_file):
                    break

        with open(config_file, 'r', encoding='utf-8') as fin:
            self._orig.update(**yaml.safe_load(fin))
        if self._orig['rootpath'] == '':
            self._orig['rootpath'] = str(
                Path(os.path.abspath(__file__)).parent.parent.absolute())
        if not self._orig['storage'].startswith('/'):
            self._orig['storage'] = os.path.join(
                self._orig['rootpath'], self._orig['storage'])

    def __getattr__(self, attr):
        return self._orig.get(attr)


instance = ConfigObject()
