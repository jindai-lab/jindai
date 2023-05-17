"""Shortcut plugin"""
import os
from PyMongoWrapper import F

from jindai import Plugin, parser, storage
from jindai.helpers import rest
from jindai.models import Meta


class Shortcuts(Plugin):
    """Shortcut plugin"""

    def __init__(self, pmanager):
        super().__init__(pmanager)
        self.read_shortcuts()
        self.register_callback('css', self.css_callback)
        app = self.pmanager.app

        @app.route('/api/plugins/shortcuts', methods=['GET', 'POST'])
        @rest()
        def shortcuts(key='', value=''):
            shortcuts = self.read_shortcuts()
            if not key:
                return [{
                    'name': key,
                    'expr': val
                } for key, val in shortcuts.items()]

            meta = Meta.first(F.shortcuts.exists(1)) or Meta()
            if value == '':
                shortcuts.pop(key, '')
            else:
                shortcuts[key] = value
            shortcuts.pop('', '')
            meta.shortcuts = shortcuts
            meta.save()
            return True
        
        parser.shortcuts = {}
        self.read_shortcuts()
        
    def read_shortcuts(self):
        """Read shortcuts from Meta settings"""
        shortcuts = (Meta.first(F.shortcuts.exists(1))
                     or Meta()).shortcuts or {}
        for key, val in shortcuts.items():
            if key.startswith(':'):
                try:
                    parser.set_shortcut(key[1:], val)
                except:
                    print('failed to set shortcut', key[1:], val)
        return shortcuts

    def css_callback(self):
        """Responding callback"""
        return ','.join(
            ['.gallery-description .t_' + v
             for k, v in self.read_shortcuts().items()
             if not k.startswith(':')]) + '{ color:orange!important; }' + '''
            #shortcuts-input #hints {
                list-style: none;
                text-align: left;
                width: 50%;
                margin: 20px auto;
            }
            '''
