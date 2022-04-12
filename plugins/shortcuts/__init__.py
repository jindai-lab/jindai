"""Shortcut plugin"""
import os
from PyMongoWrapper import F

from jindai import Plugin, parser
from jindai.helpers import rest, serve_file
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
            if not key:
                return self.read_shortcuts()

            meta = Meta.first(F.shortcuts.exists(1)) or Meta()
            shortcuts = self.read_shortcuts()
            if key in shortcuts and value == '':
                del shortcuts[key]
            else:
                shortcuts[key] = value
            meta.shortcuts = shortcuts
            meta.save()
            return True

        @app.route('/api/plugins/shortcuts.html')
        @rest()
        def _shortcuts_html():
            return serve_file(os.path.join(os.path.dirname(__file__), 'shortcuts.html'))

        @app.route('/api/plugins/shortcuts-jquery.min.js')
        def _shortcuts_jquery_js():
            return serve_file(os.path.join(os.path.dirname(__file__), 'jquery.min.js'))

    def read_shortcuts(self):
        """Read shortcuts from Meta settings"""
        shortcuts = (Meta.first(F.shortcuts.exists(1))
                     or Meta()).shortcuts or {}
        for key, val in shortcuts.items():
            if key.startswith(':'):
                parser.set_shortcut(key[1:], val)
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
