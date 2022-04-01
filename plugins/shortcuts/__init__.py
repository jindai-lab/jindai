import os

from jindai import Plugin
from jindai.helpers import rest, serve_file
from jindai.models import F, Meta, parser


class Shortcuts(Plugin):

    def __init__(self, app):
        super().__init__(app)
        self.read_shortcuts()

        @app.route('/api/plugins/shortcuts', methods=['GET', 'POST'])
        @rest()
        def shortcuts(key='', value=''):
            if not key:
                return self.read_shortcuts()
            else:
                r = Meta.first(F.shortcuts.exists(1)) or Meta()
                s = self.read_shortcuts()
                if key in s and value == '':
                    del s[key]
                else:
                    s[key] = value
                r.shortcuts = s
                r.save()
                return True

        @app.route('/api/plugins/shortcuts.html')
        def _shortcuts_html():
            return serve_file(os.path.join(os.path.dirname(__file__), 'shortcuts.html'))

        @app.route('/api/plugins/shortcuts-jquery.min.js')
        def _jquery_js():
            return serve_file(os.path.join(os.path.dirname(__file__), 'jquery.min.js'))

    def read_shortcuts(self):
        r = Meta.first(F.shortcuts.exists(1)) or Meta()
        shortcuts = r.shortcuts or {}
        for k, v in shortcuts.items():
            if k.startswith(':'):
                parser.set_shortcut(k[1:], v)
        return shortcuts

    def get_callbacks(self):
        return ['css']

    def css_callback(self):
        return ','.join(
            ['.gallery-description .t_' + v for k, v in self.read_shortcuts().items() if not k.startswith(':')]) + '{ color:orange!important; }' + '''
            #shortcuts-input #hints {
                list-style: none;
                text-align: left;
                width: 50%;
                margin: 20px auto;
            }
            '''
    