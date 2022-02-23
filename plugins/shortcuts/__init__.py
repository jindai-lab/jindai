from flask import request
from helpers import rest
from plugin import Plugin
from models import F, Meta


class Shortcuts(Plugin):

    def __init__(self, app):
        super().__init__(app)
        
        @app.route('/api/plugins/shortcuts', methods=['GET', 'POST'])
        @rest()
        def shortcuts(key='', value=''):
            if not key:
                return self.read_shortcuts()
            else:
                r = Meta.first(F.shortcuts.exists(1)) or Meta()
                s = self.read_shortcuts()
                s[key] = value
                r.shortcuts = s
                r.save()
                return 'OK'

    def read_shortcuts(self):
        r = Meta.first(F.shortcuts.exists(1)) or Meta()
        return r.shortcuts or {}

    def get_callbacks(self):
        return ['css']

    def css_callback(self):
        return ','.join(
            ['.v-card .t_' + v for k, v in self.read_shortcuts().items() if not v.startswith('(') and '&' not in v]) + '{ color:orange!important; }' + '''
            #shortcuts-input #hints {
                list-style: none;
                text-align: left;
                width: 50%;
                margin: 20px auto;
            }
            '''
    