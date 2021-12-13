from flask import request
from gallery import Plugin, rest
from models import F, Meta


def read_shortcuts():
    r = Meta.first(F.gallery_shortcuts.exists(1)) or Meta()
    if hasattr(r, 'gallery_shortcuts'): return r.gallery_shortcuts
    else: return {}

class Shortcuts(Plugin):

    def __init__(self, app):

        @app.route('/api/gallery/shortcuts', methods=['GET', 'POST'])
        @rest()
        def shortcuts(key='', value=''):
            if request.method == 'GET':
                return read_shortcuts()
            else:
                r = Meta.first(F.gallery_shortcuts.exists(1)) or Meta()
                s = read_shortcuts()
                s[key] = value
                setattr(r, 'gallery_shortcuts', s)
                r.save()
                return 'OK'

    def get_callbacks(self):
        return ['css']

    def css_callback(self):
        return ','.join(
            ['.v-card .t_' + v for k, v in read_shortcuts().items() if not v.startswith('(') and '&' not in v]) + '{ color:orange!important; }' + '''
            #shortcuts-input #hints {
                list-style: none;
                text-align: left;
                width: 50%;
                margin: 20px auto;
            }
            '''
    