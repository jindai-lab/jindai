import json
import os
from flask import request, Response, redirect, jsonify
from gallery import arg, tools, Plugin

sp = os.path.join(os.path.dirname(__file__), 'shortcuts.json')


def read_shortcuts():
    return json.load(open(sp, 'r', encoding='utf-8'))

class Shortcuts(Plugin):

    def __init__(self, app):
        self.shortcut_pages = {
            'r': '?query=~@jav&order=JSON__{"keys":["random"]}',
            'f': '?query=:fav',
            'faces': "?query=_faces!=[]"
        }
        
        @app.route('/api/gallery/shortcuts', methods=['GET', 'POST'])
        def shortcuts():
            if request.method == 'GET':
                return Response(open(sp).read(), content_type='text/json')
            else:
                with open(sp, 'w') as fo:
                    fo.write(arg('data'))
                return 'OK'

    def get_tools(self):
        return ['routine']

    def get_callbacks(self):
        return ['css']

    def css_callback(self, ctx):
        return ','.join(
            ['.v-card .t_' + v for k, v in json.load(open(sp)).items() if not v.startswith('(') and '&' not in v]) + '{ color:orange!important; }' + '''
            #shortcuts-input #hints {
                list-style: none;
                text-align: left;
                width: 50%;
                margin: 20px auto;
            }
            '''
            
    def routine(self, ctx, routine_name):
        p = read_shortcuts()
        v = p['::routine-' + routine_name]
        for l in v.split('&'):
            l = l.split(' ')
            tools[l[0]].run_tool(ctx, *l)
    
    def get_special_pages(self):
        return list(self.shortcut_pages.keys())
    
    def special_page(self, aggregate, params, *args, **kwargs):    
        p = params['post']
        if p in self.shortcut_pages:
            return jsonify({'redirect': self.shortcut_pages[p]})
