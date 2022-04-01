from jindai import Plugin
from jindai.helpers import serve_file
import os, re

class Blockly(Plugin):
    
    def __init__(self, app, **config) -> None:
        super().__init__(app, **config)

        # BLOCKLY UI
        @app.route('/api/blockly/<path:p>')
        @app.route('/api/blockly/')
        def blockly_index(p='index.html'):
            parent = os.path.dirname(__file__)
            if re.match(r'^[0-9a-f]{24}$', p):
                p = 'index.html'
            if p == 'index.html':
                if not os.path.exists(os.path.join(parent, 'blockly')):
                    return serve_file(os.path.join(parent, 'README'))
                
            for fp in (os.path.join(parent, p), os.path.join(parent, 'blockly', p)):
                if os.path.exists(fp) and os.path.isfile(fp):
                    return serve_file(fp)
            return '', 404
