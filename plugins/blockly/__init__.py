"""Blockly UI"""

import os
import re

from jindai import Plugin
from jindai.helpers import serve_file


class Blockly(Plugin):
    """Blockly UI Plugin"""

    def __init__(self, app, **config) -> None:
        super().__init__(app, **config)

        # BLOCKLY UI
        @app.route('/api/blockly/<path:p>')
        @app.route('/api/blockly/')
        def blockly_index(path='index.html'):
            parent = os.path.dirname(__file__)
            if re.match(r'^[0-9a-f]{24}$', path):
                path = 'index.html'
            if path == 'index.html':
                if not os.path.exists(os.path.join(parent, 'blockly')):
                    return serve_file(os.path.join(parent, 'README'))

            for full_path in (os.path.join(parent, path), os.path.join(parent, 'blockly', path)):
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    return serve_file(full_path)
            return '', 404
