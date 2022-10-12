"""OneDrive storage plugin"""

import encodings
from io import BytesIO
from jindai import storage, config
from jindai.helpers import safe_import
from jindai.plugin import Plugin
from jindai.storage import StorageManager
from dateutil.parser import parse as parse_time
from collections import defaultdict
import urllib
import os

safe_import('msal')
safe_import('msdrive', 'onedrive-sharepoint-python-sdk')
import msal
from msdrive import OneDrive


if not config.onedrive:
    print('Please specify onedrive section in configuration file')
    config.onedrive = defaultdict(str)


class OneDriveAuthenticator:
    
    CLIENT_ID = config.onedrive['clientId']
    CLIENT_SECRET = config.onedrive['clientSecret']
    GRAPH_USER_SCOPES = ['files.read', 'files.read.all']
    REDIRECT_TO = config.onedrive['server'] + '/api/plugins/onedrive/auth'
    AUTHORITY = 'https://login.microsoftonline.com/common'
    CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'onedrive_token_cache'))
    
    def __init__(self) -> None:
        self.cache = msal.SerializableTokenCache()

        if os.path.exists(OneDriveAuthenticator.CACHE_PATH):
            token_cache = open(OneDriveAuthenticator.CACHE_PATH, 'r').read()
            self.cache.deserialize(token_cache)
        
        self.msal_app = msal.ConfidentialClientApplication(
            OneDriveAuthenticator.CLIENT_ID, authority=OneDriveAuthenticator.AUTHORITY,
            client_credential=OneDriveAuthenticator.CLIENT_SECRET, token_cache=self.cache)
        
        self.flow = {}
        
    def save_cache(self):
        with open(OneDriveAuthenticator.CACHE_PATH, 'w') as f:
            f.write(self.cache.serialize())
        
    def start_login(self):
        self.flow = self.msal_app.initiate_auth_code_flow(
            OneDriveAuthenticator.GRAPH_USER_SCOPES,
            redirect_uri=OneDriveAuthenticator.REDIRECT_TO)
        return dict(auth_url=self.flow["auth_uri"], version=msal.__version__)

    def get_token(self):
        accounts = self.msal_app.get_accounts()
        if accounts:
            result = self.msal_app.acquire_token_silent(OneDriveAuthenticator.GRAPH_USER_SCOPES, account=accounts[0])
            return result
        
    def authenticate(self, request_args):
        result = self.msal_app.acquire_token_by_auth_code_flow(self.flow, request_args)
        if "error" in result:
            raise Exception(result)
        self.token_claims = result.get("id_token_claims")
        self.save_cache()
        return self.token_claims
        
        
oda = OneDriveAuthenticator()


class OneDriveManager(StorageManager):
    
    class _OneDriveWriter(BytesIO):
        
        def __init__(self, item_path, od, initial_bytes: bytes = ...) -> None:
            super().__init__(initial_bytes)
            self.od = od
            self.item_path = item_path
        
        def close(self):
            self.od.upload(item_path=self.item_path, data=self.getvalue())
            
    def __init__(self, oda: OneDriveAuthenticator) -> None:
        super().__init__()
        self.oda = oda
    
    @property
    def od(self):
        return OneDrive(self.oda.get_token().get('access_token', ''))
        
    def stat(self, path):
        path = path[12:].rstrip('/')
        info = self.od.get_item_data(item_path=path)
        return self._statinfo(path, info)
        
    def _statinfo(self, path, info):
        return {
            'name': info['name'],
            'fullpath': 'onedrive:///' + path,
            'ctime': parse_time(info['createdDateTime']).timestamp(),
            'mtime': parse_time(info['lastModifiedDateTime']).timestamp(),
            'size': info['size'],
            'type': 'folder' if info.get('folder') else 'file',
            '_url': info.get('@microsoft.graph.downloadUrl')
        }

    def statdir(self, path):
        path = path[12:].rstrip('/')
        return [self._statinfo(path + '/' + info['name'], info) for info in self.od.list_items(folder_path=path + '/')['value']]
    
    def listdir(self, path):
        return [info['name'] for info in self.statdir(path)]
    
    def readbuf(self, path):
        return storage.open(self.stat(path).get('_url'))
    
    def write_buf(self, path):
        path = path[12:].rstrip('/')
        return OneDriveManager._OneDriveWriter(path, self.od)
    

class OneDrivePlugin(Plugin):

    def __init__(self, pmanager, **_) -> None:
        super().__init__(pmanager)
        storage.register_scheme('onedrive', lambda *_: OneDriveManager(oda))
        
        @pmanager.app.route('/api/plugins/onedrive/login')
        def onedrive_login():
            from flask import redirect
            flow_data = oda.start_login()
            return redirect(flow_data['auth_url'])
        
        @pmanager.app.route(urllib.parse.urlparse(OneDriveAuthenticator.REDIRECT_TO).path, methods=['POST', 'GET'])
        def onedrive_auth():
            from flask import request
            oda.authenticate(request.args)
            return 'OK'
