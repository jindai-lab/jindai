"""OAuth2 Client"""
import time
from flask import redirect, request, make_response

from jindai.plugin import Plugin
from jindai.helpers import safe_import
from jindai.models import User, Token
from PyMongoWrapper import F

safe_import('authlib')


class OAuthClientPlugin(Plugin):

    def __init__(self, pmanager, client_id, client_secret, metadata, redirect_logins=False, **_) -> None:
        super().__init__(pmanager)
        from authlib.integrations.flask_client import OAuth
        oauth = OAuth(pmanager.app)
        oauth.register(
            'openid',
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=metadata,
            client_kwargs={'scope': 'openid profile email'}
        )
        
        if redirect_logins:
            @pmanager.app.route('/login')
            def overwrite_login():
                return redirect('/login/openid')

        @pmanager.app.route('/login/openid')
        def login_openid():
            return oauth.openid.authorize_redirect(f'https://{request.host}/login/authorize_callback')

        @pmanager.app.route('/login/authorize_callback')
        def authorize_callback():
            token = oauth.openid.authorize_access_token()
            username = token['userinfo']['preferred_username']
            user = User.first(F.username == username)
            if user:
                Token.query((F.user == username) & (F.expire < time.time())).delete()
                token = User.encrypt_password(str(time.time()), str(time.time_ns()))
                Token(user=username, token=token, expire=time.time() + 86400).save()
                resp = make_response(f'<html><head><script>localStorage.token = "{token}"; location.href = "/";</script></head></html>')
                resp.set_cookie('token', token)
                return resp
            return '', 403
