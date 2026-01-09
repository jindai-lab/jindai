"""OAuth2 Client"""
import json
from flask import redirect, request, make_response

from jindai.plugin import Plugin
from jindai.helpers import safe_import
from jindai.models import UserInfo, SessionLocal
from authlib.integrations.flask_client import OAuth


class OAuthClientPlugin(Plugin):

    def __init__(self, pmanager, client_id, client_secret, metadata, redirect_logins=True, **_) -> None:
        super().__init__(pmanager)
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
            atoken = token['access_token']
            
            with SessionLocal() as session:
                user = session.query(UserInfo).filter(UserInfo.username == username).first()
                if user:
                    user.token = atoken
                    session.commit()
                    print(user.username, 'logged in via oauth', atoken[:10])
                    resp = make_response(f'<html><head><script>localStorage.token = {json.dumps(atoken)}; location.href = "/";</script></head></html>')
                    resp.set_cookie('token', atoken)
                    return resp
            return '', 403
        
        print('oauth registered')
