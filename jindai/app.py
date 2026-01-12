"""Flask App definition and configuration"""
from typing import Any, Dict, Tuple, Type
from uuid import UUID

from flask import Flask, jsonify, request, session, g
from flask_oidc import OpenIDConnect
from flask_restful import Api, Resource

from .config import instance as config
from .storage import instance as storage

ResponseTuple = Tuple[Dict[str, Any], int]

app = Flask(__name__)
app.json.ensure_ascii = False
app.config.update({
    'OIDC_CLIENT_SECRETS': config.oidc_secrets,
    'OIDC_RESOURCE_SERVER_ONLY': True, # Set to True for APIs
    'OIDC_SCOPES': ['openid', 'email', 'profile'],
    'SECRET_KEY': config.secret_key
})
oidc = OpenIDConnect(app)
api = Api(app, decorators=[oidc.accept_token()])


def assert_admin():
    from .models import UserInfo
    name = g.authlib_server_oauth2_token['preferred_username']
    user = db_session.query(UserInfo).filter(UserInfo.username == name, UserInfo.roles.contains(['admin'])).first()
    assert user, f'Must be admin. You logged in as: {name}'
    return user