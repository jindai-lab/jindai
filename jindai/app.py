"""Flask App definition and configuration"""
from typing import Any, Dict, Tuple, Type
from uuid import UUID

from flask import Flask, jsonify, request, session
from flask_oidc import OpenIDConnect
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy

from .config import instance as config
ResponseTuple = Tuple[Dict[str, Any], int]

app = Flask(__name__)
app.json.ensure_ascii = False
app.config.update({
    'SQLALCHEMY_DATABASE_URI': config.database,
    'OIDC_CLIENT_SECRETS': config.oidc_secrets,
    'OIDC_RESOURCE_SERVER_ONLY': True, # Set to True for APIs
    'OIDC_SCOPES': ['openid', 'email', 'profile'],
    'SECRET_KEY': config.secret_key
})
oidc = OpenIDConnect(app)
db = SQLAlchemy(app)
api = Api(app, decorators=[oidc.accept_token()])


def assert_admin():
    from .models import UserInfo
    name = session.get("user", {}).get('name', '')
    user = UserInfo.query.filter(UserInfo.username == name, UserInfo.roles.contains('admin')).first()
    assert user, 'Must be admin'
    return user