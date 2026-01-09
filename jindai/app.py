"""Flask App definition and configuration"""
from typing import Type, Tuple, Dict, Any

from uuid import UUID
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask_oidc import OpenIDConnect

from .config import instance as config


app = Flask(__name__)
app.json.ensure_ascii = False
app.config['SQLALCHEMY_DATABASE_URI'] = config.database
app.config["SECRET_KEY"] = config.secret_key
app.config["OIDC_CLIENT_SECRETS"] = 'oidc-secrets.json'
oidc = OpenIDConnect(app)
db = SQLAlchemy(app)
api = Api(app)


ResponseTuple = Tuple[Dict[str, Any], int]