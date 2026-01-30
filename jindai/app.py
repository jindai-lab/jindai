"""Flask App definition and configuration"""

from typing import Any, Dict, Tuple, Type
from uuid import UUID

from asteval import Interpreter
from flask import Flask, g, jsonify, request, session
from flask_oidc import OpenIDConnect
from flask_restful import Api, Resource

from .config import instance as config
from .storage import instance as storage

ResponseTuple = Tuple[Dict[str, Any], int]

app = Flask(__name__)
app.json.ensure_ascii = False
app.config.update(
    {
        "OIDC_CLIENT_SECRETS": config.oidc_secrets,
        "OIDC_RESOURCE_SERVER_ONLY": True,  # Set to True for APIs
        "OIDC_SCOPES": ["openid", "email", "profile"],
        "SECRET_KEY": config.secret_key,
    }
)
oidc = OpenIDConnect(app)
api = Api(app, decorators=[oidc.accept_token()])


def aeval(expr, context) -> list | None:
    """Evaluate an expression in a safe context

    :param expr: Expression to evaluate
    :type expr: str
    :param context: Context object for evaluation
    :type context: dict or object with as_dict() method
    :return: Evaluation result
    :rtype: list | None
    """
    if not isinstance(context, dict):
        context = context.as_dict()
    ee = Interpreter(context)
    return ee(expr)


def assert_admin():
    """Assert that the current user has admin role

    :raises AssertionError: If user is not found or doesn't have admin role
    :return: User object with admin role
    :rtype: UserInfo
    """
    from .models import UserInfo, db_session

    name = g.authlib_server_oauth2_token["preferred_username"]
    user = (
        db_session.query(UserInfo)
        .filter(UserInfo.username == name, UserInfo.roles.contains(["admin"]))
        .first()
    )
    assert user, f"Must be admin. You logged in as: {name}"
    return user
