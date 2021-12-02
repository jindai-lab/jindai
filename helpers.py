from functools import wraps
from flask import request, Response, jsonify
import traceback
from PyMongoWrapper import F
from models import Token, User
from concurrent.futures import ThreadPoolExecutor


def rest(login=True, cache=False, user_role=''):
    def do_rest(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                if (login and not logined()) or \
                    (user_role and not User.first((F.roles == user_role) & (F.username == logined()))):
                    raise Exception('Forbidden.')
                if request.json:
                    kwargs.update(**request.json)
                f = func(*args, **kwargs)
                if isinstance(f, Response): return f
                resp = jsonify({'result': f})
            except Exception as ex:
                resp = jsonify({'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)})
            
            resp.headers.add("Access-Control-Allow-Origin", "*")
            if cache:
                resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp
        return wrapped

    return do_rest


def logined():
    t = Token.check(request.headers.get('X-Authentication-Token', request.cookies.get('token')))
    if t:
        return t.user


def tmap(func, iterable, n=10):
    tpe = ThreadPoolExecutor(n)
    return tpe.map(func, iterable)

