"""Helper functions"""
import datetime
import glob
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from werkzeug.exceptions import HTTPException
from functools import wraps
from typing import IO, Dict, Type, Union

from dataclasses import dataclass, is_dataclass, field

import iso639
import numpy as np
import requests
import werkzeug.wrappers.response
from bson import ObjectId
from flask import Response, jsonify, request, Flask, abort
from flask.json.provider import JSONProvider as JSONProvideBase
from PIL.Image import Image
from PyMongoWrapper import MongoOperand, QExprEvaluator, F
from PyMongoWrapper.dbo import (
    create_dbo_json_decoder,
    create_dbo_json_encoder,
    DbObject,
)

from .config import instance as config
from .dbquery import parser
from .models import Token
from .storage import instance as storage


ee = QExprEvaluator()


def _me(param=""):
    """Add me() Function for query

    :param param: a string in query, defaults to ''
    :type param: str, optional
    :return: A string in form of "{param:}{logined user}"
    :rtype: _type_
    """
    param = str(param)
    if param:
        param += ":"
    return param + logined()


class JSONProvider(JSONProvideBase):

    def dumps(self, obj, **kwargs):
        return json.dumps(obj, **kwargs, cls=JSONEncoder)
    
    def loads(self, sbuf, **kwargs):
        return json.loads(sbuf, **kwargs, cls=JSONDecoder)


class WordStemmer:
    """
    Stemming words
    """

    _language_stemmers = {}

    @staticmethod
    def get_stemmer(lang):
        """Get stemmer for language"""
        safe_import("nltk")
        stemmer = safe_import("nltk.stem.snowball").SnowballStemmer
        if lang not in WordStemmer._language_stemmers:
            lang = language_iso639.get(lang, lang).lower()
            if lang not in stemmer.languages:
                return WordStemmer.get_stemmer("en")
            stemmer = stemmer(lang)
            WordStemmer._language_stemmers[lang] = stemmer
        return WordStemmer._language_stemmers[lang]

    def stem_tokens(self, lang, tokens):
        """Stem words

        :param tokens: list of words
        :type tokens: list
        :return: stemmed words
        :rtype: list
        """
        tokens = [WordStemmer.get_stemmer(lang).stem(_) for _ in tokens]
        return tokens

    def stem_from_params(self, word, lang="en"):
        """Add stem() function for query"""
        assert isinstance(lang, str) and isinstance(
            word, str
        ), f"Parameter type error for stem function: got {type(word)} and {type(lang)}"
        return {"keywords": self.stem_tokens(lang, [word])[0]}


parser.functions["me"] = _me
parser.functions["stem"] = WordStemmer().stem_from_params


def safe_import(module_name, package_name=""):
    """
    Import a module and if it's not installed install it.

    @param module_name - The name of the module to import.
    @param package_name - The name of the package to import the module from. Defaults to the module name if not specified.

    @return The imported module object.
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.call(
            [sys.executable, "-m", "pip", "install", package_name or module_name]
        )
    return importlib.import_module(module_name)


def rest(login=True, cache=False, role="", mapping=None):
    """
    Decorator for REST API endpoints. Decorates a function to be used as a WSGI application and returns a JSON response.

    @param login - If True ( default ) login the user with the given role
    @param cache - If True cache the response in the cache directory. This is useful if you want to make sure a user is logged in before accessing the endpoint.
    @param role - The role to check login status against. Defaults to''
    @param mapping - A dictionary of key / value pairs to be used as keyword arguments.

    @return A JSON response to the endpoint or an error if something went wrong
    """
    # Set mapping to a new mapping.
    if mapping is None:
        mapping = {}

    def do_rest(func):
        """
        Decorator to wrap REST calls. Decorated function will check login role and return 403 if user is not logged in.

        @param func - function to be wrapped. This is the function that will be called.

        @return a response to the function or an exception if something went wrong
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            """
            Wraps the function and returns a response. If an exception is raised it is logged and the response is returned to the client.


            @return The response to the client or a JSON object with error
            """
            try:
                erred = False
                # If login is not logged in return 403.
                if login and not logined(role):
                    return f"Forbidden. Client: {request.remote_addr}", 403
                if request.content_type == "application/json" and request.json:
                    for key, val in request.json.items():
                        kwargs[mapping.get(key, key)] = val
                elif request.method == 'GET':
                    for key, val in request.args.items():
                        kwargs[mapping.get(key, key)] = val

                request.lang = request.headers.get("X-Preferred-Language", "")
                result = func(*args, **kwargs)
                if isinstance(
                    result, (tuple, Response, werkzeug.wrappers.response.Response)
                ):
                    return result

                resp = jsonify(result)
            except HTTPException as hx:
                raise hx
            except Exception as ex:
                erred = True
                resp = jsonify(
                    {
                        "__exception__": type(ex).__name__ + ": " + str(ex),
                        "__tracestack__": traceback.format_tb(ex.__traceback__),
                    }
                )

            resp.headers.add("Access-Control-Allow-Origin", "*")
            # Add Cache Control header to the response.
            if cache and not erred:
                resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp

        return wrapped

    return do_rest


def logined(role=""):
    """
    Check if user logged in and return user object.
    This is used to handle requests that have a token in their request headers.

    @param role - Role to check. Check only if is logined if empty.

    @return User object or None if not logged in or token
    """
    token = Token.check(
        request.headers.get(
            "X-Authentication-Token",
            request.cookies.get("token", request.args.get("_token", "")),
        )
    )

    # Returns the user who owns the token.
    if token and (not role or role in token.roles):
        return token.user

    inet_addr = request.headers.get("X-Real-IP") or request.remote_addr

    # Check if IP address exists in automatic login mapping.
    if inet_addr in config.allowed_ips:
        return config.allowed_ips[inet_addr]

    return None


def serve_proxy(server, path):
    """Serve from remote server

    :param server: server host
    :type server: str
    :param path: path
    :type path: str
    :return: response from remote server
    :rtype: Response
    """
    resp = requests.get(f"http://{server}/{path}", timeout=1000)
    return Response(resp.content, headers=dict(resp.headers))


RE_DIGITS = re.compile(r"[\+\-]?\d+")


def evaluateqx(expr, obj):
    """Check according to parsed query expression

    :param parsed: Query Expression
    :type parsed: Union[Dict, str]
    :param obj: input object
    :type obj: Union[Dict, List, object]
    """
    if isinstance(expr, str):
        expr = parser.parse(f"expr({expr})")

    return ee.evaluate(expr, obj)


def get_context(directory: str, parent_class: Type, *sub_dirs: str) -> Dict:
    """Get context for given directory and parent class

    :param directory: directory path relative to the working directory
    :type directory: str
    :param parent_class: parent class of all defined classes to include
    :type parent_class: Type
    :return: a directory in form of {"ClassName": Class}
    :rtype: Dict
    """

    def _prefix(sub_dir, name):
        """Prefixing module name"""
        dirpath = directory
        if sub_dir != ".":
            dirpath += os.sep + sub_dir
        return dirpath.replace(os.sep, ".") + "." + name

    if len(sub_dirs) == 0:
        sub_dirs = ["."]
    modules = []
    for sub_dir in sub_dirs:
        modules += [
            _prefix(sub_dir, os.path.basename(f).split(".")[0])
            for f in glob.glob(os.path.join(directory, sub_dir, "*.py"))
        ] + [
            _prefix(sub_dir, f.split(os.path.sep)[-2])
            for f in glob.glob(os.path.join(directory, sub_dir, "*/__init__.py"))
        ]
    ctx = {}
    for module_name in modules:
        try:
            print("Loading", module_name)
            module = importlib.import_module(module_name)
            for k in module.__dict__:
                if (
                    k != parent_class.__name__
                    and not k.startswith("_")
                    and isinstance(module.__dict__[k], type)
                    and issubclass(module.__dict__[k], parent_class)
                ):
                    ctx[k] = module.__dict__[k]
        except Exception as exception:
            print("Error while importing", module_name, ":", exception)

    return ctx


JSONEncoderCls = create_dbo_json_encoder(json.JSONEncoder)


class JSONEncoder(json.JSONEncoder):
    """JSONEncoder for api use"""

    def __init__(self, **kwargs):
        """Initialize the JSON Encoder"""
        kwargs["ensure_ascii"] = False
        super().__init__(**kwargs)

    def default(self, o):
        """Encode the object o

        :param o: the object
        :type o: Any
        :return: str or JSON-compatible objects
        :rtype: Any
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int32):
            return o.tolist()
        if isinstance(o, Image):
            return str(o)
        if isinstance(o, datetime.datetime):
            return o.isoformat() + "Z"
        if isinstance(o, ObjectId):
            return str(o)
        if is_dataclass(o):
            return o.__dict__

        return JSONEncoderCls.default(self, o)


# JSONDecoder for api use
JSONDecoder = create_dbo_json_decoder(json.JSONDecoder)

# ISO639 language codes
language_iso639 = {
    lang.pt1: lang.name for lang in iso639.iter_langs() if lang.pt1 and lang.pt1 != "zh"
}
language_iso639.update(zhs="Chinese Simplified", zht="Chinese Traditional")

# API Response


@dataclass
class APIUpdate:
    success: bool = True
    bundle: dict = field(default_factory=dict)


@dataclass
class APIResults:
    results: list
    total: int
    query: str

    def __init__(self, results=None, total=-1, query=""):
        """
        Initialize the object with results.

        @param results - List of results to return. If None an empty list is returned.
        @param total - The total number of results to return. Defaults to - 1 which means there are no results in the result set.
        @param query - The query to return. Defaults to''
        """
        if results is None:
            results = []
        elif not isinstance(results, list):
            results = list(results)
        self.results = results
        self.total = total
        self.query = query


class IterableWithTotal:
    """An iterable object with total number recorded"""

    def __init__(self, iterable, total=-1):
        """
        Initialize the iterator with an iterable.

        @param iterable - The iterable to iterate over. It can be anything that supports iter. __iter__ ( such as a list or dictionary ).
        @param total - The total number of items in the iterable. If - 1 it is assumed that there are no more items
        """
        self.iterable = iterable
        self.total = total

    def __iter__(self):
        """
        Iterate over the elements of the iterable. This is useful for debugging and to ensure that the iterable is indeed a single element
        """
        yield from self.iterable

    def __len__(self):
        """
        Returns the number of elements in the iterable. This is equivalent to the Python built - in len () function.


        @return int ** of the number of elements in the iterable
        """
        return len(self.iterable)


class APICrudEndpoint:
    """API CRUD Endpoint Base Class"""

    def __init__(
        self, namespace, db_cls: DbObject, filtered_fields=None, allowed_fields=None
    ) -> None:
        """
        Initialize the instance. This is the entry point for the class. It should be called before any methods are called

        @param namespace - The namespace to use for the database
        @param db_cls - The class to use for the database
        @param filtered_fields - A list of fields to filter out
        @param allowed_fields - A list of fields to allow in the database

        @return An instance of the DbObject class with the namespace
        """
        self.filtered_fields = filtered_fields
        self.allowed_fields = allowed_fields
        self.db_cls = db_cls
        self.namespace = (
            f'/{namespace.strip("/")}/{self.db_cls.__name__.lower() + "s"}/'
        )
        self.maps = {}

    def get_operation(self):
        """
        Get operation to perform. This is used to determine the type of operation that should be performed by the API.


        @return A string representing the operation to perform on the API
        """
        # Returns the request method to use
        if request.method == "DELETE":
            return "delete"
        elif request.method == "PUT":
            return "create"
        elif request.method == "POST":
            return "update"
        else:
            return "read"

    def can_include(self, field):
        """
        Checks if the field can be included in the result. This is based on the allowed_fields and filtered_fields.

        @param field - The field to check. It can be a field or a tuple of fields.

        @return True if the field can be included False otherwise.
        """
        if self.filtered_fields and field in self.filtered_fields:
            return False
        if self.allowed_fields and field not in self.allowed_fields:
            return False
        return True

    def apply_sorting(self, results, limit, offset, sort):
        """
        Apply sorting to results. This is a wrapper around : meth : ` QuerySet. sort ` and

        @param results - The results to sort.
        @param limit - The limit to apply to the results. If None no limit is applied.
        @param offset - The offset to apply to the results. If None no offset is applied.
        @param sort - The sort to apply to the results. It can be a list of columns or a function that takes a column as a parameter and returns a string.

        @return A new QuerySet with sorting applied and the total number of results
        """
        results = results.sort(sort)
        total = results.count()
        # Skips the number of results to skip. limit limit
        if limit:
            results = results.skip(int(offset)).limit(int(limit))
        results.total = total
        return results

    def build_query(self, id, ids, query, data):
        """
        Build a query to search for documents. This is a helper method for L { get_documents } and L { get_documents_by_name }.

        @param id - The id of the document to search for.
        @param ids - A list of ids to search for. If None the document is not found in the list.
        @param query - The query to build. Can be a string or a MongoOperand
        @param data - A dictionary of field / value pairs that will be used to filter the results

        @return A mongo query that can be used to find documents
        """
        # Parse the query string and return a parsed query.
        if isinstance(query, str):
            query = parser.parse(query)

        query = MongoOperand(query or {})

        # query id name or id
        if id:
            query = F.id == id if re.match(r"^[0-9a-fA-F]{24}$", id) else F.name == id

        # Remove all ids from the list.
        if not ids:
            ids = []

        # Add ids to the ids list
        if data:
            ids += list(data.keys())

        ids = [ObjectId(i) for i in ids if re.match(r"^[0-9a-fA-F]{24}$", i)]

        # Returns a query to find all the ids in the database.
        if ids:
            query = F.id.in_(ids)

        return query

    def get_dbobjs(
        self, id=None, ids=None, query=None, limit=0, offset=0, sort="id", **data
    ):
        """
        Get objects from the database. This is a wrapper around query () to allow you to specify what fields you want to get in the result set

        @param id - If provided only the object with this id will be returned
        @param ids - If provided only the object with this ids will be returned
        @param query - Query to use for the query defaults to None
        @param limit - Limit the number of objects returned. Default is 0
        @param offset - Offset the number of objects returned. Default is 0
        @param sort - Sort the results using the given field. Default is'id '

        @return A list of objects matching the query or None if no objects match
        """
        query = self.build_query(id, ids, query, data)
        results = self.db_cls.query(query)

        # Returns the first result of the query.
        if id:
            return results.first()

        return self.apply_sorting(results, limit, offset, sort)

    def select_fields(self, obj, selection=None):
        """
        Return a dictionary of fields to be included in the query. This is a convenience method for

        @param obj - The object to select fields from
        @param selection - A list of field names to include

        @return A dictionary of fields to
        """
        return {
            k: w
            for k, w in obj.as_dict().items()
            if self.can_include(k) and (not selection or k in selection) or k == "_id"
        }

    def update_object(self, obj, data):
        """
        Update an object with new data. This is a low - level method to be used by : meth : ` update_json ` and

        @param obj - The object to update.
        @param data - The new data to update the object with.

        @return A list of fields that were updated in the object
        """
        updated_fields = set()
        # This function is used to add new values to the object.
        for field, newval in data.items():
            # Skips the include of the field.
            if not self.can_include(field):
                continue
            # Skip the _id field.
            if field == "_id":
                continue

            updated_fields.add(field)
            # This function is used to convert a dictionary to a field.
            if isinstance(newval, dict) and [1 for k in newval if k.startswith("$")]:
                # handle special assignments
                # Pushes the value from newval to the field.
                newval = dict(newval)
                if "$push" in newval:
                    vals = newval.pop("$push")
                    # Append the values to the field.
                    for val in vals:
                        # Append val to the field.
                        if val not in obj[field]:
                            obj[field].append(val)
                # Remove the pull value from the field.
                if "$pull" in newval:
                    vals = newval.pop("$pull")
                    # Remove all values from the field
                    for val in vals:
                        # Remove the value from the field.
                        if val in obj[field]:
                            obj[field].remove(val)
                # Add inc to the field.
                if "$inc" in newval:
                    obj[field] += newval.pop("$inc")
                # Remove the lower bound from the value of the field.
                if "$lowerBound" in newval:
                    obj[field] = max(newval.pop("$lowerBound"), obj[field])
                # upper bound of the field.
                if "$upperBound" in newval:
                    obj[field] = max(newval.pop("$upperBound"), obj[field])
                # round the value of the field
                if "$round" in newval:
                    obj[field] = round(obj[field] / newval["$round"]) * newval["$round"]
                    newval.pop("$round")
                # Set the value of the field to the new value.
                if newval:
                    obj[field] = ee.evaluate(newval, obj)
            elif newval is None and field in obj:
                del obj[field]

            else:
                setattr(obj, field, newval)

        obj.save()
        return self.select_fields(obj, updated_fields)

    def check_role(self, role):
        """
        Check if role is allowed to access the resource. This is a wrapper around logined () to avoid logging the request in production environments

        @param role - Role to check for.

        @return Error message or 403 if not allowed to access the
        """
        # require role role. If role is not logged in return forbidden
        if not logined(role):
            abort(403)

    def create(self, **data):
        """
        Create a new object in the database. This is a convenience method for creating a new object and saving it to the database.


        @return An APIUpdate with the fields of the newly created object
        """
        obj = self.db_cls(**data)
        obj.save()
        return APIUpdate(bundle=self.select_fields(obj))

    def update(self, objs, **data):
        """
        Update one or more objects. This is a low - level method for updating a set of objects in the API.

        @param objs - A DbObject or iterable of DbObjects to update.

        @return An APIUpdate containing the result of the update operation
        """
        # Update the data of the given objects.
        if isinstance(objs, DbObject):
            result = self.update_object(objs, data)
        else:
            result = {}
            # Update the data of all objects in the list
            for obj in objs:
                result[str(obj.id)] = self.update_object(
                    obj, data.get(str(obj.id), data)
                )
        return APIUpdate(bundle=result)

    def read(self, objs, **data):
        """
        Read a set of objects from the database. This is a wrapper around select_fields to allow you to pass an object or a DbObject to this method

        @param objs - the objects to read from the database

        @return a : class : ` APIResults ` object with the
        """
        # Select fields from a DbObject or a DbObject.
        if isinstance(objs, DbObject):
            return self.select_fields(objs)
        else:
            return APIResults(objs, objs.total)

    def delete(self, objs, **data):
        """
        Delete one or more objects. This is equivalent to calling : meth : ` Model. delete ` on the object and returning

        @param objs - An API object that is the target of the delete operation.

        @return An API object that is the result of the delete operation
        """
        objs.delete()
        return APIUpdate(bundle=str(objs.id))

    def bind_endpoint(self, func):
        """
        Decorator to bind an endpoint to a list of objects. This is useful for the endpoints that are called from the client and should take care of fetching the data from the server ( such as a query ) and passing it to the function as an argument

        @param func - The function to be bound

        @return The function that was bound to the endpoint's
        """

        @wraps(func)
        def wrapped(
            id=None, ids=None, query=None, limit=0, offset=0, sort="id", **data
        ):
            """
            Wrapper for get_dbobjs that returns a list of objects.

            @param id - id of the object to get. If None all objects will be returned.
            @param ids - list of ids to get. If None all ids will be returned.
            @param query - query to use when getting objects. If None all objects will be returned.
            @param limit - limit the number of objects to return. Default is 0.
            @param offset - skip this many objects before returning results. Default is 0.
            @param sort - sort the results by this field. Default is'id '.

            @return the return value of func ( objs ** data )
            """
            objs = self.get_dbobjs(id, ids, query, limit, offset, sort, **data)
            return func(objs, **data)

        self.maps[func.__name__] = wrapped
        return wrapped

    def bind(self, app: Flask, **options):
        """
        Bind REST handlers to Flask. This is a convenience method for creating and / or binding CRUD operations for the resource.

        @param app - The Flask application to bind the REST handlers to.

        @return The application passed in as ` ` app ` `
        """

        @rest(**options)
        def do_crud(
            id=None, ids=None, limit=0, query=None, offset=0, sort="id", **data
        ):
            """
            Perform CRUD operations on objects. By default this will return a 404 Not Found if there is no object with the given ID.

            @param id - ID of the object to get ( default None )
            @param ids - List of IDs of objects to get ( default None )
            @param limit - Limit the number of objects returned ( default 0 )
            @param query - Query to filter the objects by ( default None )
            @param offset - Offset the object to start from ( default 0 )
            @param sort - Sort the objects by the given field ( default 'id')

            @return Response to be sent to the client
            """
            objs = self.get_dbobjs(id, ids, query, limit, offset, sort, **data)
            operation = self.get_operation()

            # Returns a 404 if the object is not found.
            if id and objs is None:
                return f"{id} not matched", 404

            # Returns the object for the operation.
            if operation == "create":
                return self.create(**data)
            else:
                return getattr(self, operation)(objs, **data)

        endpoint = f"{self.namespace.replace('/', '_')}_crud"
        app.add_url_rule(
            f"{self.namespace}",
            methods=["GET", "PUT", "POST"],
            view_func=do_crud,
            endpoint=endpoint,
        )
        app.add_url_rule(
            f"{self.namespace}<id>",
            methods=["GET", "POST", "DELETE"],
            view_func=do_crud,
            endpoint=endpoint,
        )

        @rest()
        def get_scheme():
            """
            Get the name of the fields. This is used to determine what fields are in the database and can be used for a lookup of them.


            @return Dictionary of field names mapped to their type names
            """
            return {
                k: v.type.__name__ if getattr(v, "type") else type(v).__name__
                for k, v in self.db_cls.fields.items()
            }

        app.add_url_rule(
            f"{self.namespace}scheme",
            methods=["GET"],
            view_func=get_scheme,
            endpoint=endpoint + "_scheme",
        )

        # Add url rule for all maps in the map.
        for func_name, func in self.maps.items():
            app.add_url_rule(
                f"{self.namespace}{func_name}",
                methods=["GET", "POST"],
                view_func=rest(**options)(func),
                endpoint=endpoint + "_" + func_name,
            )

        return do_crud
