import datetime
import glob
import importlib
import json
import os
import time
from hashlib import sha1
from typing import Dict, Type, Union
from io import BytesIO

from flask import request
import requests
from bson import ObjectId
from PIL import Image
from PyMongoWrapper import F, Fn, MongoOperand, QueryExprParser, dbo
from PyMongoWrapper.dbo import Anything, DbObjectInitializer, MongoConnection

import config
from storage import safe_open

db = MongoConnection('mongodb://' + config.mongo + '/' + config.mongoDbName)

MongoJSONEncoder = dbo.create_dbo_json_encoder(json.JSONEncoder)
MongoJSONDecoder = dbo.create_dbo_json_decoder(json.JSONDecoder)


def _expr_groupby(params):
    if isinstance(params, MongoOperand):
        params = params()
    if 'id' in params:
        params['_id'] = {k[1:]: k for k in params['id']}
        del params['id']
    return Fn.group(orig=Fn.first('$$ROOT'), **params), Fn.replaceRoot(newRoot=Fn.mergeObjects('$orig', {'group_id': '$_id'}, {k: f'${k}' for k in params if k != '_id'}))


def _object_id(params):
    if isinstance(params, MongoOperand):
        params = params()
    if isinstance(params, str):
        return ObjectId(params)
    if isinstance(params, datetime.datetime):
        return ObjectId.from_datetime(params)
    return ObjectId()

parser = QueryExprParser(abbrev_prefixes={None: 'keywords=', '_': 'images.', '?': 'source.url%'}, allow_spacing=True, functions={
    'groupby': _expr_groupby,
    'object_id': _object_id,
    'expand': lambda *x: [Fn.unwind('$images')(), Fn.lookup(from_='imageitem', localField='images', foreignField='_id', as_='images')()]
}, force_timestamp=False)


class StringOrDate(DbObjectInitializer):

    def __init__(self):
        def func(a=None, *args):
            if not a:
                return ''
            try:
                a = parser.parse_literal(a)
                if isinstance(a, datetime.datetime):
                    return a
            except:
                return a

        super().__init__(func, None)


class ImageItem(db.DbObject):

    flag = int
    rating = float
    width = int
    height = int
    thumbnail = str
    source = DbObjectInitializer(dict, dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image = None
        self._image_flag = False

    @property
    def image(self):
        if self._image == None:
            return Image.open(self.image_raw)
        else:
            return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self._image_flag = True

    @property
    def image_raw(self) -> BytesIO:
        if self.source.get('file'):
            fn = self.source['file']
            if fn.lower().endswith('.pdf') and self.source.get('page') is not None:
                return safe_open('{file}#pdf/png:{page}'.format(**self.source), 'rb')
            elif fn == 'blocks.h5':
                return safe_open(f"hdf5://{self.source.get('block_id', self.id)}", 'rb')
            else:
                return safe_open(fn, 'rb')
        elif self.source.get('url'):
            return safe_open(self.source['url'], 'rb')

    def save(self):
        im = self._image
        if self._image_flag:
            self._image = None
            self.source['file'] = 'blocks.h5'

            with safe_open(f'hdf5://{self.id}', 'wb') as fo:
                buf = im.tobytes('jpeg')
                fo.write(buf)

        super().save()
        self._image = im
        self._image_flag = False

    @classmethod
    def on_initialize(cls):
        cls.ensure_index('flag')
        cls.ensure_index('rating')
        cls.ensure_index('source')


class Paragraph(db.DbObject):

    dataset = str
    author = str
    source = DbObjectInitializer(dict, dict)
    keywords = list
    pdate = StringOrDate()
    outline = str
    content = str
    pagenum = Anything
    lang = str
    images = dbo.DbObjectCollection(ImageItem)

    @classmethod
    def on_initialize(cls):
        cls.ensure_index('dataset')
        cls.ensure_index('author')
        cls.ensure_index('source')
        cls.ensure_index('keywords')
        cls.ensure_index('pdate')
        cls.ensure_index('outline')
        cls.ensure_index('pagenum')
        cls.ensure_index('images')

    def as_dict(self, expand=False):
        d = super().as_dict(expand)
        for k in [_ for _ in d if _.startswith('_') and _ != '_id']:
            del d[k]
        return d

    def save(self):
        if 'mongocollection' in self.__dict__:
            del self.mongocollection
        self.keywords = list(set(self.keywords))
        for i in list(self.images):
            if not isinstance(i, ImageItem):
                self.images.remove(i)
                continue
            if i.id is None:
                i.save()
        super().save()

    @staticmethod
    def get_coll(coll):
        if coll and coll not in ('null', 'default', 'undefined', 'paragraph'):
            class _Temp(Paragraph):
                _collection = coll
            return _Temp

        return Paragraph

    @staticmethod
    def get_converter(coll):
        a = Paragraph.get_coll(coll)
        if a is Paragraph:
            return lambda x: x
        else:
            return lambda x: a(**a.as_dict())


class History(db.DbObject):

    user = str
    created_at = datetime.datetime
    querystr = str


class Meta(db.DbObject):

    app_title = str


class Dataset(db.DbObject):

    allowed_users = list
    order_weight = int
    mongocollection = str
    name = str
    sources = list


class TaskDBO(db.DbObject):

    name = str
    params = dict
    pipeline = list
    resume_next = bool
    last_run = datetime.datetime
    concurrent = DbObjectInitializer(
        lambda *x: 3 if len(x) == 0 else int(x), int)
    shortcut_map = dict
    creator = str
    shared = bool


class User(db.DbObject):

    username = str
    password = str
    roles = list
    datasets = list

    @staticmethod
    def encrypt_password(u, p):
        up = '{}_corpus_{}'.format(u, p).encode('utf-8')
        return '{}:{}'.format(u, sha1(up).hexdigest())

    def set_password(self, password_plain=''):
        self.password = User.encrypt_password(self.username, password_plain)

    @staticmethod
    def authenticate(u, p):
        if User.first((F.username == u) & (F.password == User.encrypt_password(u, p))):
            return u
        else:
            return None


class Token(db.DbObject):

    user = str
    token = str
    expire = float

    _cache = {}

    @staticmethod
    def check(token_string):
        t = Token._cache.get(token_string)
        if t and t.expire > time.time():
            if t.expire - time.time() < 86400:
                t.expire = time.time() + 86400
                t.save()
            return t
        else:
            t = Token.first((F.token == token_string)
                            & (F.expire > time.time()))
            if t:
                Token._cache[token_string] = t
                return t
        return None

    @staticmethod
    def uncheck(user):
        for t in Token._cache.values():
            if t.user == user:
                t.expire = 0
        Token.query(F.user == user).delete()

    @property
    def roles(self):
        if not hasattr(self, '_roles'):
            self._roles = User.first(F.username == self.user).roles
        return self._roles


def get_context(directory: str, parent_class: Type) -> Dict:
    modules = [
        directory + '.' + os.path.basename(f).split('.')[0]
        for f in glob.glob(os.path.join(os.path.dirname(__file__), directory, "*.py"))
    ] + [
        directory + '.' + f.split(os.path.sep)[-2]
        for f in glob.glob(os.path.join(os.path.dirname(__file__), directory, '*/__init__.py'))
    ]
    ctx = {}
    for mm in modules:
        try:
            m = importlib.import_module(mm)
            for k in m.__dict__:
                if k != parent_class.__name__ and not k.startswith('_') and isinstance(m.__dict__[k], type) and issubclass(m.__dict__[k], parent_class):
                    ctx[k] = m.__dict__[k]
        except Exception as ie:
            print('Error while importing', mm, ':', ie)
    
    return ctx
