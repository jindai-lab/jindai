import datetime
import glob
import importlib
import json
import os
import time
from hashlib import sha1
from typing import Dict, Type, Union
from io import BytesIO
from pdf2image import convert_from_path

import requests
from bson import ObjectId
from PIL import Image
from PyMongoWrapper import F, Fn, MongoOperand, QueryExprParser, dbo
from PyMongoWrapper.dbo import Anything, DbObjectInitializer, MongoConnection

import config
from storage import StorageManager

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


parser = QueryExprParser(abbrev_prefixes={None: 'keywords=', '_': 'items.', '?': 'source.url%'}, allow_spacing=True, functions={
    'groupby': _expr_groupby,
    'object_id': _object_id
}, force_timestamp=False)


def _pdf_image(file, page, **kwargs):
    buf = BytesIO()
    page = int(page)
    
    if file.endswith('.pdf'):
        for file in [file, os.path.join(config.storage, file)]:
            if os.path.exists(file): break
        else:
            return 'Not found', 404

        img, = convert_from_path(file, 120, first_page=page+1,
                                last_page=page+1, fmt='png') or [None]
        if img:
            img.save(buf, format='png')
            buf.seek(0)

    return buf


class StringOrDate(DbObjectInitializer):

    def __init__(self):
        def func(a=None, *args):
            if not a: return ''
            try:
                a = parser.parse_literal(a)
                if isinstance(a, datetime.datetime):
                    return a
            except:
                return a

        super().__init__(func, None)


class LengthedIO:
    
    def __init__(self, f, sz):
        self._f = f
        self._sz = sz

    def __len__(self):
        return self._sz

    def __getattribute__(self, name: str):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self._f, name)


class ImageItem(db.DbObject):

    flag = int
    rating = float
    width = int
    height = int
    dhash = bytes
    whash = bytes
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
    def image_setter(self, value):
        self._image = value
        self._image_flag = True        

    @property
    def image_raw(self) -> BytesIO:
        if self.source.get('file'):
            fn = self.source['file']
            if fn.startswith('/'): fn = fn[1:]
            if fn.lower().endswith('.pdf') and self.source.get('page') is not None:
                return _pdf_image(file=fn, page=self.source['page'])
            elif fn == 'blocks.h5':
                vt = self.source.get('block_id', self.id)
                return StorageManager().read(vt)
            else:
                if not os.path.exists(fn):
                    fn = os.path.join(config.storage, fn)
                fin = open(fn, 'rb')
                return LengthedIO(fin, os.stat(fn).st_size)
        elif self.source.get('url'):
            return BytesIO(try_download(self.source['url']))

    def save(self):
        im = self._image
        if self._image_flag:
            self._image = None
            self.source['file'] = 'blocks.h5'

            with StorageManager() as mgr:
                buf = im.tobytes('jpeg')
                mgr.write(buf, self.id)

        super().save()
        self._image = im
        self._image_flag = False


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
            if i.id is None: i.save()
        super().save()
    

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
    pipeline = list
    datasource = str
    datasource_config = dict
    resume_next = bool
    last_run = datetime.datetime
    concurrent = DbObjectInitializer(lambda *x: 3 if len(x) == 0 else int(x), int)
    shortcut_map = dict


class User(db.DbObject):

    username = str
    password = str
    roles = list

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
            t = Token.first((F.token == token_string) & (F.expire > time.time()))
            if t:
                Token._cache[token_string] = t
                return t
        return None

    @staticmethod
    def uncheck(user):
        for t in Token._cache.values():
            if t.user == user:
                t.expire = 0
        Token.query(F.user==user).delete()


class Album(Paragraph):

    liked_at = datetime.datetime


class AutoTag(db.DbObject):
    """Auto Tagging Object"""
    
    from_tag = str
    pattern = str
    tag = str


def get_context(directory : str, parent_class : Type) -> Dict:
    modules = [
                directory + '.' + os.path.basename(f)[:-3] 
                for f in glob.glob(os.path.join(os.path.dirname(__file__), directory, "*.py"))
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


def try_download(url: str, referer: str = '', attempts: int = 3, proxies = {}) -> Union[bytes, None]:
    """Try download from url

    Args:
        url (str): url
        referer (str, optional): referer url
        attempts (int, optional): max attempts

    Returns:
        Union[bytes, None]: response content or None if failed
    """

    buf = None
    for itry in range(attempts):
        try:
            if '://' not in url and os.path.exists(url):
                buf = open(url, 'rb').read()
            else:
                code = -1
                if isinstance(url, tuple):
                    url, referer = url
                headers = {
                    "user-agent": "Mozilla/5.1 (Windows NT 6.0) Gecko/20180101 Firefox/23.5.1", "referer": referer.encode('utf-8')}
                try:
                    r = requests.get(url, headers=headers, cookies={},
                                     proxies=proxies, verify=False, timeout=60)
                    buf = r.content
                    code = r.status_code
                except requests.exceptions.ProxyError:
                    buf = None
            if code != -1:
                break
        except Exception as ex:
            time.sleep(1)
    return buf
