import datetime
import glob
import importlib
import json
import os
import re
import time
from hashlib import sha1
from typing import Any, Dict, Type, Union

import requests
from bson import ObjectId
from PIL import Image
from PyMongoWrapper import F, Fn, MongoOperand, QueryExprParser, dbo
from PyMongoWrapper.dbo import DbObject, DbObjectInitiator

import config
from storage import StorageManager

dbo.connstr = 'mongodb://' + config.mongo + '/' + config.mongoDbName
readonly_storage = StorageManager()

MongoJSONEncoder = dbo.create_dbo_json_encoder(json.JSONEncoder)


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


class Paragraph(DbObject):

    collection = str
    source = dict
    keywords = list
    pdate = str
    outline = str
    content = str
    pagenum = int
    lang = str
    image_storage = dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image = None
        self._image_flag = False

    @property
    def image(self):
        if self._image == None and self.image_storage:
            buf = readonly_storage.read(self.image_storage.get('id', self.id))
            self._image = Image.open(buf)
        return self._image

    @image.setter
    def image_setter(self, value):
        self._image = value
        self._image_flag = True

    def as_dict(self, expand=False):
        d = super().as_dict(expand)
        for k in [_ for _ in d if _.startswith('_') and _ != '_id']:
            del d[k]
        return d

    def save(self):
        im = self._image
        if self._image_flag:
            self._image = None
            if not self.image_storage:
                self.image_storage = {'blocks': True}
            else:
                self.image_storage = {'blocks': ObjectId()}

            with StorageManager() as mgr:
                buf = im.tobytes('jpeg')
                mgr.write(buf, self.id)

        super().save()
        self._image = im
        self._image_flag = False
    

class History(DbObject):

    user = str
    created_at = DbObjectInitiator(datetime.datetime.now)
    querystr = str


class Meta(DbObject):

    app_title = str


class Collection(DbObject):

    allowed_users = list
    order_weight = int
    mongocollection = str
    name = str
    sources = list


class TaskDBO(DbObject):

    name = str
    pipeline = list
    datasource = str
    datasource_config = dict
    resume_next = bool
    last_run = DbObjectInitiator(datetime.datetime.now)
    concurrent = DbObjectInitiator(lambda: 3)
    shortcut_map = dict


class User(DbObject):

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


class Token(DbObject):
    
    user = str
    token = str
    expire = float

    _cache = {}

    @staticmethod
    def check(token_string):
        t = Token._cache.get(token_string)
        if t and t.expire > time.time():
            t.expire = time.time() + 86400
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


class ImageItem(Paragraph):

    flag = int
    rating = int
    width = int
    height = int
    dhash = str
    whash = str
    
    storage = dbo.DbObjectInitiator(lambda: None)
    thumbnail = dbo.DbObjectInitiator(lambda: None)

    @classmethod
    def valid_item(cls) -> MongoOperand:
        """Returns condition for valid items (storage not null, flag == 0)

        Returns:
            MongoOperand: condition for valid items
        """        
        return (F.storage != None) & (F.flag == 0)

    def __repr__(self):
        return f'<ImageItem {self.source["url"]}>'

    def read_image(self):
        if not self.source.get('url') or not self.storage: return
        vt = self.id
        if self.source['url'].endswith('.mp4') or self.source['url'].endswith('.avi'):
            if not hasattr(self, 'thumbnail') or not self.thumbnail:
                try:
                    self.generate_thumbnail()
                except:
                    pass
            vt = self.thumbnail
        if vt:
            return StorageManager().read(vt)

    def generate_thumbnail(self, file_path=''):
        import os

        import cv2

        self.thumbnail = None
        p = file_path

        try:
            if not p:
                if self.storage:
                    p = f'_vtt{str(self.id)}'
                    with StorageManager() as mgr:
                        with open(p, 'wb') as fo:
                            blen = fo.write(mgr.read(self.id).read())
                    if not blen:
                        os.unlink(p)
                        return
                else:
                    p = self.source['url']

            cap = cv2.VideoCapture(p)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
                rval, frame = cap.read()
                cap.release()
                if rval:
                    rval, npa = cv2.imencode('.jpg', frame)
                    pic = npa.tobytes()
                    with StorageManager() as mgr:
                        vt = mgr.write(pic, f'{self.id}.thumb.jpg')
                    self.thumbnail = f'{self.id}.thumb.jpg'
                    self.save()
        except Exception as ex:
            print(ex)

        if p.startswith('_vtt') and os.path.exists(p):
            os.unlink(p)

    def delete(self):
        storages = []
        if self.storage:
            storages.append(self.id)
        if self.thumbnail:
            storages.append(self.thumbnail)
        super().delete()


class Album(Paragraph):    

    author = str
    liked_at = DbObjectInitiator(lambda: datetime.datetime.utcnow())
    items = dbo.DbObjectCollection(ImageItem)

    def __repr__(self):
        return f'<Album {self.source["url"]}>'

    def save(self):
        self.keywords = list(set(self.keywords))
        for i in self.items:
            if isinstance(i, DbObject) and i.id is None: i.save()
        super().save()


class AutoTag(DbObject):
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
        referer (str, optional): referer url. Defaults to ''.
        attempts (int, optional): max attempts. Defaults to 3.

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
