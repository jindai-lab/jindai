import jieba
from typing import Dict, Type
import glob, os, re, datetime
import importlib
import config
from hashlib import sha1
from PIL import Image
from PyMongoWrapper import dbo, F, QueryExprParser
from PyMongoWrapper.dbo import DbObject, DbObjectInitiator
import storage
dbo.connstr = 'mongodb://' + config.mongo + '/' + config.mongoDbName
readonly_storage = storage.StorageManager()
parser = QueryExprParser(abbrev_prefixes={None: 'keywords='}, allow_spacing=True)


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
        for k in [_ for _ in d if _.startswith('_')]:
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

            with storage.StorageManager() as mgr:
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

    users = list
    rootpath = str
    collections = list
    datasets = list

    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, '')
        super().__init__(**kwargs)


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


def get_context(directory : str, parent_class : Type) -> Dict:
    modules = [
                directory + '.' + os.path.basename(f)[:-3] 
                for f in glob.glob(os.path.join(os.path.dirname(__file__), directory, "*.py"))
            ]
    ctx = {}
    for m in modules:
        try:
            m = importlib.import_module(m)
            for k in m.__dict__:
                if k != parent_class.__name__ and not k.startswith('_') and isinstance(m.__dict__[k], type) and issubclass(m.__dict__[k], parent_class):
                    ctx[k] = m.__dict__[k]
        except ImportError as ie:
            print(ie)
    return ctx
