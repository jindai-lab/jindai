import jieba
from typing import Dict, Type
import glob, os, re, datetime
import importlib
import config
from hashlib import sha1
from PIL import Image
from PyMongoWrapper import *
from PyMongoWrapper.dbo import *
dbo.connstr = 'mongodb://' + config.mongo + '/hamster'


class Paragraph(DbObject):
    collection = str
    pdffile = str
    pdfpage = int
    keywords = list
    year = int
    outline = str
    content = str
    pagenum = int
    lang = str
    

class History(DbObject):

    user = str
    created_at = dbo.DbObjectInitiator(datetime.datetime.now)
    querystr = str


class Meta(DbObject):

    users = list
    pdffiles = dict
    rootpath = str
    collections = list

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
    last_run = dbo.DbObjectInitiator(datetime.datetime.now)
    concurrent = dbo.DbObjectInitiator(lambda: 3)


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
        m = importlib.import_module(m)
        for k in m.__dict__:
            if k != parent_class.__name__ and not k.startswith('_') and isinstance(m.__dict__[k], type) and issubclass(m.__dict__[k], parent_class):
                ctx[k] = m.__dict__[k]
    return ctx
