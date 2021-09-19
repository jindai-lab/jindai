import time
from typing import Dict, Type, Union
import glob, os, re, datetime
import importlib
import config
from hashlib import sha1
from PIL import Image
from PyMongoWrapper import dbo, F, Fn, QueryExprParser, MongoOperand
from PyMongoWrapper.dbo import DbObject, DbObjectInitiator
import storage
dbo.connstr = 'mongodb://' + config.mongo + '/' + config.mongoDbName
readonly_storage = storage.StorageManager()


def _expr_groupby(params):
    if isinstance(params, MongoOperand):
        params = params()
    if 'id' in params:
        params['_id'] = {k[1:]: k for k in params['id']}
        del params['id']
    return Fn.group(orig=Fn.first('$$ROOT'), **params), Fn.replaceRoot(newRoot=Fn.mergeObjects('$orig', {'group_id': '$_id'}, {k: f'${k}' for k in params if k != '_id'}))


parser = QueryExprParser(abbrev_prefixes={None: 'keywords='}, allow_spacing=True, functions={
    'groupby': _expr_groupby
})


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
        ctx (PluginContext, optional): plugin context. Defaults to None.

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
