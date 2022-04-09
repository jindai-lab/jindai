"""数据库对象"""
import datetime
import re
import time
from hashlib import sha1
from io import BytesIO
import pyotp

from PIL import Image
from PyMongoWrapper import F, Fn, MongoOperand, QueryExprParser, dbo, ObjectId
from PyMongoWrapper.dbo import (Anything,
                                DbObjectInitializer, MongoConnection)

from .config import instance as config
from .storage import safe_open

db = MongoConnection('mongodb://' + config.mongo + '/' + config.mongoDbName)


def _expr_groupby(params):
    if isinstance(params, MongoOperand):
        params = params()
    if 'id' in params:
        params['_id'] = {k[1:]: k for k in params['id']}
        del params['id']
    return [
        Fn.group(orig=Fn.first('$$ROOT'), **params),
        Fn.replaceRoot(newRoot=Fn.mergeObjects(
            '$orig', {'group_id': '$_id'}, {k: f'${k}' for k in params if k != '_id'}))
    ]


def _object_id(params):
    if isinstance(params, MongoOperand):
        params = params()
    if isinstance(params, str):
        return ObjectId(params)
    if isinstance(params, datetime.datetime):
        return ObjectId.from_datetime(params)
    return ObjectId()


parser = QueryExprParser(
    abbrev_prefixes={None: 'keywords=', '_': 'images.', '?': 'source.url%'},
    allow_spacing=True,
    functions={
        'groupby': _expr_groupby,
        'object_id': _object_id,
        'expand': lambda *x: [
            Fn.unwind('$images')(),
            Fn.lookup(from_='imageitem', localField='images',
                      foreignField='_id', as_='images')()
        ],
        'begin': lambda x: F.keywords.regex('^' + re.escape(x))
    },
    force_timestamp=False,
)


class StringOrDate(DbObjectInitializer):
    """字符串或日期"""

    def __init__(self):

        def func(*args):
            if not args or not args[0]:
                return ''
            try:
                arg = parser.parse_literal(args[0])
                if isinstance(arg, datetime.datetime):
                    return arg
            except ValueError:
                return arg
            return ''

        super().__init__(func, None)


class ImageItem(db.DbObject):
    """图像项目信息"""

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
        """图像信息"""
        if self._image is None:
            self._image = Image.open(self.image_raw)
        return self._image

    @image.setter
    def image(self, value):
        """设置图像"""
        self._image = value
        self._image_flag = True

    @property
    def image_raw(self) -> BytesIO:
        """获取附带图像的原始字节缓冲"""

        if self.source.get('file'):
            filename = self.source['file']
            if filename.lower().endswith('.pdf') and self.source.get('page') is not None:
                return safe_open(f'{self.source["file"]}#pdf/png:{self.source["page"]}', 'rb')
            if filename == 'blocks.h5':
                return safe_open(f"hdf5://{self.source.get('block_id', self.id)}", 'rb')
            return safe_open(filename, 'rb')

        if self.source.get('url'):
            return safe_open(self.source['url'], 'rb')

        return None

    def save(self):
        """保存"""
        image = self._image
        self._image = None

        if self._image_flag:
            self.source['file'] = 'blocks.h5'
            with safe_open(f'hdf5://{self.id}', 'wb') as output:
                buf = image.tobytes('jpeg')
                output.write(buf)
            self._image_flag = False

        super().save()
        self._image = image

    @classmethod
    def on_initialize(cls):
        """初始化时调用"""
        cls.ensure_index('flag')
        cls.ensure_index('rating')
        cls.ensure_index('source')


class Paragraph(db.DbObject):
    """语段信息"""

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
        result = super().as_dict(expand)
        for k in [_ for _ in result if _.startswith('_') and _ != '_id']:
            del result[k]
        return result

    def save(self):
        if 'mongocollection' in self.__dict__:
            del self.mongocollection
        self.keywords = [str(_).strip()
                         for _ in set(self.keywords) if _ and str(_).strip()]
        for i in list(self.images):
            if not isinstance(i, ImageItem):
                self.images.remove(i)
                continue
            if i.id is None:
                i.save()
        super().save()

    @staticmethod
    def get_coll(coll):
        """获取指向特定数据库集合 Collection 的语段对象"""

        if coll and coll not in ('null', 'default', 'undefined', 'paragraph'):
            class _Temp(Paragraph):
                _collection = coll
            return _Temp

        return Paragraph

    @staticmethod
    def get_converter(coll):
        """转换为指向特定数据库集合 Collection 的语段的转换器"""

        temp = Paragraph.get_coll(coll)
        if temp is Paragraph:
            return lambda x: x

        return lambda x: temp(**x.as_dict())


class History(db.DbObject):
    """历史记录对象"""

    user = str
    created_at = datetime.datetime
    querystr = str


class Meta(db.DbObject):
    """元设置对象"""

    app_title = str


class Dataset(db.DbObject):
    """数据集信息"""

    allowed_users = list
    order_weight = int
    mongocollection = str
    name = str
    sources = list


class TaskDBO(db.DbObject):
    """任务对象"""

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
    """用户对象"""

    username = str
    password = str
    roles = list
    datasets = list
    otp_secret = str

    @staticmethod
    def encrypt_password(username, password_plain):
        """加密密码"""
        user_pass_salt = f'{username}_corpus_{password_plain}'.encode('utf-8')
        return f'{username}:{sha1(user_pass_salt).hexdigest()}'

    def set_password(self, password_plain=''):
        """设置密码"""
        self.password = User.encrypt_password(self.username, password_plain)

    @staticmethod
    def authenticate(username, password, otp=''):
        """认证授权"""

        if otp:
            cond = (F.otp_secret != '') if username == '' else (
                F.username == username)
            for user in User.query(cond):
                totp = pyotp.TOTP(user.otp_secret)
                if totp.verify(otp):
                    return user.username

        if User.first(F.username == username,
                      F.password == User.encrypt_password(username, password)):
            return username

        return None


class Token(db.DbObject):
    """用户认证对象"""

    user = str
    token = str
    expire = float
    _cache = {}

    @staticmethod
    def check(token_string):
        """检查 token 是否有效"""

        token = Token._cache.get(token_string)
        if token and token.expire > time.time():
            if token.expire - time.time() < 86400:
                token.expire = time.time() + 86400
                token.save()
            return token

        token = Token.first((F.token == token_string)
                            & (F.expire > time.time()))
        if token:
            Token._cache[token_string] = token
            return token

    @staticmethod
    def uncheck(user):
        """注销用户登录"""

        for token in Token._cache.values():
            if token.user == user:
                token.expire = 0
        Token.query(F.user == user).delete()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._roles = None

    @property
    def roles(self):
        """获取用户角色"""

        if self._roles is None:
            self._roles = User.first(F.username == self.user).roles
        return self._roles
