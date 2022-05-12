"""DB Objects"""
import datetime
import dateutil
import time
from hashlib import sha1
from io import BytesIO
import pyotp

from PIL import Image
from PyMongoWrapper import F, Fn, Var
from PyMongoWrapper.dbo import (Anything, DbObjectCollection,
                                DbObjectInitializer, MongoConnection)

from .config import instance as config
from .storage import safe_open

db = MongoConnection('mongodb://' + config.mongo + '/' + config.mongoDbName)


class StringOrDate(DbObjectInitializer):
    """String or date"""

    def __init__(self):

        def func(*args):
            if not args or not args[0]:
                return ''
            try:
                arg = dateutil.parser.parse(args[0])
                if isinstance(arg, datetime.datetime):
                    return arg
            except ValueError:
                return arg
            return ''

        super().__init__(func, None)


class Term(db.DbObject):
    """Term indexing"""

    MAX_LENGTH = 10
    term = str
    field = str

    @classmethod
    def on_initialize(cls):
        """Initialize indecies
        """
        cls.ensure_index('term')
        cls.ensure_index('field')

    @staticmethod
    def write(term, field):
        """"Save term into database"""
        if len(term) > Term.MAX_LENGTH: return
        tobj = Term.first(F.term == term, F.field == field)
        if term is not None:
            return tobj
        return Term(term=term, field=field).save()


class ImageItem(db.DbObject):
    """Image item"""

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
        """Get the PIL.Image.Image object for the image"""
        if self._image is None:
            self._image = Image.open(self.image_raw)
        return self._image

    @image.setter
    def image(self, value):
        """Set the associated image"""
        self._image = value
        self._image_flag = True

    @property
    def image_raw(self) -> BytesIO:
        """Get raw BytesIO for image data"""

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
        """Save image items"""
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
        """Initialize indicies"""
        cls.ensure_index('flag')
        cls.ensure_index('rating')
        cls.ensure_index('source')


class Paragraph(db.DbObject):
    """Paragraph object"""

    dataset = str
    author = str
    source = DbObjectInitializer(dict, dict)
    keywords = list
    pdate = StringOrDate()
    outline = str
    content = str
    pagenum = Anything
    lang = str
    images = DbObjectCollection(ImageItem)

    @classmethod
    def on_initialize(cls):
        """Initialize indecies
        """
        cls.ensure_index('dataset')
        cls.ensure_index('author')
        cls.ensure_index('source')
        cls.ensure_index('keywords')
        cls.ensure_index('pdate')
        cls.ensure_index('outline')
        cls.ensure_index('pagenum')
        cls.ensure_index('images')

    def as_dict(self, expand: bool = False) -> dict:
        """Convert the paragraph object to dict

        :param expand: expand its members, defaults to False
        :type expand: bool, optional
        :return: a dict object representing the paragraph object
        :rtype: dict
        """
        result = super().as_dict(expand)
        for k in [_ for _ in result if _.startswith('_') and _ != '_id']:
            del result[k]
        return result

    def save(self):
        """Save the paragraph
        """

        if 'mongocollection' in self.__dict__:
            del self.mongocollection

        self.keywords = [str(_).strip()
                         for _ in set(self.keywords) if _ and str(_).strip()]

        for field in ['keywords', 'author']:
            vals = self[field]
            if not isinstance(vals, list):
                vals = [vals]
            for val in vals:
                Term.write(val, field)

        for i in list(self.images):
            if not isinstance(i, ImageItem):
                self.images.remove(i)
                continue
            if i.id is None:
                i.save()
        super().save()

    @staticmethod
    def get_coll(coll):
        """Get subclass of Paragraph pointing to specific collection

        :param coll: collection name in MongoDB
        :type coll: str
        :return: the subclass or Paragraph itself
        :rtype: Type
        """

        if coll and coll not in ('null', 'default', 'undefined', 'paragraph'):
            class _Temp(Paragraph):
                _collection = coll
            return _Temp

        return Paragraph

    @staticmethod
    def get_converter(coll):
        """Get converter

        :param coll: collection name in MongoDB
        :type coll: str
        :return: the converter function
        :rtype: Callable[Paragraph]
        """

        temp = Paragraph.get_coll(coll)
        if temp is Paragraph:
            return lambda x: x

        return lambda x: temp(**x.as_dict())


class History(db.DbObject):
    """History record"""

    user = str
    created_at = datetime.datetime
    queries = list


class Meta(db.DbObject):
    """Meta settings"""

    app_title = str


class Dataset(db.DbObject):
    """Dataset info"""

    allowed_users = list
    order_weight = int
    mongocollection = str
    name = str
    sources = list


class TaskDBO(db.DbObject):
    """Task DB Object"""

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
    """User"""

    username = str
    password = str
    roles = list
    datasets = list
    otp_secret = str

    @staticmethod
    def encrypt_password(username, password_plain):
        """Encrypt the password

        :param username: username
        :type username: str
        :param password_plain: password in plain text
        :type password_plain: str
        :return: encrypted password
        :rtype: str
        """
        user_pass_salt = f'{username}_corpus_{password_plain}'.encode('utf-8')
        return f'{username}:{sha1(user_pass_salt).hexdigest()}'

    def set_password(self, password_plain=''):
        """Set password for the user

        :param password_plain: password in plain text, defaults to ''
        :type password_plain: str, optional
        """
        self.password = User.encrypt_password(self.username, password_plain)

    @staticmethod
    def authenticate(username, password, otp=''):
        """Authenticate

        :param username: user name
        :type username: str
        :param password: password in plain text
        :type password: str
        :param otp: one-time-password, defaults to ''
        :type otp: str, optional
        :return: user name if succeed, None otherwise
        :rtype: str | None
        """

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
    """User authentication token"""

    user = str
    token = str
    expire = float
    _cache = {}

    @staticmethod
    def check(token_string):
        """Check if token is valid

        :param token_string: token
        :type token_string: str
        :return: if valid
        :rtype: bool
        """

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
        """Log out user"""

        for token in Token._cache.values():
            if token.user == user:
                token.expire = 0
        Token.query(F.user == user).delete()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._roles = None

    @property
    def roles(self):
        """Get user roles"""

        if self._roles is None:
            self._roles = User.first(F.username == self.user).roles
        return self._roles
