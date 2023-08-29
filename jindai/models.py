"""DB Objects"""
import datetime
from typing import Iterable, List, Union
import unicodedata
import dateutil
import time
from hashlib import sha1
from io import BytesIO
import pyotp

from PIL import Image, ImageFile
from PyMongoWrapper import F, Fn, Var, MongoField, MongoOperand
from PyMongoWrapper.dbo import (Anything, DbObjectCollection,
                                DbObjectInitializer, MongoConnection, classproperty)

from jindai.common import DictObject

from .config import instance as config
from .storage import instance as storage

db = MongoConnection('mongodb://' + config.mongo + '/' + config.mongoDbName)
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        if len(term) > Term.MAX_LENGTH:
            return
        tobj = Term.first(F.term == term, F.field == field)
        if term is not None:
            return tobj
        return Term(term=term, field=field).save()
    
    
class ObjectWithSource(db.DbObject):
    """Object with Source"""
    source = DbObjectInitializer(DictObject, DictObject)
    
    @classmethod
    def get(cls, url_or_cond: Union[str, MongoOperand, dict], **kwargs):
        source = kwargs.pop('source', DictObject())
        if isinstance(url_or_cond, str):
            source['url'] = url_or_cond
            cond = F.source.url == url_or_cond
        else:
            cond = url_or_cond

        p = cls.first(cond)
        if p:
            return p
        return cls(source=source, **kwargs)

    @staticmethod
    def merge_source(source1, source2):
        s_url = source1.get('url', '')
        t_url = source2.get('url', '')
        if '://' not in s_url and '://' in t_url:
            source1['url'] = t_url
            
        s_file = source1.get('file', '')
        t_file = source2.get('file', '')
        if not s_file and t_file:
            source1['file'] = t_file
        return source1
    
    
class SourceMongoOperand(MongoField):
    
    @property
    def url(self):
        return MongoField(self._literal + '.url')
    
    @property
    def file(self):
        return MongoField(self._literal + '.file')
    

F.source = SourceMongoOperand('source')
    

class MediaItem(ObjectWithSource):
    """Image item"""

    width = int
    height = int
    thumbnail = str
    item_type = str
    rating = float

    _IMAGE_EXTS = ['jpg',
                   'jpeg',
                   'jfif',
                   'jp2',
                   'png',
                   'gif',
                   'bmp']
    _VIDEO_EXTS = [
        'avi',
        'mp4',
        'flv',
        'm4v',
        '3gp',
        'mov'
    ]
    _AUDIO_EXTS = [
        'wav',
        'mp3',
        'aac',
        'ape'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image = None
        self._image_flag = False
        self._data = None

    @classproperty
    def acceptable_extensions(cls) -> list:
        """Get acceptable extension names
        """
        return MediaItem._IMAGE_EXTS + MediaItem._VIDEO_EXTS + MediaItem._AUDIO_EXTS

    @staticmethod
    def get_type(extname: str) -> str:
        """Decide media type from extension name
        """
        if extname in MediaItem._IMAGE_EXTS:
            return 'image'
        if extname in MediaItem._VIDEO_EXTS:
            return 'video'
        if extname in MediaItem._AUDIO_EXTS:
            return 'audio'
        return ''

    @property
    def image(self) -> Image.Image:
        """Get the PIL.Image.Image object for the image/thumbnail"""
        if self.item_type == 'video' and self.thumbnail:
            return Image.open(storage.open(self.thumbnail, 'rb'))

        if self.item_type == 'audio':
            return None

        # if image
        if self._image_flag:
            return self._image
        else:
            return Image.open(self.data)
        
    @image.setter
    def image(self, value: Image.Image):
        """Set the associated image"""
        self._image = value
        self._image_flag = True
        
    @property
    def data_path(self) -> str:
        """Get full path for data"""
        if self.source.get('file'):
            filename = self.source['file']
            if '://' in filename:
                return filename.replace('$', str(self.id))
            elif filename.lower().endswith('.pdf') and 'page' in self.source:
                return f'{self.source["file"]}#pdf/{self.source["page"]}'
            else:
                return storage.expand_path(filename)
        elif self.source.get('url'):
            return self.source['url']

    @property
    def data(self) -> BytesIO:
        """Get raw BytesIO for image data"""
        if not self._data:
            try:
                buf = storage.open(self.data_path, 'rb').read()
                self._data = BytesIO(buf)
            except OSError:
                self._data = BytesIO()
        else:
            self._data.seek(0)
            
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value

    def as_dict(self, expand=False):
        d = super().as_dict(expand)
        for k in [_ for _ in d if _.startswith('_') and _ != '_id']:
            del d[k]
        return d

    def save(self):
        """Save image items"""
        if self._image_flag:
            self.source['file'] = 'sblobs.db'
            with storage.open(f'sqlite://{self.id}', 'wb') as output:
                buf = self._image.tobytes('jpeg')
                output.write(buf)
            self._image_flag = False
           
        super().save()
        
    @classmethod
    def on_initialize(cls):
        """Initialize indicies"""
        cls.ensure_index('rating')
        cls.ensure_index('source')

    def __lt__(self, another):
        return id(self) < id(another)
    

class Paragraph(ObjectWithSource):
    """Paragraph object"""

    dataset = str
    author = str
    keywords = list
    pdate = StringOrDate()
    outline = str
    content = str
    pagenum = Anything
    lang = str
    images = DbObjectCollection(MediaItem, allow_duplicates=False)

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
        result['mongocollection'] = type(self).db.name
        return result

    def save(self):
        """Save the paragraph
        """

        if 'mongocollection' in self.__dict__:
            del self.mongocollection

        self.keywords = [k for k in set(self.keywords) if isinstance(k, str) and k.strip()]

        for field in ['keywords', 'author']:
            vals = self[field]
            if not isinstance(vals, list):
                vals = [vals]
            for val in vals:
                Term.write(val, field)

        for i in list(self.images):
            if not isinstance(i, MediaItem):
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
    
    @staticmethod
    def merge_by_mediaitems(preserved: MediaItem, dups: Iterable[MediaItem]):
        assert isinstance(preserved, MediaItem), 'Must specify one MediaItem to preserve'
        
        result = Paragraph.first(F.images == preserved.id) or Paragraph(images=[preserved])
        dup_ids = {x.id for x in dups if x.id != preserved.id}
        result.images = [i for i in result.images if i.id not in dup_ids]
        references = list(Paragraph.query(F.images.in_(list(dup_ids))))
        
        for para in references:
            para.images.remove(dup_ids)
            para.save()
                
        result.merge(references, no_images=True)
        result.images.append(preserved)

        for i in dups:
            if i.id == preserved.id:
                continue
            preserved.source = ObjectWithSource.merge_source(preserved.source, i.source)
            i.delete()

        result.save()
        preserved.save()

    def __lt__(self, another):
        return id(self) < id(another)
    
    def merge(self, others, no_images=True):
        
        if not others:
            return
        
        def _date_str(s):
            try:
                dt = dateutil.parser.parse(s) if isinstance(s, str) else s
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return s
        
        for field in self._fields:
            if not self[field]:
                for another in others:
                    if another[field]:
                        self[field] = another[field]
                        break
            else:
                if field in ('pdate',):
                    self[field] = min(self, *others, key=lambda x: _date_str(x[field]))[field]

                elif field == 'source':
                    for another in others:
                        if '://' in self[field].get('url', ''):
                            break
                        self[field] = ObjectWithSource.merge_source(self[field], another[field])

                elif field in ('keywords', 'images'):
                    if no_images and field == 'images':
                        continue
                    
                    arr = self[field]
                    for another in others:
                        arr += another[field]
                    self[field] = list(set(arr))


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
    display_name = str
    sources = list

    def update_sources(self):
        """Update sources to include all files in the dataset
        """
        results = Paragraph.get_coll(self.mongocollection).aggregator.match(
            F.dataset == self.name
        ).group(
            _id='$dataset', sources=Fn.addToSet('$source.file')
        )
        self.sources = []
        for result in results.perform(raw=True):
            self.sources += result['sources']
        self.save()

    def rename(self, new_name):
        """Rename the dataset
        """
        Paragraph.get_coll(self.mongocollection).query(
            F.dataset == self.name
        ).update(Fn.set(F.dataset == new_name))
        new_ds = Dataset.first(F.name == new_name)
        if new_ds:
            new_ds.sources = sorted(set(self.sources + new_ds.sources))
            self.delete()
        else:
            self.name = new_name
            self.save()


class TaskDBO(db.DbObject):
    """Task DB Object"""

    name = str
    params = dict
    pipeline = list
    resume_next = bool
    last_run = datetime.datetime
    concurrent = DbObjectInitializer(
        lambda *x: 3 if len(x) == 0 else int(float(x[0])), int)
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

    _TTL = 86400

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
            if token.expire - time.time() < Token._TTL:
                token.expire = time.time() + Token._TTL
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

