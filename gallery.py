import glob
import os
import re
import shutil
import time
import zipfile
from helpers import *
from collections import defaultdict
from io import BytesIO
from multiprocessing.pool import ThreadPool
import hashlib
import requests
from flask import Response, request, send_file, stream_with_context, jsonify, abort
from PIL import Image
from PyMongoWrapper import QueryExprParser, F, Fn, Var, SON, MongoOperand
from tqdm import tqdm
from urllib.parse import quote, unquote
import datetime
from typing import IO, Any, Callable, List, Dict, Iterable, Tuple, Union
import json
import base64
from bson import ObjectId
import config
from plugin import PluginContext, Plugin
from models import AutoTag, Post, Item, MongoJSONEncoder, Token, User
from storage import StorageManager

# prepare environment for requests
proxy = config.gallery.get('proxy') or os.environ.get(
    'http_proxy', os.environ.get('HTTP_PROXY'))
proxies = {'http': proxy, 'https': proxy} if proxy else {}
requests.packages.urllib3.disable_warnings()


# HELPER FUNCS

def chunks(l: Iterable, n: int = 10) -> List:
    """Split iterable to chunks of size `n`

    Args:
        l (Iterable): list
        n (int, optional): chunk size. Defaults to 10.

    Yields:
        Iterator[List]: chunks of size `n`
    """
    r = []
    for i in l:
        r.append(i)
        if len(r) == n:
            yield r
            r = []
    if r:
        yield r


def split_array(lst: Iterable, fn: Callable[[Any], bool]) -> Tuple[List, List]:
    """Split a list `lst` into two parts according to `fn`

    Args:
        lst (Iterable): list
        fn (Callable[Any, bool]): criteria function

    Returns:
        Tuple[List, List]: Splited list, fn -> True and fn -> False respectively
    """
    a, b = [], []
    for x in lst:
        if fn(x):
            a.append(x)
        else:
            b.append(x)
    return a, b


def single_item(pid: str, iid: str) -> List[Post]:
    """Return a single-item post object with id = `pid` and item id = `iid`

    Args:
        pid (str): Post ID
        iid (str): Item ID

    Returns:
        List[Post]: a list with at most one element, i.e., the single-item post object
    """
    if pid:
        pid = ObjectId(pid)
        p = Post.first(F._id == pid)
    elif iid:
        p = Post.first(F.items == ObjectId(iid))

    if iid and p:
        iid = ObjectId(iid)
        p.items = [i for i in p.items if i.id == iid]
        p.group_id = f"id={p['_id']}"
        return [p]
    else:
        return []


# HTTP SERV HELPERS
def arg(k: str, default: Any = None) -> Any:
    """Get arg from request.args or request.form or request.json

    Args:
        k (str): key name
        default (Any, optional): default value. Defaults to None.

    Returns:
        Any: the value of the key `k` in request context
    """
    return request.values.get(k) or (request.json or {}).get(k, default)


def argBool(k: str) -> bool:
    """Get arg from request.args or request.form or request.json

    Args:
        request (Request): request context
        k (str): key name

    Returns:
        bool: arg value of the key `k` in request context
    """
    v = arg(k)
    return argBoolv(v)


def argBoolv(v) -> bool:
    """Get bool value from"""
    return v == '1' or v == 'true' or v == 1 or v == True


def tmap(action: Callable, iterable: Iterable[Any], pool_size: int = 10) -> Tuple[Any, Tuple]:
    """Multi-threaded mapping with args included

    Args:
        action (Callable): action function
        iterable (Iterable): a list of args for the function call
        pool_size (int, optional): pool size. Defaults to 10.

    Yields:
        Iterator[Tuple[Any, Tuple]]: tuples of (result, args) 
    """
    count = -1
    if hasattr(iterable, '__len__'):
        count = len(iterable)
    elif hasattr(iterable, 'count'):
        count = iterable.count()

    with tqdm(total=count) as tq:

        def _action(a):
            r = action(a)
            tq.update(1)
            return r

        try:
            p = ThreadPool(pool_size)
            for r in chunks(iterable, pool_size):
                yield from zip(r, p.map(_action, r))
        except KeyboardInterrupt:
            return


def serve_file(p: Union[str, IO], ext: str = '', file_size: int = 0) -> Response:
    """Serve static file or buffer

    Args:
        p (Union[str, IO]): file name or buffer
        ext (str, optional): extension name. Defaults to '' for auto.
        file_size (int, optional): file size. Defaults to 0 for auto.

    Returns:
        Response: a flask response object
    """
    if isinstance(p, str):
        f = open(p, 'rb')
        ext = p.rsplit('.', 1)[-1]
        file_size = os.stat(p).st_size
    else:
        f = p

    mimetype = {
        'html': 'text/html',
                'htm': 'text/html',
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'gif': 'image/gif',
                'json': 'application/json',
                'css': 'text/css',
                'js': 'application/javascript',
                'mp4': 'video/mp4'
    }.get(ext, 'text/plain')

    start, length = 0, 1 << 20
    range_header = request.headers.get('Range')
    if file_size and file_size > 10 << 20:
        start = 0
        if range_header:
            # example: 0-1000 or 1250-
            m = re.search('([0-9]+)-([0-9]*)', range_header)
            g = m.groups()
            byte1, byte2 = 0, None
            if g[0]:
                byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])
            if byte1 < file_size:
                start = byte1
            if byte2:
                length = byte2 + 1 - byte1
            else:
                length = file_size - start
        else:
            length = file_size

        def _generate_chunks():
            l = length
            with f:
                f.seek(start)
                while l > 0:
                    chunk = f.read(min(l, 1 << 20))
                    l -= len(chunk)
                    yield chunk

        rv = Response(stream_with_context(_generate_chunks()), 206,
                      content_type=mimetype, direct_passthrough=True)
        rv.headers.add(
            'Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
        return rv
    else:
        return send_file(f, mimetype=mimetype, conditional=True)


def thumb(p: Union[str, IO], size: int) -> bytes:
    """Thumbnail image

    Args:
        p (Union[str, IO]): image source
        size (int): max size for thumbnail

    Returns:
        bytes: thumbnailed image bytes
    """
    img = Image.open(p).convert('RGB')
    buf = BytesIO()
    img.thumbnail(size)
    img.save(buf, 'jpeg')
    return buf.getvalue()


def apply_auto_tags(posts, ctx=None):
    """Apply auto tags to posts

    Args:
        posts (Iterable[Post]): post objects
        ctx (PluginContext, optional): plugin context. Defaults to None.
    """
    log = ctx.log if ctx else print
    m = list(AutoTag.query({}))
    if not m:
        return
    for p in tqdm(posts):
        for i in m:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in p.tags) or (pattern and re.search(pattern, p.source_url)):
                if tag not in p.tags:
                    p.tags.append(tag)
                if tag.startswith('@'):
                    p.author = tag
        p.save()


# POSTS HELPERS
def _groupby(params):
    if isinstance(params, MongoOperand):
        params = params()
    return [Fn.group(orig=Fn.first('$$ROOT'), **params), Fn.replaceRoot(newRoot=Fn.mergeObjects('$orig', {'group_id': '$_id'}, {k: f'${k}' for k in params if k != '_id'}))]


queryparser = QueryExprParser(abbrev_prefixes={'_': 'items.', None: 'tags=', '?': 'source_url%'}, functions={
    'groupby': _groupby
},
    shortcuts={
    'authored': {'tags': {'$regex': '^@'}},
    'groupped': {'tags': {'$regex': r'^\*'}},
    'fav': ((F.tags == 'fav') | (F['items.rating'] > 0))()
},
    force_timestamp=True, allow_spacing=True)


def Q(tags: str) -> Iterable[Post]:
    """Shortcut for querying posts with items resolved

    Args:
        tags (str): filtering expression

    Returns:
        Iterable[Post]: posts matching `tags`
    """
    ag = Post.aggregator
    ands = queryparser.eval(tags)
    match_first = 'items.' not in tags
    if match_first:
        ag.match(ands)
    ag.lookup(
        from_='item', localField=F.items, foreignField=F._id, as_=F.items2
    ).addFields(
        items=Fn.cond(Var.items2 == [], Var.items, Var.items2)
    )
    if not match_first:
        ag.match(ands)
    return ag


# PLUGINS
class DefaultTools(Plugin):
    """Plugin for default tools."""

    instance = None

    def __init__(self, app, **config):
        super().__init__(app, **config)
        DefaultTools.instance = self

    def get_tools(self) -> List[str]:
        """Return list of tools

        Returns:
            List[str]: tools
        """
        return [
            'check-images',
            'download',
            'tags-batch',
            'call',
            'import-page',
            'import-raw',
            'import-local',
            'user',
            'dump',
            'restore',
            'long-lasting'
        ]

    def check_images(self, ctx: PluginContext, condition: str = 'flag=0,width=0', *args):
        """Check images

        Args:
            ctx (PluginContext): plugin context
            condition (str, optional): query string. Defaults to 'width=0'.
        """
        items = Item.query(queryparser.eval(condition))

        def do_check_images(i):
            try:
                p = i.read_image()
                if not p:
                    return
                im = Image.open(p)
                im.verify()
                i.width, i.height = im.width, im.height
                i.save()
            except OSError as e:
                ctx.log(i.id, e)
                i.flag = 10
                i.save()
            except Exception as ex:
                ctx.log(i.id, ex)
        if not items:
            items = Item.query(F.width.empty())
        if not isinstance(items, list):
            items = list(items)

        for _ in tmap(do_check_images, items, 5):
            pass

        if 'nocallback' not in args:
            for cb in callbacks['check-images']:
                cb.run_callback(ctx, 'check-images', items)

    def download(self, ctx: PluginContext, limit: int = 0, check: bool = True):
        """Download

        Args:
            ctx (PluginContext): plugin context
            limit (int, optional): max number of items to download. Defaults to 0 standing for unlimited.
        """

        def __items(limit):
            rs = Item.query(((F.flag.eq(0) & F.storage.in_(
                None, False)) | (F.flag.eq(10) & F.storage.eq(True))) & (F.url.regex('^https?://'))).sort(-F.id)
            ctx.log('count:', rs.count())
            if limit:
                rs = rs.limit(limit)
            for i in rs:
                p = Post.first(F.items == i.id)
                if not p:
                    i.delete()
                    continue
                yield i, p

        def __download(i_p):
            i, p = i_p
            buf = try_download(i.url, p.source_url, ctx=ctx)
            time.sleep(0.2)
            return buf

        with StorageManager() as mgr:
            for (i, p), buf in tmap(__download, __items(limit)):
                if buf:
                    i.flag = 0
                    i.storage = mgr.write(buf, i.id)
                    i.save()

        if check:
            self.check_images(ctx)

    def tags_batch(self, ctx: PluginContext, q: str, tags: str):
        """Batch tagging posts

        Args:
            ctx (PluginContext): plugin context
            q (str): query string
            tags (str): tags string
        """
        q = F.id.in_([p.id for p in Q(q)])
        for t in tags.split(','):
            if t.startswith('~'):
                tparam = queryparser.eval(t[1:])
                ret = Post.query(q).update(Fn.pull(tparam))
            else:
                ret = Post.query(q & F.tags.ne(t)).update(Fn.push(tags=t))
            ctx.log('updated', ret.modified_count)

    def call(self, ctx: PluginContext, *args):
        """Call api at localhost

        Args:
            ctx (PluginContext): plugin context
        """
        ctx.log(requests.get('http://localhost:8877/tools',
                             params={'action': args[0], 'args': ' '.join(args[1:])}, proxies={}).content.decode('utf-8'))

    def import_posts(self, ctx: PluginContext, posts: List[Post]):
        """Save post objects and apply auto tags

        Args:
            ctx (PluginContext): plugin context
            posts (List[Post]): post objects
        """
        if not posts:
            return

        apply_auto_tags(posts, ctx=ctx)
        items = []
        for p in posts:
            for i in p.items:
                i.save()
            p.save()
            items += p.items

    def import_local(self, ctx: PluginContext, tag: str, *locs) -> List[Post]:
        """Import local files

        Args:
            ctx (PluginContext): plugin context
            tag (str): append tag to imported images
            locs (List[str]): local files

        Returns:
            posts (List[Post]): post objects
        """
        zips = []

        def __expand_zip(src):
            if '.zip#' in src:
                src = os.path.join('__zip{}'.format(
                    hash(src[:src.find('#')])), src[src.find('#')+1:])
            return src

        def __list_all(locs):
            locs = list(locs)
            l = []
            for loc in locs:
                if '*' in loc:
                    locs += glob.glob(loc)
                elif loc.endswith('.zip'):
                    zips.append(loc)
                    ctx.log(loc)
                    with zipfile.ZipFile(loc, 'r') as z:
                        l += [(loc, loc + '#' + _) for _ in z.namelist()]
                        z.extractall('__zip{}'.format(hash(loc)))
                elif os.path.isdir(loc):
                    ctx.log(loc)
                    l += [(loc, os.path.join(loc, _)) for _ in os.listdir(loc)]
                elif os.path.isfile(loc):
                    l.append((loc, loc))
            return l

        def __get_mtime(src):
            if '.zip#' in src:
                src = src[:src.find('#')]
            return int(os.stat(src).st_mtime)

        for _ in glob.glob('._*'):
            os.unlink(_)

        posts = defaultdict(Post)

        with StorageManager() as mgr:
            for loc, _f in sorted(__list_all(locs)):
                if _f.split('.')[-1] in ['txt', 'log', 'xlsx', 'xls', 'zip', 'csv'] or _f.endswith('.mp4.thumb.jpg'):
                    continue
                pu = loc.split('/')[-1]
                ftime = __get_mtime(_f)+8*3600

                p = posts[pu]
                if not p.source_url:
                    p.source_url = pu
                    if tag:
                        p.tags += tag.split(',')
                    p.created_at = ftime

                i = Item(url=_f)
                fn = __expand_zip(_f)
                if _f.endswith('.mp4') or _f.endswith('.avi'):
                    i.generate_thumbnail()
                else:
                    im = Image.open(fn)
                    i.width, i.height = im.size

                i.save()
                i.storage = mgr.write(fn, i.id)
                p.items.append(i)

        posts = posts.values()
        self.import_posts(ctx, posts)

        for _ in glob.glob('__zip*'):
            shutil.rmtree(_)
        for _ in zips:
            os.unlink(_)

        return posts

    def import_raw(self, ctx: PluginContext):
        """Read urls from console and import

        Args:
            ctx (PluginContext): plugin context
        """
        tags = []
        while True:
            try:
                l = input()
            except:
                l = ''
            if not l:
                break
            if not re.search(r'^https?://', l):
                tags = l.split()
                continue

            if '\t' in l:
                l, pu = l.split('\t')
            else:
                pu = l

            p = Post.query(F.source_url == pu).first() or Post(source_url=pu)
            p.tags = tags
            p.items.append(Item(url=l))
            ctx.log(p.source_url)
            p.save()

    def import_page(self, ctx: PluginContext, path, tag='', rng_start=0, rng_end=0):
        """Import images from web-page urls

        Args:
            ctx (PlguinContext): [description]
            path ([type]): [description]
            tag (str, optional): [description]. Defaults to ''.
            rng_start (int, optional): [description]. Defaults to 0.
            rng_end (int, optional): [description]. Defaults to 0.
        """

        from urllib.parse import urljoin

        posts = []

        if not isinstance(tag, str):
            tag = tag.decode('utf-8')
        rng = ['']
        if '##' in path:
            rng = range(int(rng_start), int(rng_end)+1)

        imgset = set()

        for i in rng:
            url = path.replace('##', str(i))
            p = Post.first(F.source_url == url) or Post(
                source_url=url, items=[])
            if url.endswith('.jpg'):
                imgs = [('', url)]
                title = ''
            else:
                ctx.log(url)
                html = try_download(url)
                assert html, 'Download failed.'
                try:
                    html = html.decode('utf-8')
                except:
                    try:
                        html = html.decode('gbk')
                    except:
                        try:
                            html = html.decode('euc-jp')
                        except:
                            html = html.decode('utf-8', errors='ignore')
                title = re.search(r'<title>(.*?)</title>', html) or ''
                if title:
                    title = title.group(1)
                title = re.sub(r'[\s]', u',', title)
                imgs = []
                for img in re.findall(r'<img.*?>|<div.*?>', html):
                    imgs += re.findall(
                        r'(zoomfile|data-original|data-src|src|file|data-echo)=["\'](.*?)["\']', img)
                imgs += re.findall(r'<a[^>]+(href)="([^"]*?\.jpe?g)"',
                                   html, flags=re.I)
                ctx.log(len(imgs), 'images found.')

            for _, img in imgs:
                imgurl = urljoin(url, img)
                if '.fc2.com/' in imgurl:
                    if imgurl.endswith('s.jpg'):
                        continue
                elif '/cute-' in imgurl:
                    imgurl = imgurl.replace('/cute-', '/')
                elif '/small/' in imgurl:
                    imgurl = imgurl.replace('/small/', '/big/')
                elif '.imagebam.com/' in imgurl:
                    imgfile = imgurl.split('/')[-1].split('.')[0]
                    html = try_download('http://www.imagebam.com/image/' + imgfile,
                                        referer='http://www.imagebam.com/').decode('utf-8')
                    imgurl = html[html.find('"og:image"'):]
                    imgurl = imgurl[imgurl.find('http://'):imgurl.find('"/>')]
                elif '/thumbs/' in imgurl or '/graphics/' in imgurl:
                    continue
                if imgurl not in imgset:
                    ctx.log(imgurl)
                    i = Item.first(F.url == imgurl) or Item(url=imgurl)
                    i.save()
                    p.items.append(i)
                    imgset.add(imgurl)
            p.tags = list(set(tag.split(',') + title.split(u',')))
            posts.append(p)

        self.import_posts(ctx, posts)

    def user(self, ctx: PluginContext, username = '', password = '', delete = False):
        if delete:
            User.query(F.username == username).delete()
            return True
        else:
            u = User.first(F.username == username) or User(username=username)
            u.set_password(password)
            u.save()
            return u.id

    def long_lasting(self, ctx : PluginContext):
        for _ in range(100):
            ctx.log(_)
            print(_)
            time.sleep(1)

    def dump(self, ctx: PluginContext, output: str = '', *colls):
        """Dump the current status of database to a zip file of jsons.

        Args:
            ctx (PluginContext): plugin context
            output (str, optional): output zip filename. Defaults to '' for a date string.
            colls (optional): list of collections. Defaults to [].
        """
        if not output:
            output = f'dump-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}{("," + ",".join(colls)) if colls else ""}.zip'
        jsonenc = MongoJSONEncoder(ensure_ascii=False)
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
            for coll in colls or ('post', 'item', 'jav', 'auto_tags', 'user', 'token'):
                fo = BytesIO()
                ctx.log('dumping', coll, '...')
                for p in tqdm(mongodb(coll).find(), total=mongodb(coll).count()):
                    fo.write(jsonenc.encode(p).encode('utf-8') + b'\n')
                fo.seek(0)
                z.writestr(coll, fo.read())
        with open('auditor.log', 'w'):
            pass

    def restore(self, ctx: PluginContext, infile: str, *colls, force: bool = True):
        """Restore the status of database from a zip file of jsons.

        Args:
            ctx (PluginContext): plugin context
            infile (str): input zip filename.
            colls (optional): list of collections. Defaults to [].
            force (bool, optional): ignore errors. Defaults to True.
        """
        def _hook(dic: Dict):
            """JSON decoder hook for restoring collections.

            Args:
                dic (dict): decoded dict

            Returns:
                dict: dic
            """
            if '_id' in dic:
                dic['_id'] = ObjectId(dic['_id'])
            if 'items' in dic and isinstance(dic['items'], list):
                dic['items'] = [ObjectId(_) for _ in dic['items']]
            for hw in ('dhash', 'whash'):
                if hw in dic:
                    if isinstance(dic[hw], int):
                        dic[hw] = '%016x' % dic[hw]
                        dic[hw] = bytes.fromhex(dic[hw])
                    elif isinstance(dic[hw], str):
                        if len(dic[hw]) == 12:
                            dic[hw] = base64.b64decode(dic[hw])
                        else:
                            dic[hw] = bytes.fromhex(dic[hw])
            return dic

        def _save_db(coll: str, ps: Iterable[Dict]):
            """Write items to database.

            Args:
                ps (list[dict]): list of decoded dicts from jsons.
            """
            try:
                if force:
                    mongodb(coll).delete_many({'_id': {'$in': [p['_id'] for p in ps]}})
                mongodb(coll).insert_many(ps, ordered=False,
                                          bypass_document_validation=True)
            except Exception as ex:
                ctx.log(ex)
                exit()

        with zipfile.ZipFile(infile, 'r') as z:
            restore_posts = set()
            restore_items = set()
            collections = []
            
            if not colls: colls = z.namelist()
            for coll in colls:
                if ':' in coll:
                    coll, cid = coll.split(':', 1)
                    if len(cid) == 24: cid = ObjectId(cid)
                    if coll == 'item':
                        restore_items.add(cid)
                    else:
                        restore_posts.add(cid)
                    collections.append('post')
                    collections.append('item')

                if coll not in collections: collections.append(coll)

            for coll in collections:
                ctx.log('restoring', coll, '...')
                fi = BytesIO(z.read(coll))
                ps = []
                for line in tqdm(fi):
                    p = json.loads(line.decode('utf-8'), object_hook=_hook)
                    if (not restore_items and not restore_posts) or (
                        restore_items and (
                            (coll == 'item' and p['_id'] in restore_items) or (coll == 'post' and restore_items.intersection(set(p['items'])))
                        )
                    ):
                        # print('\nfound match', p['_id'])
                        ps.append(p)
                    elif restore_posts and (
                            coll == 'post' and (p['_id'] in restore_posts or restore_posts.intersection(set(p['tags'])))
                        ):
                        # print('\nfound match', p['_id'], p['items'])
                        ps.append(p)
                        for i in p['items']:
                            restore_items.add(i)

                    if len(ps) > 100:
                        _save_db(coll, ps)
                        ps = []
                if ps:
                    _save_db(coll, ps)


# prepare plugins
plugins = []
tools = {}
special_pages = {}
callbacks = defaultdict(list)


def register_plugins(app):
    """Register plugins in config

    Args:
        app (Flask): Flask app object
    """
    import plugins as _plugins

    for pl in config.gallery.get('plugins') + [DefaultTools]:
        if isinstance(pl, tuple) and len(pl) == 2:
            pl, kwargs = pl
        else:
            kwargs = {}

        if isinstance(pl, str):
            if '.' in pl:
                plpkg, plname = pl.rsplit('.', 1)
                pkg = __import__('plugins.' + plpkg)
                for seg in pl.split('.'):
                    pkg = getattr(pkg, seg)
                pl = pkg
            else:
                pl = getattr(_plugins, pl)
        try:
            pl = pl(app, **kwargs)

            for name in pl.get_tools():
                tools[name] = pl

            for name in pl.get_callbacks():
                callbacks[name].append(pl)

            for name in pl.get_special_pages():
                special_pages[name] = pl

            plugins.append(pl)
        except Exception as ex:
            print('Error while registering plugin: ', pl, ex)
            continue


def init(app):
           
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    readonly_mgr = StorageManager()
    app.filters = []
    
    # STATIC AND RESOURCES PROVIDERS
    @app.route('/block/<id>.<ext>')
    def block(id, ext=''):
        """Serve binary data in h5 files

        Args:
            id (str): id of the file
            ext (str, optional): extension file. Defaults to ''.

        Returns:
            Response: binary data
        """
        if id.endswith('.thumb'):
            id += '.jpg'
        try:
            p = readonly_mgr.read(id)
            length = len(p.getvalue())
        except AssertionError:
            if len(id) == 24:
                Item.query(F.id == ObjectId(id)).update(Fn.set(storage=False))
                abort(404)

        if arg('enhance', ''):
            img = Image.open(p)
            p = BytesIO()
            # ImageOps.autocontrast(img).save(p, 'jpeg')
            brightness = ImageStat.Stat(img).mean[0]
            if brightness < 0.2:
                ImageEnhance.Brightness(img).enhance(1.2).save(p, 'jpeg')
            p.seek(0)
            ext = 'jpg'

        if arg('w', ''):
            w = int(arg('w'))
            sz = (w, min(w, 1280))
            p = BytesIO(thumb(p, sz))
            ext = 'jpg'

        resp = serve_file(p, ext, length)
        resp.headers.add("Cache-Control", "public,max-age=86400")
        return resp


    @app.route('/', methods=["GET", "POST"])
    @app.route('/<path:p>', methods=["GET", "POST"])
    def index(p='index.html'):
        """Serve static files in the working directory

        Args:
            p (str, optional): path. Defaults to 'thumbs.html'.

        Returns:
            Response: the file, or return 404 if not found
        """

        for path in [
            p,
            p + '.html',
            os.path.join('gallery-ui/dist', p)
        ]:
            if os.path.exists(path):
                return serve_file(path)

        return '', 404


    # ITEM OPERATIONS
    @app.route('/api/gallery/item/rating', methods=["GET", "POST"])
    @rest()
    def fav(items, inc=1, val=0):
        """Increase or decrease the rating of selected items

        Returns:
            Response: 'OK' if succeeded
        """
        for i in items:
            if i is None: continue
            i.rating = int(i.rating + inc) if inc else val
            if -1 <= i.rating <= 5:
                i.save()
        return {
            str(i.id): i.rating
            for i in items
        }


    @app.route('/api/gallery/item/reset_storage', methods=["GET", "POST"])
    @rest()
    def reset_storage(items):
        """Reset storage status of selected items

        Returns:
            Response: 'OK' if succeeded
        """
        for i in items:
            i.storage = not i.storage
            i.save()
        return {
            str(i.id): i.storage
            for i in items
        }


    @app.route('/api/gallery/item/merge', methods=["POST"])
    @rest()
    def merge_items(pairs):
        for rese, dele in pairs:
            dele = Item.first(F.id == dele)
            if not dele:
                continue
            
            if rese:
                pr = Post.first(F.items == ObjectId(rese)) or Post(
                    items=[ObjectId(rese)])
                for pd in Post.query(F.items == dele.id):
                    pr.tags += pd.tags
                    if (not pr.source_url or 'restored' in pr.source_url) and pd.source_url:
                        pr.source_url = pd.source_url
                    
                pr.save()
            
            Post.query(F.items == dele.id).update(Fn.pull(items=dele.id))        
            dele.delete()

        Post.query(F.items == []).delete()

        return True


    @app.route('/api/gallery/item/delete', methods=["POST"])
    @rest()
    def delete_item(post_items: dict):

        for pid, items in post_items.items():
            p = Post.first(F.id == ObjectId(pid))
            if p is None: continue

            items = list(map(ObjectId, items))
            p.items = [_ for _ in p.items if _.id not in items]        
            p.save()

        for i in items:
            if Post.first(F.items == i):
                continue
            print('orphan item, delete', str(i))
            Item.first(F.id == i).delete()

        Post.query(F.items == []).delete()

        return True


    # POST OPERATIONS
    @app.route('/api/gallery/post/split', methods=["GET", "POST"])
    @app.route('/api/gallery/post/merge', methods=["GET", "POST"])
    @rest()
    def splitting(posts):
        """Split or merge selected items/posts into seperate posts/one post

        Returns:
            Response: 'OK' if succeeded
        """
        
        if request.path.endswith('/split'):
            for p in posts:
                for i in p.items:
                    pnew = Post(source_url=p.source_url, liked_at=p.liked_at,
                                created_at=p.created_at, tags=p.tags, items=[i])
                    pnew.save()
                p.delete()
        else:
            if not posts: return False
            
            p0 = posts[0]
            p0.tags = list(p0.tags)
            p0.items = list(p0.items)
            for p in posts[1:]:
                p0.tags += list(p.tags)
                p0.items += list(p.items)
            p0.save()
            
            for p in posts[1:]:
                p.delete()

        return True


    @app.route('/api/gallery/post/group', methods=["GET", "POST", "PUT"])
    @rest()
    def grouping(posts, group='', delete=False):
        """Grouping selected posts

        Returns:
            Response: 'OK' if succeeded
        """
        def gh(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()[-9:]

        if delete:
            group_id = ''
            for p in posts:
                p.tags = [_ for _ in p.tags if not _.startswith('*')]
                p.save()

        else:
            if not posts:
                return True
            gids = []
            for p in posts:
                gids += [_ for _ in p.tags if _.startswith('*')]
            named = [_ for _ in gids if not _.startswith('*0')]

            if group:
                group_id = '*' + group
            elif named:
                group_id = min(named)
            elif gids:
                group_id = min(gids)
            else:
                group_id = '*0' + gh(min(map(lambda p: str(p.id), posts)))

            for p in posts:
                if group_id not in p.tags:
                    p.tags.append(group_id)
                    p.save()

            gids = list(set(gids) - set(named))
            if gids:
                for p in Post.query(F.tags.in_(gids)):
                    for id0 in gids:
                        if id0 in p.tags:
                            p.tags.remove(id0)
                    if group_id not in p.tags:
                        p.tags.append(group_id)
                    p.save()

        return group_id


    @app.route('/api/gallery/post/tag', methods=["GET", "POST"])
    @rest()
    def tag(posts, delete=[], append=[]):
        """Tagging selected posts

        Returns:
            Response: 'OK' if succeeded
        """
        
        for p in posts:
            for t in delete:
                if t in p.tags:
                    p.tags.remove(t)
                if p.author == t:
                    p.author = ''
            for t in append:
                t = t.strip()
                if t not in p.tags:
                    p.tags.append(t)
                if t.startswith('@'):
                    p.author = t
            p.save()
            
        return {
            str(p.id): p.tags
            for p in posts
        }


    @app.route('/api/gallery/search_tags', methods=['GET', 'POST'])
    @rest()
    def search_tags(tag, match_initials=False):
        if not tag: return []
        tag = re.escape(tag)
        if match_initials: tag = '^' + tag
        matcher = {'tags': {'$regex': tag, '$options': '-i'}}
        return [
                _['_id'] 
                for _ in Post.aggregator.match(matcher).unwind('$tags').match(matcher).group(_id='$tags').perform(raw=True)
                if len(_['_id']) < 15
            ]


    # AUTO_TAGS OPERATIONS
    @app.route('/api/gallery/auto_tags', methods=["PUT"])
    @rest()
    def auto_tags_create(tag: str = '', pattern: str = '', from_tag : str = None):
        """Perform automatic tagging of posts based on their source URL

        Args:
            tags (list[str]): list of tags
            source_url (str): source URL pattern
            delete (bool): remove the rule if true
        """
        assert tag and (pattern or from_tag), 'Must specify tag with pattern or from tag'
        AutoTag.query(pattern=pattern, from_tag=from_tag, tag=tag).delete()
        AutoTag(pattern=pattern, from_tag=from_tag, tag=tag).save()
        posts = Post.query(F.source_url.regex(pattern)) if pattern else Post.query(F.tags == from_tag)
        apply_auto_tags(posts)

        return True


    @app.route('/api/gallery/auto_tags', methods=["GET", "POST"])
    @rest()
    def auto_tags(ids : List[str] = [], delete=False):
        """List or delete automatic tagging of posts based on their source URL

        Args:
            ids (list[id]): list of ids
        """
        if delete:
            AutoTag.query(F.id.in_([ObjectId(_) for _ in ids])).delete()
            return True
        else:
            return [_.as_dict() for _ in AutoTag.query({}).sort([('_id', -1)])]


    @app.route('/api/gallery/get', methods=["GET", "POST"])
    @rest()
    def get(query='', flag=0, post='', limit=20, offset=0, order={'keys':['random']}, direction='next', groups=False, archive=False, count=False):
        """Get records

        Returns:
            Response: json document of records
        """
        
        if order == 'random':
            order = {'keys': ['random']}

        if 'keys' not in order:
            order['keys'] = ['-liked_at']

        if 'random' not in order['keys'] and '_id' not in order['keys']:
            order['keys'].append('_id')
            order['_id'] = ObjectId('00'*12)

        orders = [
            (o.split('-')[-1], -1 if o.startswith('-')
            == (direction == 'next') else 1)
            for o in order['keys'] if o != 'random'
        ]

        order_conds = []
        vconds = {}
        unwind_first = False

        for k, dr in orders:
            v = order.get(k)
            if v is None:
                continue
            if k.startswith('items.'):
                unwind_first = True
            if k == '_id':
                v = ObjectId(v or ('0' * 24))
            d = '$gt' if dr == 1 else '$lt'
            vconds[k] = {d: v}
            order_conds.append(dict(vconds))
            vconds[k] = v

        if order_conds:
            order_conds = [{'$or': order_conds}]

        ands = []

        querystr = query
        query = queryparser.eval(query)

        if isinstance(query, list) and len(query) > 0:
            ands = [query[0]] if query[0] else []
            query = query[1:]
        else:
            ands = [query] if query else []
            query = None

        ands += order_conds if (not groups and not archive) else []
        if len(ands) == 0:
            ands = []
        elif len(ands) == 1:
            ands = ands[0]
        else:
            ands = {'$and': ands}

        def __examine_dict(d):
            stk = [d]
            flag = False
            while stk:
                if flag:
                    break
                d = stk.pop()
                if isinstance(d, list):
                    for e in d:
                        stk.append(e)
                elif not isinstance(d, dict):
                    continue
                else:
                    for k, v in d.items():
                        if 'items.' in k:
                            flag = True
                        elif k == 'items' and isinstance(v, dict):
                            flag = sum([1 for k_ in v if not k_.startswith('$')])
                        else:
                            stk.append(v)
            return flag

        def __apply_app_filters(r):
            for f in app.filters:
                by = f['by']
                if by == 'tags':
                    if len(set(f['tags']).intersection(r['tags'])) > 0:
                        f['func'](r)

        match_first = not __examine_dict(ands)
        if not match_first:
            unwind_first = True

        aggregate = Post.aggregator

        if match_first:
            if ands:
                aggregate.match(ands)
            if not orders:
                aggregate.sample(size=limit*2)

        aggregate.lookup(
            from_='item', localField=F.items, foreignField=F._id, as_=F.items
        )

        if unwind_first:
            aggregate.unwind(Var.items).addFields(
                items=[Var.items])

        if not match_first and ands:
            aggregate.match(ands)

        if query:
            aggregate.aggregators += query

        if groups or archive:
            if not orders:
                orders = [('liked_at', -1)]

            aggregate.addFields(
                group_id=Fn.filter(input=Var.tags, as_='t',
                                cond=Fn.substrCP(Var._t, 0, 1) == '*')
            ).unwind(
                path=Var.group_id, preserveNullAndEmptyArrays=archive
            ).unwind(
                Var.items
            ).addFields(
                group_id=Fn.ifNull(Var.group_id, Fn.concat('id=`', Fn.toString(Var._id), '`')),
                **{'items.post_id': '$_id'}
            ).group(
                _id=Var.group_id,
                id=Fn.first(Var._id),
                liked_at=Fn.max(Var.liked_at),
                created_at=Fn.max(Var.created_at),
                source_url=Fn.first(Var.source_url),
                items=Fn.addToSet(Var.items),
                author=Fn.first(Var.author),
                tags=Fn.first(Var.tags),
                counts=Fn.sum(1),
                rating=Fn.max(Var['items.rating'])
            ).addFields(
                items='$items' if archive else [Fn.first(
                    Fn.filter(input=Var.items, as_='i', cond=Var['$i.rating'] == Var.rating))],
                _id=Var.id,
                group_id=Var._id
            )
            if archive and order_conds: aggregate.match({'$and': order_conds})
        
        if count:
            return (list(aggregate.group(_id='', count=Fn.sum(1)).perform(raw=True)) + [{'count': 0}])[0]['count']

        prev_order, next_order = {}, {}

        if orders:
            orders_params = {
                'order_conds': order_conds,
                'orders': SON(orders),
                'offset': offset,
                'limit': limit
            }
        else:
            orders_params = {}

        special_page = special_pages.get(post.split('/')[0])
        if special_page:
            ret = special_page.special_page(aggregate, locals(),
                                            orders_params=orders_params) or ([], None, None)
            if isinstance(ret, tuple) and len(ret) == 3:
                rs, prev_order, next_order = ret
            else:
                return ret
        else:
            if orders:
                if not query or not [_ for _ in query if '$sort' in _]:
                    aggregate.sort(SON(orders))
                    if offset:
                        aggregate.skip(offset)

                aggregate.limit(limit)
            else:
                aggregate.sample(size=limit)

            rs = aggregate.perform(raw=True)

        r = []
        res = {}
        try:
            for res in rs:
                if isinstance(res, Post):
                    res = res.as_dict(expand=True)

                if 'count' not in res:
                    res['count'] = ''

                if '_id' not in res or not res['items']:
                    continue

                res['items'] = [_ for _ in res['items'] if isinstance(
                    _, dict) and _.get('flag', 0) == flag]
                if not res['items']:
                    continue

                if not orders:  # random
                    res['items'] = random.sample(res['items'], 1)
                elif archive or groups or 'counts' in res:
                    cnt = res.get('counts', len(res['items']))
                    if cnt > 1:
                        res['count'] = '(+{})'.format(cnt)

                __apply_app_filters(res)
                if direction != 'next':
                    r.insert(0, res)
                else:
                    r.append(res)

        except Exception as ex:
            import traceback
            return jsonify({
                'exception': repr(ex),
                'trackstack': traceback.format_exc(),
                'results': [res],
                '_filters': [
                    ands, {'orders': orders, 'aggregate': aggregate.aggregators}
                ]
            }), 500

        def mkorder(rk):
            o = dict(order)
            for k in o['keys']:
                k = k[1:] if k.startswith('-') else k
                if '.' in k:
                    o[k] = rk
                    for k_ in k.split('.'):
                        o[k] = o[k].get(k_, {})
                        if isinstance(o[k], list):
                            o[k] = o[k][0] if o[k] else {}
                    if o[k] == {}:
                        o[k] = 0
                else:
                    o[k] = rk.get(k, 0)
            return o

        if r:
            if not prev_order:
                prev_order = mkorder(r[0])
            if not next_order:
                next_order = mkorder(r[-1])

        return jsonify({
            'total_count': len(r),
            'params': request.json,
            'prev': prev_order,
            'next': next_order,
            'results': r,
            '_filters': [
                ands, {'orders': orders, 'aggregate': aggregate.aggregators}
            ]
        })


    @app.route('/api/gallery/plugins/style.css')
    def plugins_style():
        """Returns css from all enabled plugins

        Returns:
            Response: css document
        """
        css = '\n'.join([p.run_callback(PluginContext(), 'css')
                        for p in callbacks['css']])
        return Response(css, mimetype='text/css')


    @app.route('/api/gallery/plugins/script.js')
    def plugins_script():
        """Returns js scripts from all enabled plugins

        Returns:
            Response: js document
        """
        js = '\n'.join([p.run_callback(PluginContext(), 'js')
                    for p in callbacks['js']])
        return Response(js, mimetype='text/javascript')


    @app.route('/api/gallery/plugins/special_pages', methods=["GET", "POST"])
    @rest()
    def plugins_special_pages():
        """Returns names for special pages in every plugins
        """
        return list(special_pages.keys())


    @app.route('/api/gallery/plugins/tool', methods=["GET", "POST"])
    @rest()
    def tools_view(action='', args=[]):
        """Call tools

        Returns:
            Response: json document for a list of tools

        Yields:
            str: output logs of the running tool
        """
        if not action:
            return sorted(tools.keys())

        if action not in tools:
            abort(404)

        args = [queryparser.expand_literals(_) for _ in args if _]

        def generate():
            """Generate log text from plugin context

            Yields:
                str: log text
            """
            f = tools[action]
            yield 'args: ' + str(args) + '\n\n'

            ctx = PluginContext(action)
            ctx.run(f.run_tool, action, *args)
            while ctx.alive:
                yield from ctx.fetch()
                time.sleep(0.1)

            yield from ctx.fetch()
            yield 'returned: ' + MongoJSONEncoder(ensure_ascii=False).encode(ctx.ret) + '\n'

            yield 'finished.\n'

        return Response(stream_with_context(generate()), status=200,
                        mimetype="text/plain",
                        content_type="text/event-stream"
                        )


    @app.route('/api/gallery/stats')
    @rest()
    def stats():
        """Print out statistics

        Returns:
            Response: json document of statistics
        """
        return {
            'posts': Post.query({}).count(),
            'items': Item.query({}).count(),
            'items_saved': Item.query({'storage': True}).count(),
        }

    register_plugins(app)


def main():
    import sys
    register_plugins(None)
    ctx = PluginContext('cmdline', join=True, logging_hook=print)
    
    if sys.argv[1] in tools:
        tool_func = tools[sys.argv[1]]
        if isinstance(tool_func, Plugin):
            tool_func.run_tool(ctx, *sys.argv[1:])
        else:
            tool_func(*sys.argv[2:])

    else:
        ctx.log('unknown command', sys.argv[1])


if __name__ == '__main__':
    main()

