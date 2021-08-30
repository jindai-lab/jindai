#!env python3
import hashlib
import os
import sys
import time
from typing import List

from bson import ObjectId
from flask import Flask, abort, jsonify, Response, request, stream_with_context
from io import BytesIO
import random
import re
from urllib.parse import quote
from PIL import Image, ImageOps

import config
from databackend import F, Fn, Var,Post, Item, MongoJSONEncoder, mongodb, SON
from storage import StorageManager
from plugin import PluginContext, Plugin

readonly_mgr = StorageManager()

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


def try_download(url: str, referer: str = '', attempts: int = 3, ctx: PluginContext = None) -> Union[bytes, None]:
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
    log = print if ctx is None else ctx.log
    for itry in range(attempts):
        try:
            if '://' not in url and os.path.exists(url):
                buf = open(url, 'rb').read()
            else:
                time.sleep(1)
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
                    log('Proxy error,', proxies)
                    buf = None
                except Exception as ex:
                    log(url, ex)
                    buf = None
            if code != -1:
                break
        except Exception as ex:
            log(url, itry, ex)
            time.sleep(1)
    return buf


def client_ip():
    """Get client ip from request

    Returns:
        str: client ip
    """
    return request.environ.get('HTTP_X_FORWARDED_FOR', '').split(', ')[0] or request.environ.get('CLIENT_IP', '')


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
    m = list(mongodb('auto_tags').find())
    if not m:
        return
    for p in posts:
        for i in m:
            if 'from_tag' in i:
                pattern, tag = i['from_tag'], i['tag']
                if pattern in p.tags and tag not in p.tags:
                    p.tags.append(tag)
            else:
                pattern, tag = i['pattern'], i['tag']
                if re.search(pattern, p.source_url):
                    log(p.source_url, pattern, tag)
                    if tag not in p.tags:
                        p.tags.append(tag)
        p.save()


# POSTS HELPERS
def find_domain(source_url: str) -> str:
    """Find domain / twitter user from string

    Args:
        source_url (str): source url

    Returns:
        str: domain url
    """
    if source_url.startswith('http:') or source_url.startswith('https:'):
        if source_url.startswith('https://twitter.com/'):
            return source_url[max(source_url.find(u'//'), 0):source_url.find('/status')]
        else:
            return source_url[max(source_url.find(u'//'), 0):8+source_url[8:].find(u'/')]
    else:
        return source_url


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
    force_timestamp=True)

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
            'dump',
            'restore'
        ]

    def check_images(self, ctx: PluginContext, condition: str = 'width=0', *args):
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

        def __excluded_url(url):
            for pattern in config.excluded_urls:
                if re.search(pattern, url):
                    return True
            return False

        def __items(limit):
            rs = Item.query(F.flag.in_(10, 0) & F.storage.in_(
                None, False) & (F.url.regex('^https?://'))).sort(-F.id)
            ctx.log('count:', rs.count())
            if limit:
                rs = rs.limit(limit)
            for i in rs:
                p = Post.first(F.items == i.id)
                if not p:
                    i.delete()
                    continue
                if __excluded_url(i.url):
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
        q = Post.query(queryparser.eval(q))
        for t in tags.split(','):
            if t.startswith('~'):
                ret = q.update(Fn.pull(tags=t[1:]))
            else:
                ret = q.update(Fn.push(tags=t))
            ctx.log('updated', ret.modified_count)

    def call(self, ctx: PluginContext, *args):
        """Call api at localhost

        Args:
            ctx (PluginContext): plugin context
        """
        del os.environ['http_proxy']
        del os.environ['https_proxy']
        ctx.log(requests.get('http://localhost:8877/tools',
                             params={'action': args[0], 'args': ' '.join(args[1:])}).content.decode('utf-8'))

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
            for coll in colls or ('post', 'item', 'jav', 'auto_tags'):
                fo = BytesIO()
                ctx.log('dumping', coll, '...')
                for p in tqdm(mongodb(coll).find(), total=mongodb(coll).count()):
                    fo.write(jsonenc.encode(p).encode('utf-8') + b'\n')
                fo.seek(0)
                z.writestr(coll, fo.read())
        with open('auditor.log', 'w'):
            pass

    def restore(self, ctx: PluginContext, infile: str, *colls, force: bool = False):
        """Restore the status of database from a zip file of jsons.

        Args:
            ctx (PluginContext): plugin context
            infile (str): input zip filename.
            colls (optional): list of collections. Defaults to [].
            force (bool, optional): ignore errors. Defaults to False.
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
                mongodb(coll).insert_many(ps, ordered=False,
                                          bypass_document_validation=True)
            except Exception as ex:
                if not force:
                    ctx.log(ex)
                    exit()

        with zipfile.ZipFile(infile, 'r') as z:
            for coll in colls or z.namelist():
                ctx.log('restoring', coll, '...')
                fi = BytesIO(z.read(coll))
                ps = []
                for line in tqdm(fi):
                    p = json.loads(line.decode('utf-8'), object_hook=_hook)
                    ps.append(p)
                    if len(ps) > 100:
                        _save_db(coll, ps)
                        ps = []
                if ps:
                    _save_db(coll, ps)


def selected(solo=None):
    """Return selected posts and items from request argument `id` or `ids`

    Args:
        solo (bool or None, optional): Choose only selected items regardless of their siblings in the posts. Defaults to None, standing for read from request arguments.

    Returns:
        tuple: selected posts and items in corresponding order
    """    
    if solo is None:
        solo = argBool('solo')
    posts = []
    items = []
    pids = {}

    for pi in (arg('id', '') or arg('ids', '')).split(','):
        if not pi:
            continue
        p, i = pi.split('.', 1) if '.' in pi else (pi, '')
        if p:
            post = Post.first(F._id == ObjectId(p))
        else:
            post = Post.first(F.items == ObjectId(i))

        if post:
            if post.id in pids:
                post = pids[post.id]
            else:
                pids[post.id] = post
            posts.append(post)
            if solo:
                items += [item for item in post.items if not i or item.id ==
                          ObjectId(i)]
            else:
                items += post.items

    if not posts:
        abort(404)
    return posts, items


def register_gallery(app):

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
        
        return serve_file(p, ext, length)

    @app.route('/', methods=["GET", "POST"])
    @app.route('/<path:p>', methods=["GET", "POST"])
    def index(p='thumbs.html'):
        """Serve static files in the working directory

        Args:
            p (str, optional): path. Defaults to 'thumbs.html'.

        Returns:
            Response: the file, or return 404 if not found
        """    
        if p == 'favicon.ico':
            return '', 404
        elif os.path.exists(p):
            return serve_file(p)
        elif os.path.exists(p + '.html'):
            return serve_file(p + '.html')
        else:
            return serve_file('thumbs.html')

    def auto_tags(tags : List[str], source_url : str, delete=False):
        """Perform automatic tagging of posts based on their source URL

        Args:
            tags (list[str]): list of tags
            source_url (str): source URL pattern
            delete (bool): remove the rule if true
        """    
        at = mongodb('auto_tags')
        at.delete_many({'pattern': source_url, 'tag': {'$in': tags}})
        if not delete:
            at.insert_many([
                {'pattern': source_url, 'tag': t}
                for t in tags
            ])
            apply_auto_tags(Post.query(
                F.source_url.regex(source_url.replace('.', '\\.'))))

    @app.route('/group', methods=["GET", "POST"])
    def grouping():
        """Grouping selected posts
        
        Returns:
            Response: 'OK' if succeeded
        """    
        gh = lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()[-9:]
        posts, _ = selected()
        group = arg('group') or ''
        delete = argBool('del')
        batch = not argBool('solo')

        if batch:
            pu = find_domain(posts[0].source_url)
            auto_tags(['*' + group], pu, delete)

        if delete:
            for p in posts:
                dels = [_ for _ in p.tags if _.startswith('*')]
                for d in dels:
                    p.tags.remove(d)
                p.save()

        else:
            if not posts:
                return 'OK'
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

        return 'OK'

    @app.route('/tag', methods=["GET", "POST"])
    def tag():
        """Tagging selected posts

        Returns:
            Response: 'OK' if succeeded
        """    
        delete = argBool('del')
        batch = not argBool('solo')
        tags = arg('tag', '').split(',')
        posts, _ = selected()

        if batch:
            pu = find_domain(posts[0].source_url)
            auto_tags(tags, pu, delete)

        else:
            for p in posts:
                if delete:
                    if tags == ['@']:
                        p.tags = [_ for _ in p.tags if not _.startswith('@')]
                    elif tags == ['.']:
                        p.tags = []
                    else:
                        p.tags = list(set(p.tags) - set(tags))
                else:
                    p.tags = list(set(p.tags).union(tags))
                p.save()

        return 'OK'

    @app.route('/del', methods=["GET", "POST"])
    def delete():
        posts, items = selected(True)

        for p, i in zip(posts, items):
            if i in p.items:
                p.items.remove(i)

        for p in set(posts):
            if not p.items:
                print('empty post, delete', str(p.id))
                p.delete()
            else:
                p.save()

        for i in items:
            if Post.first(F.items == i.id):
                continue
            print('orphan item, delete', str(i.id))
            i.delete()

        Post.query(F.items == []).delete()

        return f'OK'
    
    @app.route('/tools', methods=["GET", "POST"])
    def tools_view():
        """Call tools

        Returns:
            Response: json document for a list of tools

        Yields:
            str: output logs of the running tool
        """    
        action = arg('action', '')
        if not action:
            return jsonify(sorted(tools.keys()))

        if action not in tools:
            return 'No such action.', 404

        def generate():
            """Generate log text from plugin context

            Yields:
                str: log text
            """        
            f = tools[action]
            args = [queryparser.expand_literals(_) for _ in arg('args', '').split(' ') if _]
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

    @app.route('/get', methods=["GET", "POST"])
    def get():
        """Get records

        Returns:
            Response: json document of records
        """    
        params = {
            'tag': '',
            'flag': 0,
            'post': '',
            'limit': 20,
            'offset': 0,
            'order': {'keys': ['random']},
            'direction': 'next',
            'groups': 0,
            'archive': 0
        }

        if request.json:
            params.update(**request.json)

        limit, offset = int(params['limit']), int(params['offset'])
        groups, archive = int(params['groups']) == 1, int(params['archive']) == 1

        if params['order'] == 'random':
            params['order'] = {'keys': ['random']}

        if 'keys' not in params['order']:
            params['order']['keys'] = []

        if 'random' not in params['order']['keys'] and '_id' not in params['order']['keys']:
            params['order']['keys'].append('_id')
            params['order']['_id'] = '000000000000000000000000'

        orders = [
            (o.split('-')[-1], -1 if o.startswith('-')
            == (params['direction'] == 'next') else 1)
            for o in params['order']['keys'] if o != 'random'
        ]

        order_conds = []
        vconds = {}
        unwind_first = False

        for k, dr in orders:
            v = params['order'].get(k)
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

        ands = {'$and': [_ for _ in [
            queryparser.eval(params['tag'])
        ] if _
        ] + (order_conds if not groups else [])
        }

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
            if ands['$and']:
                aggregate.match(ands)
            if not orders:
                aggregate.sample(size=limit*2)

        aggregate.lookup(
            from_='item', localField=F.items, foreignField=F._id, as_=F.items
        )

        if unwind_first:
            aggregate.unwind(Var.items).project(
                _id=1, liked_at=1, created_at=1, source_url=1, tags=1,
                items=[Var.items])

        if not match_first:
            aggregate.match(ands)

        if groups:
            if not orders:
                orders = [('liked_at', -1)]

            aggregate.project(
                _id=1, liked_at=1, created_at=1, items=1, source_url=1, tags=1,
                group_id=Fn.filter(input=Var.tags, as_='t',
                                cond=Fn.substrCP(Var._t, 0, 1) == '*')
            ).unwind(
                path=Var.group_id, preserveNullAndEmptyArrays=archive
            ).unwind(
                Var.items
            ).project(
                _id=1, liked_at=1, created_at=1, items=1, source_url=1, tags=1,
                group_id=Fn.ifNull(Var.group_id, Var.source_url)
            ).group(
                _id=Var.group_id,
                id=Fn.first(Var._id),
                liked_at=Fn.max(Var.liked_at),
                created_at=Fn.max(Var.created_at),
                source_url=Fn.first(Var.source_url),
                items=Fn.push(Var.items),
                tags=Fn.first(Var.tags),
                counts=Fn.sum(1),
                rating=Fn.max(Var['items.rating'])
            ).addFields(items=Fn.reduce(
                input=Var.items,
                initialValue=[],
                in_=Fn.concatArrays([Var._value, [Var._this]])
            )).project(
                counts=1, liked_at=1, created_at=1, source_url=1,
                items=1 if archive else [Fn.first(
                    Fn.filter(input=Var.items, as_='i', cond=Var['$i.rating'] == Var.rating))],
                _id=Var.id,
                group_id=Var._id,
                tags=1
            ).match(Fn.and_(order_conds))

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

        special_page = special_pages.get(params['post'].split('/')[0])
        if special_page:
            ret = special_page.special_page(aggregate, params,
                                            orders_params=orders_params) or ([], None, None)
            if isinstance(ret, tuple) and len(ret) == 3:
                rs, prev_order, next_order = ret
            else:
                return ret
        else:
            if orders:
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
                    _, dict) and _.get('flag', 0) == params['flag']]
                if not res['items']:
                    continue

                if not orders:  # random
                    res['items'] = random.sample(res['items'], 1)
                elif archive or groups or 'counts' in res:
                    cnt = res.get('counts', len(res['items']))
                    if cnt > 1:
                        res['count'] = '(+{})'.format(cnt)

                t = u''

                t += u'<a href="?archive=0&tag=source_url%25`{source_url_params}`" class="samepost" target="_blank">{source_url}</a>'.format(
                    source_url_params=quote(
                        re.sub(r'/\d+(/|$)', r'/.*\1', res['source_url'])),
                    **res
                )

                t += u' <a href="/?limit=50&tag=?{source_url_domain}" class="cog t_func" target="_blank">Cog.</a>'.format(
                    source_url_domain=find_domain(res['source_url']),
                )

                t += u' <a href="javascript:void(0);" onclick="searchByImage(this)" target="_blank">🔍</a>'

                res['text'] = t

                __apply_app_filters(res)
                if params['direction'] != 'next':
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
            o = dict(params['order'])
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
            'params': params,
            'prev': prev_order,
            'next': next_order,
            'results': r,
            '_filters': [
                ands, {'orders': orders, 'aggregate': aggregate.aggregators}
            ]
        })

    if config.gallery.get('proxy'):
        os.environ['http_proxy'] = os.environ['https_proxy'] = config.gallery.get('proxy')
    ctx = PluginContext('cmdline', join=True, logging_hook=print)
    register_plugins(app)
