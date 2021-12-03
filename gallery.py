import base64
import datetime
import glob
import hashlib
import json
import os
import random
import re
import shutil
import time
import zipfile
from collections import defaultdict
from io import BytesIO
from multiprocessing.pool import ThreadPool
from typing import IO, Any, Callable, Iterable, List, Tuple, Union
import requests
from bson import ObjectId
from flask import Response, abort, jsonify, request, stream_with_context
from PIL import Image, ImageEnhance, ImageStat
from PyMongoWrapper import F, Fn, Var
from tqdm import tqdm

import config
from datasources.gallerydatasource import GalleryAlbumDataSource, queryparser
from helpers import *
from models import Album, AutoTag, ImageItem, MongoJSONEncoder
from plugin import Plugin
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


def single_item(pid: str, iid: str) -> List[Album]:
    """Return a single-item album object with id = `pid` and item id = `iid`

    Args:
        pid (str): Album ID
        iid (str): ImageItem ID

    Returns:
        List[Album]: a list with at most one element, i.e., the single-item album object
    """
    if pid:
        pid = ObjectId(pid)
        p = Album.first(F.id == pid)
    elif iid:
        iid = ObjectId(iid)
        p = Album.first(F.items == iid)

    if iid and p:
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


def apply_auto_tags(albums):
    """Apply auto tags to albums

    Args:
        albums (Iterable[Album]): album objects
    """
    log = print if ctx else print
    m = list(AutoTag.query({}))
    if not m:
        return
    for p in tqdm(albums):
        for i in m:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in p.tags) or (pattern and re.search(pattern, p.source['url'])):
                if tag not in p.tags:
                    p.tags.append(tag)
                if tag.startswith('@'):
                    p.author = tag
        p.save()


# ALBUM HELPERS
def Q(tags: str) -> Iterable[Album]:
    """Shortcut for querying albums with items resolved

    Args:
        tags (str): filtering expression

    Returns:
        Iterable[Album]: albums matching `tags`
    """
    ag = Album.aggregator
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


# prepare plugins
plugins = []
special_pages = {}
callbacks = defaultdict(list)


def register_plugins(app):
    """Register plugins in config

    Args:
        app (Flask): Flask app object
    """
    import plugins as _plugins

    for pl in config.gallery.get('plugins'):
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
    
    # STATIC AND RESOURCES PROVIDERS
    @app.route('/api/gallery/block/<id>.<ext>')
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
        except OSError:
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


    # ITEM OPERATIONS
    @app.route('/api/gallery/imageitem/rating', methods=["GET", "POST"])
    @rest()
    def fav(items, inc=1, val=0):
        """Increase or decrease the rating of selected items

        Returns:
            Response: 'OK' if succeeded
        """
        
        items = list(ImageItem.query(F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in items])))
        for i in items:
            if i is None: continue
            i.rating = int(i.rating + inc) if inc else val
            if -1 <= i.rating <= 5:
                i.save()
        return {
            str(i.id): i.rating
            for i in items
        }


    @app.route('/api/gallery/imageitem/reset_storage', methods=["GET", "POST"])
    @rest()
    def reset_storage(items):
        """Reset storage status of selected items

        Returns:
            Response: 'OK' if succeeded
        """
        
        items = list(ImageItem.query(F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in items])))
        for i in items:
            i.storage = not i.storage
            i.save()
        return {
            str(i.id): i.storage
            for i in items
        }


    @app.route('/api/gallery/imageitem/merge', methods=["POST"])
    @rest()
    def merge_items(pairs):
        for rese, dele in pairs:
            dele = ImageItem.first(F.id == dele)
            if not dele:
                continue
            
            if rese:
                pr = Album.first(F.items == ObjectId(rese)) or Album(
                    items=[ObjectId(rese)])
                for pd in Album.query(F.items == dele.id):
                    pr.tags += pd.tags
                    if (not pr.source.get('url') or 'restored' in pr.source['url']) and pd.source.get('url'):
                        pr.source = pd.source
                    
                pr.save()
            
            Album.query(F.items == dele.id).update(Fn.pull(items=dele.id))        
            dele.delete()

        Album.query(F.items == []).delete()

        return True


    @app.route('/api/gallery/imageitem/delete', methods=["POST"])
    @rest()
    def delete_item(album_items: dict):
        for pid, items in album_items.items():
            p = Album.first(F.id == ObjectId(pid))
            if p is None: continue

            items = list(map(ObjectId, items))
            p.items = [_ for _ in p.items if _.id not in items]        
            p.save()

        for i in items:
            if Album.first(F.items == i):
                continue
            print('orphan item, delete', str(i))
            ImageItem.first(F.id == i).delete()

        Album.query(F.items == []).delete()

        return True


    # ALBUM OPERATIONS
    @app.route('/api/gallery/album/split', methods=["GET", "POST"])
    @app.route('/api/gallery/album/merge', methods=["GET", "POST"])
    @rest()
    def splitting(albums):
        """Split or merge selected items/albums into seperate albums/one album

        Returns:
            Response: 'OK' if succeeded
        """        
        albums = list(Album.query(F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in albums])))

        if request.path.endswith('/split'):
            for p in albums:
                for i in p.items:
                    pnew = Album(source={'url': p.source['url']}, liked_at=p.liked_at,
                                pdate=p.pdate, tags=p.tags, items=[i])
                    pnew.save()
                p.delete()
        else:
            if not albums: return False
            
            p0 = albums[0]
            p0.tags = list(p0.tags)
            p0.items = list(p0.items)
            for p in albums[1:]:
                p0.tags += list(p.tags)
                p0.items += list(p.items)
            p0.save()
            
            for p in albums[1:]:
                p.delete()

        return True


    @app.route('/api/gallery/album/group', methods=["GET", "POST", "PUT"])
    @rest()
    def grouping(albums, group='', delete=False):
        """Grouping selected albums

        Returns:
            Response: 'OK' if succeeded
        """
        def gh(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()[-9:]
        
        albums = list(Album.query(F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in albums])))

        if delete:
            group_id = ''
            for p in albums:
                p.tags = [_ for _ in p.tags if not _.startswith('*')]
                p.save()

        else:
            if not albums:
                return True
            gids = []
            for p in albums:
                gids += [_ for _ in p.tags if _.startswith('*')]
            named = [_ for _ in gids if not _.startswith('*0')]

            if group:
                group_id = '*' + group
            elif named:
                group_id = min(named)
            elif gids:
                group_id = min(gids)
            else:
                group_id = '*0' + gh(min(map(lambda p: str(p.id), albums)))

            for p in albums:
                if group_id not in p.tags:
                    p.tags.append(group_id)
                    p.save()

            gids = list(set(gids) - set(named))
            if gids:
                for p in Album.query(F.tags.in_(gids)):
                    for id0 in gids:
                        if id0 in p.tags:
                            p.tags.remove(id0)
                    if group_id not in p.tags:
                        p.tags.append(group_id)
                    p.save()

        return group_id


    @app.route('/api/gallery/album/tag', methods=["GET", "POST"])
    @rest()
    def tag(albums, delete=[], append=[]):
        """Tagging selected albums

        Returns:
            Response: 'OK' if succeeded
        """
        
        albums = list(Album.query(F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in albums])))
        for p in albums:
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
            for p in albums
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
                for _ in Album.aggregator.match(matcher).unwind('$tags').match(matcher).group(_id='$tags').perform(raw=True)
                if len(_['_id']) < 15
            ]


    # AUTO_TAGS OPERATIONS
    @app.route('/api/gallery/auto_tags', methods=["PUT"])
    @rest()
    def auto_tags_create(tag: str = '', pattern: str = '', from_tag : str = None):
        """Perform automatic tagging of albums based on their source URL

        Args:
            tags (list[str]): list of tags
            pattern (str): source URL pattern
            delete (bool): remove the rule if true
        """
        assert tag and (pattern or from_tag), 'Must specify tag with pattern or from tag'
        AutoTag.query(F.pattern.eq(pattern) & F.from_tag.eq(from_tag) & F.tag.eq(tag)).delete()
        AutoTag(pattern=pattern, from_tag=from_tag, tag=tag).save()
        albums = Album.query(F['source.url'].regex(pattern)) if pattern else Album.query(F.tags == from_tag)
        apply_auto_tags(albums)

        return True


    @app.route('/api/gallery/auto_tags', methods=["GET", "POST"])
    @rest()
    def auto_tags(ids : List[str] = [], delete=False):
        """List or delete automatic tagging of albums based on their source URL

        Args:
            ids (list[id]): list of ids
        """
        if delete:
            AutoTag.query(F.id.in_([ObjectId(_) for _ in ids])).delete()
            return True
        else:
            return [_.as_dict() for _ in AutoTag.query({}).sort(-F.id)]


    @app.route('/api/gallery/get', methods=["GET", "POST"])
    @rest()
    def get(query='', flag=0, post='', limit=20, offset=0, order={'keys':['random']}, direction='next', groups=False, archive=False, count=False):
        """Get records

        Returns:
            Response: json document of records
        """
        ds = GalleryAlbumDataSource(query, limit, offset, groups, archive, True, '', direction, order)
        
        if count:
            return (list(ds.aggregator.group(_id='', count=Fn.sum(1)).perform(raw=True)) + [{'count': 0}])[0]['count']

        prev_order, next_order = {}, {}

        post_args = post.split('/')
        special_page = special_pages.get(post_args[0])
        if special_page:
            ret = special_page.special_page(ds, post_args) or ([], None, None)
            if isinstance(ret, tuple) and len(ret) == 3:
                rs, prev_order, next_order = ret
            else:
                return ret
        else:
            rs = ds.fetch()

        r = []
        res = {}
        try:
            for res in rs:
                if isinstance(res, Album):
                    res = res.as_dict(expand=True)

                if 'count' not in res:
                    res['count'] = ''

                if '_id' not in res or not res['items']:
                    continue

                res['items'] = [_ for _ in res['items'] if isinstance(
                    _, dict) and _.get('flag', 0) == flag]
                if not res['items']:
                    continue

                if ds.random:
                    res['items'] = random.sample(res['items'], 1)
                elif archive or groups or 'counts' in res:
                    cnt = res.get('counts', len(res['items']))
                    if cnt > 1:
                        res['count'] = '(+{})'.format(cnt)

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
                '_filters': ds.aggregator.aggregators
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
            '_filters': ds.aggregator.aggregators
        })

    @app.route('/api/gallery/plugins/style.css')
    def plugins_style():
        """Returns css from all enabled plugins

        Returns:
            Response: css document
        """
        css = '\n'.join([p.run_callback('css')
                        for p in callbacks['css']])
        return Response(css, mimetype='text/css')

    @app.route('/api/gallery/plugins/script.js')
    def plugins_script():
        """Returns js scripts from all enabled plugins

        Returns:
            Response: js document
        """
        js = '\n'.join([p.run_callback('js')
                    for p in callbacks['js']])
        return Response(js, mimetype='text/javascript')

    @app.route('/api/gallery/plugins/special_pages', methods=["GET", "POST"])
    @rest()
    def plugins_special_pages():
        """Returns names for special pages in every plugins
        """
        return list(special_pages.keys())

    register_plugins(app)
