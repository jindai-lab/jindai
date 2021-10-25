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


def register_gallery(app):

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
        
    if config.gallery.get('proxy'):
        os.environ['http_proxy'] = os.environ['https_proxy'] = config.gallery.get('proxy')
    ctx = PluginContext('cmdline', join=True, logging_hook=print)
    register_plugins(app)
