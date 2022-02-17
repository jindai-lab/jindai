"""图集数据源"""
import datetime
import glob
import os
import re
import shutil
import zipfile
from collections import defaultdict
from typing import Iterable, List
from PIL import Image
from PyMongoWrapper import F, Fn, Var
from urllib.parse import urljoin
from bson import SON
from matplotlib import collections

from datasource import DataSource
from models import Album, ImageItem, ObjectId, try_download, parser
from storage import StorageManager
from models import try_download

weburl = re.compile('^https?://')


class GalleryAlbumDataSource(DataSource):
    """图集（图册）数据源"""

    def __init__(self, cond='', limit=20, offset=0, groups=False, archive=False, raw=False, sort_keys='-liked_at', direction='next', order={}):
        """
        Args:
            cond (QUERY): 检索表达式
            flag (int): 图像项目的标记
            limit (int): 最多返回的数量
            offset (int): 跳过记录的数量
            groups (bool): 是否按用户标记的分组返回
            archive (bool): 是否按用户标记的分组或来源分组返回
            sort_keys (str): 排序表达式
        """
        super().__init__()
        self.cond = cond
        self.query = parser.eval(cond)
        self.limit = limit
        self.offset = offset
        self.groups = groups
        self.archive = archive
        self.raw = raw

        # sorting order
        if order == 'random' or not isinstance(order, dict) or not isinstance(order.get('keys'), list):
            order = {'keys': []}
        
        if sort_keys and sort_keys != 'random':
            order['keys'] = sort_keys.split(',')
        
        if len(order['keys']) > 0 and 'random' not in order['keys'] and '_id' not in order['keys'] and '-_id' not in order['keys']:
            order['keys'].append('_id')
        
        self.orders = [
            (o.split('-')[-1], -1 if o.startswith('-') == (direction == 'next') else 1)
            for o in order['keys'] if o != 'random' and o != 'offset'
        ]
        self.order = order
        self.random = len(self.orders) == 0

        if self.random and self.limit == 0: self.limit = 100

        order_conds = []
        vconds = {}
        unwind_first = False

        for k, dr in self.orders:
            v = self.order.get(k)
            if v is None:
                continue
            if k.startswith('images.'):
                unwind_first = True
            if k == '_id':
                v = ObjectId(v)
            d = '$gt' if dr == 1 else '$lt'
            vconds[k] = {d: v}
            order_conds.append(dict(vconds))
            vconds[k] = v

        if order_conds:
            order_conds = [{'$or': order_conds}]

        ands = []

        query = parser.eval(self.cond)

        if isinstance(query, list) and len(query) > 0:
            ands = [query[0]] if query[0] else []
            query = query[1:]
        else:
            ands = [query] if query else []
            query = None

        ands += order_conds if (not self.groups and not self.archive) else []
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
                        if 'images.' in k:
                            flag = True
                        elif k == 'images' and isinstance(v, dict):
                            flag = sum([1 for k_ in v if not k_.startswith('$')])
                        else:
                            stk.append(v)
            return flag

        match_first = not __examine_dict(ands)
        if not match_first:
            unwind_first = True

        self.aggregator = Album.aggregator
        if match_first:
            if ands:
                self.aggregator.match(ands)
            if not self.orders:
                self.aggregator.sample(size=limit*2)

        self.aggregator.lookup(
            from_='imageitem', localField=F.images, foreignField=F._id, as_=F.images
        )

        if unwind_first:
            self.aggregator.unwind(Var.images).addFields(
                images=[Var.images])

        if not match_first and ands:
            self.aggregator.match(ands)

        if query:
            self.aggregator.aggregators += query

        if groups or archive:
            if not self.orders:
                self.orders = [('liked_at', -1)]

            self.aggregator.addFields(
                group_id=Fn.filter(input=Var.keywords, as_='t',
                                cond=Fn.substrCP(Var._t, 0, 1) == '*')
            ).unwind(
                path=Var.group_id, preserveNullAndEmptyArrays=archive
            ).unwind(
                Var.images
            ).addFields(
                group_id=Fn.ifNull(Var.group_id, Fn.concat('id=`', Fn.toString(Var._id), '`')),
                **{'images.album_id': '$_id'}
            ).group(
                _id=Var.group_id,
                id=Fn.first(Var._id),
                liked_at=Fn.max(Var.liked_at),
                pdate=Fn.max(Var.pdate),
                source=Fn.first(Var.source),
                images=Fn.addToSet(Var.images),
                author=Fn.first(Var.author),
                keywords=Fn.first(Var.keywords),
                counts=Fn.sum(1),
                rating=Fn.max(Var['images.rating'])
            ).addFields(
                images='$images' if archive else [Fn.first(
                    Fn.filter(input=Var.images, as_='i', cond=Var['$i.rating'] == Var.rating))],
                _id=Var.id,
                group_id=Var._id
            )
            if archive and order_conds: self.aggregator.match({'$and': order_conds})
        
        if self.orders:
            self.orders_params = {
                'order_conds': order_conds,
                'orders': SON(self.orders),
                'offset': offset,
                'limit': limit
            }
        else:
            self.orders_params = {}


    def fetch(self) -> Iterable[Album]:
        if self.random:
            self.aggregator.sample(size=self.limit)
        else:
            if not self.query or not [_ for _ in self.query if '$sort' in _]:
                self.aggregator.sort(SON(self.orders))
                if self.offset:
                    self.aggregator.skip(self.offset)
            if self.limit: self.aggregator.limit(self.limit)
        
        yield from self.aggregator.perform(raw=self.raw)


class GalleryImageItemDataSource(DataSource):
    """图像项目数据源"""

    def __init__(self, cond='', limit=20, offset=0, raw=False, sort_keys='-_id'):
        """
        Args:
            cond (QUERY): 查询表达式
            limit (int): 数量限制
            offset (int): 跳过的结果数量
            raw (bool): 返回字典而非 ImageItem
            sort_keys (str): 排序表达式
        """
        super().__init__()
        self.cond = cond
        self.query = parser.eval(cond)
        self.raw = raw
        self.sort_keys = sort_keys.split(',')
        self.rs = ImageItem.query(self.query)
        if offset:
            self.rs = self.rs.skip(offset)
        if limit:
            self.rs = self.rs.limit(limit)

    def fetch(self) -> Iterable[ImageItem]:
        yield from self.rs


class ImageImportDataSource(DataSource):
    """从本地文件或网址导入图像到图集
    """

    def __init__(self, locs, dataset='默认图集', tags='', proxy='', excluding_patterns=''):
        """
        Args:
            locs (str): 网址或文件通配符，一行一个
            excluding_patterns (str): 排除的图片网址正则表达式，一行一个
            tags (str): 标签，一行一个
            dataset (DATASET): 数据集名称
            proxy (str): 代理服务器
        """
        super().__init__()
        self.keywords = tags.split('\n')
        locs = locs.split('\n')
        self.local_locs = [_ for _ in locs if not weburl.match(_)]
        self.web_locs = [_ for _ in locs if weburl.match(_)]
        self.proxies = {
            'http': proxy,
            'https': proxy
        } if proxy else {}
        self.excluding_patterns = [re.compile(pattern) for pattern in excluding_patterns.split('\n') if pattern]
        self.dataset = dataset

    def fetch(self):
        if self.local_locs:
            yield from self.import_local(self.local_locs)
        if self.web_locs:
            for loc in self.web_locs:
                yield from self.import_page(loc)

    def import_local(self, locs) -> List[Album]:
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
                    self.logger(loc)
                    with zipfile.ZipFile(loc, 'r') as z:
                        l += [(loc, loc + '#' + _) for _ in z.namelist()]
                        z.extractall('__zip{}'.format(hash(loc)))
                elif os.path.isdir(loc):
                    self.logger(loc)
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

        albums = defaultdict(Album)

        with StorageManager() as mgr:
            for loc, _f in sorted(__list_all(locs)):
                if _f.split('.')[-1] in ['txt', 'log', 'xlsx', 'xls', 'zip', 'csv'] or _f.endswith('.mp4.thumb.jpg'):
                    continue
                pu = loc.split('/')[-1]
                ftime = __get_mtime(_f)

                p = albums[pu]
                if not p.source:
                    p.source = {'url': pu}
                    p.keywords += self.keywords
                    p.pdate = datetime.datetime.utcfromtimestamp(ftime)
                    p.dataset = self.dataset

                i = ImageItem(source={'url': _f})
                fn = __expand_zip(_f)
                if not _f.lower().endswith(('.mp4', '.avi')):
                    try:
                        im = Image.open(fn)
                        i.width, i.height = im.size
                    except Exception as ex:
                        self.logger('Error while handling', fn, ':', ex)
                        continue
                
                i.save()
                if mgr.write(fn, i.id):
                    i.source = dict(i.source, file='blocks.h5')
                i.save()
                p.images.append(i)

        albums = albums.values()

        for _ in glob.glob('__zip*'):
            shutil.rmtree(_)
        for _ in zips:
            os.unlink(_)

        yield from albums

    def import_page(self, path, rng_start=0, rng_end=0):
        """Import images from web-page urls

        Args:
            path (str): url path
            rng_start (int, optional): [description]
            rng_end (int, optional): [description]
        """
        albums = []

        rng = ['']
        rngmatch = re.match(r'(.+##.*)\s(\d+)-(\d+)$', path)
        if rngmatch:
            path, rng_start, rng_end = rngmatch.groups()
            rng = range(int(rng_start), int(rng_end)+1)

        imgset = set()

        for i in rng:
            url = path.replace('##', str(i))
            p = Album.first(F.source == {'url': url}) or Album(
                dataset=self.dataset,
                source={'url': url}, images=[], pdate=datetime.datetime.utcnow())
            if url.endswith('.jpg'):
                imgs = [('', url)]
                title = ''
            else:
                self.logger(url)
                html = try_download(url, proxies=self.proxies)
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
                self.logger(len(imgs), 'images found.')

            for _, img in imgs:
                imgurl = urljoin(url, img)
                for rep in self.excluding_patterns:
                    if rep.search(imgurl): continue
                if imgurl not in imgset:
                    self.logger(imgurl)
                    i = ImageItem.first(F.source == {'url': imgurl}) or ImageItem(source={'url': imgurl})
                    i.save()
                    p.images.append(i)
                    imgset.add(imgurl)
            p.keywords = list(set(self.keywords + title.split(u',')))
            albums.append(p)
        
        yield from albums