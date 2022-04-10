"""来自数据库
"""
import datetime
import glob
import os
import re
import shutil
import zipfile
from collections import defaultdict
from typing import Iterable, List
from urllib.parse import urljoin
from bson import SON
from PIL import Image
from PyMongoWrapper import F

from jindai import safe_open, DBQuery
from jindai.models import ImageItem, Paragraph, parser
from jindai.pipeline import DataSourceStage

class DBQueryDataSource(DataSourceStage):
    """从数据库查询
    """
    class Implementation(DataSourceStage.Implementation):
        """Implementing datasource"""

        def __init__(self, query, mongocollections='', limit=0, skip=0, sort='', raw=False, groups='none'):
            """
            Args:
                query (QUERY): 查询字符串，或以 ? 开头的查询表达式，或以 ?? 开头的聚合查询表达式
                sort (str): 排序表达式
                limit (int): 查询最多返回的结果数量，默认为0即无限制
                skip (int): 返回从第%1个开始的结果
                mongocollections (str): 数据库中其他数据集的名称，一行一个
                raw (bool): 若为 False（默认值）则返回 Paragraph 对象，否则返回原始数据，仅对于聚合查询有效
                groups (无:none|按组:group|按来源:source|分组和来源:both): 分组
            """
            self.dbquery = DBQuery(query, mongocollections, limit, skip, sort, raw, groups)

        def fetch():
            return self.dbquery.fetch()


class ImageItemDataSource(DataSourceStage):
    """图像项目数据源"""
    class Implementation(DataSourceStage.Implementation):
        """Implementing datasource"""

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
            if self.sort_keys:
                self.rs = self.rs.sort(*self.sort_keys)
            if offset:
                self.rs = self.rs.skip(offset)
            if limit:
                self.rs = self.rs.limit(limit)

        def fetch(self) -> Iterable[ImageItem]:
            yield from self.rs


class ImageImportDataSource(DataSourceStage):
    """从本地文件或网址导入图像到图集
    """

    class Implementation(DataSourceStage.Implementation):
        """Implementing datasource"""

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

            weburl = re.compile('^https?://')
            locs = locs.split('\n')
            self.local_locs = [_ for _ in locs if not weburl.match(_)]
            self.web_locs = [_ for _ in locs if weburl.match(_)]
            self.proxies = {
                'http': proxy,
                'https': proxy
            } if proxy else {}
            self.excluding_patterns = [re.compile(
                pattern) for pattern in excluding_patterns.split('\n') if pattern]
            self.dataset = dataset

        def fetch(self):
            if self.local_locs:
                yield from self.import_local(self.local_locs)
            if self.web_locs:
                for loc in self.web_locs:
                    yield from self.import_page(loc)

        def import_local(self, locs) -> List[Paragraph]:
            zips = []

            def _expand_zip(src):
                if '.zip#' in src:
                    src = os.path.join('__zip{}'.format(
                        hash(src[:src.find('#')])), src[src.find('#')+1:])
                return src

            def _list_all(locs):
                locs = list(locs)
                l = []
                for loc in locs:
                    if '*' in loc:
                        locs += glob.glob(loc)
                    elif loc.endswith('.zip'):
                        zips.append(loc)
                        self.logger(loc)
                        with zipfile.ZipFile(loc, 'r') as z:
                            l += [(loc, loc + '#zip/' + _)
                                  for _ in z.namelist()]
                            z.extractall('__zip{}'.format(hash(loc)))
                    elif os.path.isdir(loc):
                        self.logger(loc)
                        l += [(loc, os.path.join(loc, _))
                              for _ in os.listdir(loc)]
                    elif os.path.isfile(loc):
                        l.append((loc, loc))
                return l

            def __get_mtime(src):
                if '.zip#' in src:
                    src = src[:src.find('#')]
                return int(os.stat(src).st_mtime)

            for _ in glob.glob('._*'):
                os.unlink(_)

            albums = defaultdict(Paragraph)

            for loc, _f in sorted(_list_all(locs)):
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
                fn = _expand_zip(_f)
                i.save()
                with safe_open(f'hdf5://{i.id}', 'wb') as fo:
                    fo.write(open(fn).read())

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
                p = Paragraph.first(F.source == {'url': url}) or Paragraph(
                    dataset=self.dataset,
                    source={'url': url}, images=[], pdate=datetime.datetime.utcnow())
                if url.endswith('.jpg'):
                    imgs = [('', url)]
                    title = ''
                else:
                    self.logger(url)
                    html = safe_open(url, proxies=self.proxies).read()
                    assert html, 'Download failed.'
                    try:
                        html = html.decode('utf-8')
                    except Exception:
                        try:
                            html = html.decode('gbk')
                        except Exception:
                            try:
                                html = html.decode('euc-jp')
                            except Exception:
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
                        if rep.search(imgurl):
                            continue
                    if imgurl not in imgset:
                        self.logger(imgurl)
                        i = ImageItem.first(F.source == {'url': imgurl}) or ImageItem(
                            source={'url': imgurl})
                        i.save()
                        p.images.append(i)
                        imgset.add(imgurl)
                p.keywords = list(set(self.keywords + title.split(u',')))
                albums.append(p)

            yield from albums
