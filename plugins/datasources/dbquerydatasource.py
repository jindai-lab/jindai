"""
Query Database
@zhs 来自数据库
"""
import datetime
import os
import re
import time
from collections import defaultdict
from typing import Iterable, List
from urllib.parse import urljoin
from bson import SON

from jindai import storage, parser, DBQuery
from jindai.models import MediaItem, Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai.storage import instance as storage


class DBQueryDataSource(DataSourceStage):
    """Query from Database
    @zhs 从数据库查询
    """

    def apply_params(self, query, req='', mongocollections='', limit=0, skip=0, sort='', raw=False, groups='none'):
        """
        Args:
            query (QUERY):
                Query expression, or keywords
                @zhs 查询字符串，或以 ? 开头的查询表达式，或以 ?? 开头的聚合查询表达式
            req (QUERY):
                Additional query expression
                @zhs 附加的条件
            sort (str):
                Sorting expression
                @zhs 排序表达式
            limit (int):
                Limitation for maximal results, 0 for none
                @zhs 查询最多返回的结果数量，默认为0即无限制
            skip (int):
                Skip %1 results
                @zhs 返回从第%1个开始的结果
            mongocollections (LINES):
                Name for colletion name in MongoDB, one item per line
                @zhs 数据库中其他数据集的名称
            raw (bool):
                Return dicts instead of Paragraph objects
                @zhs 若为 False（默认值）则返回 Paragraph 对象，否则返回原始数据，仅对于聚合查询有效
            groups (str):
                @choose(none|group|source|both) Groups
                @zhs @choose(无:none|按组:group|按来源:source|分组和来源:both) 分组
        """
        self.dbquery = DBQuery(
            query if not req else (query, req), mongocollections, limit, skip, sort, raw, groups)

    def fetch(self):
        return self.dbquery.fetch()


class MediaDataSource(DataSourceStage):
    """
    Data Source for Media Items
    @zhs 多媒体项目数据源
    """

    def apply_params(self, query='', limit=20, offset=0, raw=False, sort_keys='-id'):
        """
        Args:
            query (QUERY):
                Query expression
                @zhs 查询表达式
            limit (int):
                Limit
                @zhs 数量限制
            offset (int):
                Skipped results
                @zhs 跳过的结果数量
            raw (bool):
                Return dicts instead of MediaItem objects
                @zhs 返回字典而非 MediaItem
            sort_keys (str):
                Sorting expression
                @zhs 排序表达式
        """
        self.query_str = query
        self.query = parser.parse(query)
        self.raw = raw

        if not isinstance(self.query, list):
            self.query = [{'$match': self.query or {}}]

        if sort_keys == 'random':
            self.sort_keys = []
            self.query.append({'$sample': {'size': limit or 100}})
            limit = 0
        else:
            self.sort_keys = parser.parse_sort(sort_keys)

        self.result_set = MediaItem.aggregate(self.query, raw=raw)

        if self.sort_keys:
            self.result_set = self.result_set.sort(SON(self.sort_keys))
        if offset:
            self.result_set = self.result_set.skip(offset)
        if limit:
            self.result_set = self.result_set.limit(limit)

    def fetch(self) -> Iterable[MediaItem]:
        yield from self.result_set


class MediaImportDataSource(DataSourceStage):
    """
    Import media from datasource
    @zhs 从本地文件或网址导入图像信息
    """

    def apply_params(self, patterns, dataset='', tags='', proxy='', excluding_patterns=''):
        """
        Args:
            patterns (LINES):
                Import images from locations, one per line
                @zhs 网址或文件通配符
            excluding_patterns (LINES):
                Excluding patterns
                @zhs 排除的图片网址正则表达式
            tags (LINES):
                Tags, one tag per line
                @zhs 标签
            dataset (DATASET):
                Dataset name
                @zhs 数据集名称
            proxy (str):
                Proxy settings
                @zhs 代理服务器
        """
        self.keywords = PipelineStage.parse_lines(tags)
        weburl = re.compile('^https?://')
        patterns = PipelineStage.parse_lines(patterns)
        self.local_locs = [_ for _ in patterns if not weburl.match(_)]
        self.web_locs = [_ for _ in patterns if weburl.match(_)]
        self.proxies = {
            'http': proxy,
            'https': proxy
        } if proxy else {}
        self.excluding_patterns = [
            re.compile(pattern)
            for pattern in PipelineStage.parse_lines(excluding_patterns)
            if pattern
        ]
        self.dataset = dataset

    def fetch(self):
        if self.local_locs:
            yield from self.import_local(self.local_locs)
        if self.web_locs:
            for loc in self.web_locs:
                yield from self.import_page(loc)

    def import_local(self, locs) -> List[Paragraph]:
        """Import local files from locations

        :param locs: locations
        :type locs: List[str]
        :return: Paragraphs
        :rtype: List[Paragraph]
        :yield: Paragraph
        :rtype: Iterator[List[Paragraph]]
        """
        albums = defaultdict(Paragraph)

        for loc in storage.globs(locs):
            hashed_loc = re.sub(
                r'://(.+?)(:.+?)?@([\w.]+)/', r'://\1@\3/', loc)
            extname = loc.rsplit('.', 1)[-1].lower()
            dirname = hashed_loc.rsplit('/', 1)[0]
            filename = loc.split('#')[0]
            if extname in MediaItem.acceptable_extensions or loc.endswith('.mp4.thumb.jpg'):
                ftime = int(os.stat(filename).st_mtime) if os.path.exists(
                    filename) else time.time()
                album = albums[dirname]
                if not album.source:
                    album.source = {'url': filename}
                    album.keywords += self.keywords
                    album.pdate = datetime.datetime.utcfromtimestamp(ftime)
                    album.dataset = self.dataset

                i = MediaItem.get(
                    hashed_loc, item_type=MediaItem.get_type(extname))
                i.save()
                
                # do not write local file path
                # path = storage.default_path(i.id)
                # with storage.open(path, 'wb') as fout:
                #     fout.write(storage.open(loc, 'rb').read())
                # i.source = dict(i.source, file=path)
                # i.save()
                # self.log('Writing', i.id)
                album.images.append(i)

        albums = albums.values()

        yield from albums

    def import_page(self, paths):
        """Import images from web-page urls

        :param paths: url path(s), one path per line
        :type paths: str
        :yield: Paragraphs
        :rtype: Paragraph
        """
        albums = []
        imgset = set()

        for url in storage.globs(paths):
            p = Paragraph.get(url,
                              dataset=self.dataset,
                              images=[], pdate=datetime.datetime.utcnow())
            if url.endswith('.jpg'):
                imgs = [('', url)]
                title = ''
            else:
                self.log(url)
                html = storage.open(url, proxies=self.proxies).read()
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
                self.log(len(imgs), 'images found.')

            for _, img in imgs:
                imgurl = urljoin(url, img)
                for rep in self.excluding_patterns:
                    if rep.search(imgurl):
                        continue
                if imgurl not in imgset:
                    self.log(imgurl)
                    i = MediaItem.get(
                        imgurl, item_type=MediaItem.get_type(imgurl))
                    i.save()
                    p.images.append(i)
                    imgset.add(imgurl)
            p.keywords = list(set(self.keywords + title.split(u',')))
            albums.append(p)

        yield from albums
