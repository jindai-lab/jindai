"""
Import from web or file
@zhs 来自网页或文本文件
"""

import codecs
from collections import deque
import re
import json
import datetime
from concurrent.futures import Future, ThreadPoolExecutor
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup as B

from PyMongoWrapper import F, Fn
from jindai.models import MediaItem, Paragraph
from jindai.pipeline import DataSourceStage
from jindai import storage, parser


class TextDataSource(DataSourceStage):
    """
    Read Paragraphs from text files
    @zhs 从文本文件中读取语段
    """

    def apply_params(self, dataset_name='', lang='auto', content=''):
        """
        Args:
            dataset_name (DATASET):
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (FILES):
                Paths
                @zhs HTML或包含HTML的ZIP压缩包文件列表
        """
        self.name = dataset_name
        self.lang = lang
        self.content = content.split('\n')

    def fetch(self):
        for path in storage.globs(self.content):
            for i, line in enumerate(storage.open(path)):
                yield Paragraph(content=codecs.decode(line),
                                source={
                                    'url' if '://' in path else 'file': storage.truncate_path(path)},
                                dataset=self.name, lang=self.lang, outline=f'{i+1:06d}')


class LinesDataSource(DataSourceStage):
    """
    Import paragraphs from lines
    @zhs 从直接输入的文本中获得语段，每行一个语段
    """

    def apply_params(self, dataset_name='', lang="auto", content="", params=None, delimiter='\n'):
        """
        Args:
            dataset_name (DATASET):
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (str):
                Text contents
                @zhs 文本内容
            params (object):
                Other customizable fields
                @zhs 其他自定义字段
            delimiter (str):
                Delimiter
                @zhs 分隔符，默认为换行符
        """
        self.name = dataset_name
        self.lang = lang
        self.lines = content.split(delimiter)
        self.params = params or {}

    def fetch(self):
        return map(lambda x: Paragraph(
            content=x, lang=self.lang, dataset=self.name, **self.params), self.lines)


DEFAULT_IMG_PATTERNS = 'img[src]|[zoomfile]|[data-original]|[data-src]|[file]|[data-echo]'.replace(
    '|', '\n')


class WebPageListingDataSource(DataSourceStage):
    """
    Import web page listings
    @zhs 从网页列表中导入语段
    """

    def apply_params(self, dataset='', patterns='',
                     mongocollection='', lang='auto', detail_link='',
                     list_link='', proxy='', list_depth=1, tags='',
                     parallel_n=10,
                     img_pattern=DEFAULT_IMG_PATTERNS) -> None:
        """
        Args:
            dataset (DATASET):
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            patterns (LINES):
                Patterns for web pages
                @zhs 列表页面模式
            list_depth (int):
                List depth
                @zhs 列表深度
            proxy (str):
                Proxy settings
                @zhs 代理服务器
            tags (LINES):
                Tags, one tag per line
                @zhs 标签
            detail_link (str):
                Regex for detailed page URL
                @zhs 详情页面正则表达式
            list_link (str):
                Regex for listing page URL
                @zhs 列表页面正则表达式
            img_pattern (LINES):
                Image pattern
                @zhs 图像检索标记
            mongocollection (str):
                Mongo Collection name
                @zhs 数据库集合名
            parallel_n (int):
                Parallel download threads
                @zhs 并行下载线程数
        """
        self.proxies = {} if not proxy else {
            'http': proxy,
            'https': proxy
        }
        self.patterns = patterns.split('\n')
        self.list_depth = list_depth
        self.detail_link = re.compile(detail_link)
        self.list_link = re.compile(list_link)
        self.tags = tags.split('\n')
        self.dataset = dataset
        self.lang = lang
        self.image_patterns = [_ for _ in img_pattern.split('\n') if _] or DEFAULT_IMG_PATTERNS.split('\n')
        self.mongocollection = mongocollection
        self.convert = Paragraph.get_converter(mongocollection)
        self.n = parallel_n

    def read_html(self, url):
        """Read html from url, return BeautifulSoup object"""
        try:
            html = storage.open(url, proxies=self.proxies).read()
        except OSError:
            self.logger('Cannot read from', url)
            html = b''

        b = B(html, 'lxml')
        return b

    def get_text(self, element):
        """Get text of element"""
        if element and element.text:
            return re.sub(r'\s+', ' ', element.text)
        return ''

    def parse_detail(self, url, b):
        """Parse url as a detail page"""

        if not b:
            return

        para = Paragraph.get(url, pdate=datetime.datetime.utcnow(),
                             source={'url': url},
                             dataset=self.dataset, lang=self.lang)
        para.content = self.get_text(b)
        para.html = str(b)
        para.keywords = self.tags
        para.title = self.get_text(b.find('title'))

        items = set()

        for imgp in self.image_patterns:
            for i in b.select(imgp):
                attr_m = re.search(r'\[(.+)\]', imgp)
                if attr_m:
                    attr = i[attr_m.group(1)]
                else:
                    attr = self.get_text(i)
                upath = urljoin(url, attr)
                image = MediaItem.get(upath, item_type=MediaItem.get_type(
                    upath.rsplit('.', 1)[-1]) or 'image')
                if upath in items:
                    continue
                if image.id:
                    Paragraph.query(F.images == image.id).update(
                        Fn.pull(images=image.id))
                items.add(upath)

                para.images.append(image)

        self.logger(f'Found {len(para.images)} images in {url}')
        return self.convert(para)

    def parse_list(self, url, b):
        """Parse url as a list page"""

        if not b:
            self.logger(f'Cannot read list from {url}')
            return []

        for a in b.select('a[href]'):
            a['href'] = urljoin(url, a['href'])

        links = {
            a['href'].split('#')[0]
            for a in b.select('a[href]')
        }
        
        self.logger(len(links), 'links')

        links = [u for u in links if self.list_link.search(
            u) or self.detail_link.search(u)]
        self.logger(len(links), 'matched list or detail pattern')

        return links

    def fetch(self):
        queue = deque([(p, 1)
                       for p in storage.globs(self.patterns)])
        visited = set()

        def _do_parse(url, level):
            if url in visited:
                self.logger('found', url, 'visited, skip')
                return
            visited.add(url)
            para = Paragraph.get(url)
            if para.id:
                return

            b = self.read_html(url)

            if self.list_link.search(url) or self.detail_link.search(url):
                self.logger('parse as list', url)
                for upath in self.parse_list(url, b):
                    yield upath, level + 1

            if self.detail_link.search(url):
                self.logger('parse as detail', url)
                para = self.parse_detail(url, b)
                if para:
                    yield para, level + 1

        running_futures = set()
        results = deque()

        def _enqueue(fut: Future):
            riter = fut.result()
            running_futures.remove(id(fut))
            for res, level in riter:
                if isinstance(res, Paragraph):
                    results.append(res)
                elif isinstance(res, str):
                    if res not in visited and res not in queue and level <= self.list_depth:
                        self.logger('add', res, 'to queue')
                        queue.append((res, level))

        tpe = ThreadPoolExecutor(self.n)

        while queue or running_futures or results:
            if queue:
                self.logger(f'Queuing {len(queue)} urls')
                arg = queue.popleft()
                future = tpe.submit(_do_parse, *arg)
                running_futures.add(id(future))
                future.add_done_callback(_enqueue)
            if running_futures:
                time.sleep(0.1)
            while results:
                yield results.popleft()


class JSONDataSource(DataSourceStage):
    """Parse JSON data to Paragraphs, used to interact with web interface
    @zhs 从 JSON 数据解析出语段，用于与网页客户端交互
    """

    def apply_params(self, content='', **kwargs) -> None:
        """
        :param content: JSON data
        :type content: str
        :yield: Paragraphs
        :rtype: Paragraph
        """
        self.content = json.loads(content)
        if isinstance(self.content, dict) and 'results' in self.content:
            self.content = self.content['results']
        if not isinstance(self.content, list):
            self.content = [self.content]

    def fetch(self):
        for paragraph in self.content:
            yield Paragraph().fill_dict(paragraph)


class BiblioDataSource(DataSourceStage):
    """
    Import paragraph from EndNote bibliography items
    @zhs 从 EndNote 文献条目产生语段
    """
    
    def apply_params(self, content='', dataset_name='', lang='zhs', input_format='endnote') -> None:
        """
        Args:
            dataset_name (DATASET):
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (FILES):
                Paths
                @zhs 文件列表
            format (endnote|other, unsupported):
                Format
                @zhs 文献条目信息格式
        """
        if not hasattr(self, input_format):
            raise NotImplementedError()

        self.method = getattr(self, input_format)
        self.files = storage.globs(content)
        self.dataset = dataset_name
        self.lang = lang

    def endnote(self, lines):
        """"Parse EndNote format"""

        doc = {
            'content': '',
            'authors': []
        }
        field = ''
        for line in lines:
            if not line.strip():
                if doc:
                    yield Paragraph(dataset=self.dataset, lang=self.lang, **doc)
                doc = {
                    'content': '',
                    'authors': []
                }
            else:
                line = line.decode('utf-8').strip()
                if ' ' not in line:
                    value = line
                else:
                    field, value = line.split(' ', 1)
                    field = {
                        '%0': 'item_type',
                        '%A': 'authors',
                        '%+': 'institutions',
                        '%J': 'journal',
                        '%D': 'pdate',
                        '%T': 'title',
                        '%N': 'issue',
                        '%K': 'tags',
                        '%X': 'content',
                        '%P': 'pages',
                        '%@': 'issn',
                        '%L': 'cn_publishing_number',
                        '%W': 'catalog'
                    }.get(field.upper(), 'content')

                if ';' in value and field != 'content':
                    value = [_ for _ in value.split(';') if _]
                if field in doc:
                    if field == 'content':
                        doc[field] += value
                    else:
                        if not isinstance(doc[field], list):
                            doc[field] = [doc[field]]
                        if isinstance(value, list):
                            doc[field] += value
                        else:
                            doc[field].append(value)
                else:
                    doc[field] = value
        if doc:
            yield Paragraph(dataset=self.dataset, **doc)

    def fetch(self):
        for file in self.files:
            yield from self.method(file)
