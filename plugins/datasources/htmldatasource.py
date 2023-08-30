"""
Import from web or file
@zhs 来自网页或文本文件
"""

import re
import json
import datetime
import tempfile
import os
from hashlib import sha1
from urllib.parse import urljoin
from bs4 import BeautifulSoup as B

from PyMongoWrapper import F, Fn
from jindai.models import MediaItem, Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai import storage, parser


DEFAULT_IMG_PATTERNS = 'img[src]|[zoomfile]|[data-original]|[data-src]|[file]|[data-echo]'.replace(
    '|', '\n')


class CachedWebAccess:

    def __init__(self, base=''):
        if not base:
            base = tempfile.mktemp()
        if not os.path.exists(base):
            os.mkdir(base)
        self.base = base

    def _digest(self, url):
        return os.path.join(self.base, sha1(url.encode('utf-8')).hexdigest())
    
    def get(self, url):
        hashed = self._digest(url)
        if os.path.exists(hashed):
            with open(hashed, 'rb') as fi:
                return fi.read()
        else:
            data = storage.open(url, 'rb').read()
            with open(hashed, 'wb') as fo:
                fo.write(data)
            return data


class WebPageListingDataSource(DataSourceStage):
    """
    Import web page listings
    @zhs 从网页列表中导入语段
    """
    
    visited = set()
    queued = set()
    cache = CachedWebAccess('/tmp/wpdl')

    def apply_params(self, dataset='', content='', scopes='',
                     mongocollection='', lang='auto', detail_link='',
                     list_link='', proxy='', list_depth=1, tags='',
                     img_pattern='', level=1) -> None:
        """
        Args:
            dataset (DATASET):
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (LINES):
                Entry URLs
                @zhs 入口 URL
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
            scopes (LINES):
                URL Scopes
                @zhs 要抓取的 URL 范围
            mongocollection (str):
                Mongo Collection name
                @zhs 数据库集合名
        """
        self.proxies = {} if not proxy else {
            'http': proxy,
            'https': proxy
        }
        self.paths = PipelineStage.parse_paths(content)
        self.scopes = PipelineStage.parse_paths(scopes)
        self.list_depth = list_depth
        self.detail_link = re.compile(detail_link)
        self.list_link = re.compile(list_link)
        self.tags = PipelineStage.parse_lines(tags)
        self.dataset = dataset
        self.lang = lang
        self.image_patterns = PipelineStage.parse_lines(img_pattern) or DEFAULT_IMG_PATTERNS.split('\n')
        self.collection = Paragraph.get_coll(mongocollection)
        self.level = level

    def get_url(self, url):
        """Read html from url, return BeautifulSoup object"""
        self.log('get url', url)
        try:
            data = WebPageListingDataSource.cache.get(url)
        except OSError as ose:
            self.log_exception(f'Error while reading from {url}', ose) 
            data = ''
        return self.collection(source={'url':url}, html=data, dataset=self.dataset, lang=self.lang)

    def get_text(self, element):
        """Get text of element"""
        if element and element.text:
            return re.sub(r'\s+', ' ', element.text)
        return ''

    def parse_detail(self, url, para, b):
        """Parse url as a detail page"""
        para.pdate=datetime.datetime.utcnow()
        para.source={'url': url}
        para.dataset=self.dataset
        para.lang=self.lang
        para.content=self.get_text(b)
        para.keywords=self.tags
        para.title=self.get_text(b.find('title'))

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
                    self.collection.query(F.images == image.id).update(
                        Fn.pull(images=image.id))
                items.add(upath)

                para.images.append(image)

        self.log(f'Found {len(para.images)} images in {url}')
        return para

    def parse_list(self, url, b):
        """Parse url as a list page"""

        if not b:
            self.log(f'Cannot read list from {url}')
            return []

        links = set()
        for a in b.select('a[href]'):
            link_url = a['href'] = urljoin(url, a['href'])
            link_url = link_url.split('#')[0]
            # if visited
            if link_url in WebPageListingDataSource.visited:
                continue
            # match scopes
            for scope in self.scopes:
                if link_url.startswith(scope):
                    break
            else:
                continue
            # match link or detail patterns
            if self.list_link.search(link_url) or self.detail_link.search(link_url):
                links.add(link_url)

        self.log(len(links), 'links')
        return list(links)

    def fetch(self):
        level = self.level or 1
        for url in self.paths:
            if url in WebPageListingDataSource.visited:
                continue
            WebPageListingDataSource.visited.add(url)
            
            para = self.get_url(url)
            b = B(para.html, 'lxml')
            para.html = str(b)

            if level < self.list_depth and (self.list_link.search(url) or self.detail_link.search(url)):
                self.log('parse list', url)
                for upath in self.parse_list(url, b):
                    if upath not in WebPageListingDataSource.queued:
                        WebPageListingDataSource.queued.add(upath)
                        yield Paragraph(content=upath, level=level+1), self

            if self.detail_link.search(url):
                self.log('parse detail', url)
                yield self.parse_detail(url, para, b)
                

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


class ExtractHTMLParagraphs(PipelineStage):
    """
    Extract paragraphs from HTML
    @zhs 从 HTML 中提取段落
    """

    def __init__(self, field='html', assignments='', paragraph_selector=''):
        """
        Args:
            field (str): Field to read HTML
                @zhs 保存有 HTML 的字段名
            assignments (QUERY):
                Mapping element attribute to field, e.g. field=".css-selector//attribute"
                @zhs 字段与搜索字符串的关系，形如 field=".css-selector//attribute"
            paragraph_selector (str):
                CSS selector for paragraph
                @zhs 确定段落的 CSS 选择器，为空则整个网页作为一个段落
        """
        super().__init__()
        self.field = field
        self.paragraph_selector = paragraph_selector
        if isinstance(assignments, str):
            assignments = parser.parse(assignments)
        self.assignments = assignments or {}

    def _get_text(self, bs_ele):
        if bs_ele and bs_ele.text:
            return re.sub(r'\s+', ' ', bs_ele.text)
        return ''

    def _resolve_assignments(self, bs_ele, para: Paragraph):
        for field_name, field_path in self.assignments.items():
            if '//' in field_path:
                field_path, field_attr = field_path.rsplit('//', 1)
            else:
                field_attr = 'text'
            elements = bs_ele.select(field_path) if field_path else [bs_ele]
            value = []
            for element in elements:
                if field_attr == 'text':
                    value.append(self._get_text(element))
                elif field_attr == 'html':
                    value.append(str(element))
                elif field_attr in element.attrs:
                    value.append(str(element.attrs[field_attr]))
            if field_name == 'content':
                value = '\n'.join(value)
            elif value != 'keywords':
                value = ' '.join(value)
            setattr(para, field_name, value)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        html = paragraph[self.field] or ''
        b = B(html, 'lxml')
        self.log('load html data of length', len(html))

        for html_para in b.select(self.paragraph_selector) if self.paragraph_selector else [b]:
            para = type(paragraph)(
                lang=paragraph.lang,
                content='',
                source=paragraph.source,
                pagenum=1,
                dataset=paragraph.dataset,
                outline=paragraph.outline,
                keywords=[],
                html=str(html_para),
            )
            self._resolve_assignments(html_para, para)
            self.log('Extract para at', para.source['url'])
            yield para
