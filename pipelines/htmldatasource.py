"""来自网页或文本文件
"""
import codecs
from concurrent.futures import ThreadPoolExecutor
from sys import implementation
from urllib.parse import urljoin
from bs4 import BeautifulSoup as B
import re

from models import ImageItem, Paragraph, parser
from pipeline import DataSourceStage
from storage import expand_path, expand_patterns, truncate_path, safe_open


class HTMLDataSource(DataSourceStage):
    """从HTML网页中读取语段，每个网页计作一个语段
    """
    class _Implementation(DataSourceStage._Implementation):

        def __init__(self, dataset_name, lang, content, fields='content="//text"', paragraph_selector=''):
            """
            Args:
                dataset_name (DATASET): 数据集名称
                lang (LANG): 语言标识
                content (str): HTML或包含HTML的ZIP压缩包文件列表
                paragraph_selector (str): 确定段落的 CSS 选择器，为空则整个网页作为一个段落
                fields (str): 字段与搜索字符串的关系，形如 field=".css-selector//attribute"
            """
            super().__init__()
            self.name = dataset_name
            self.lang = lang
            self.files = content.split('\n')
            self.fields = parser.eval(fields)
            self.paragraph_selector = paragraph_selector

        def fetch(self):
            def import_html_src(fname, html, outline=''):
                b = B(html, 'lxml')

                for para in b.select(self.paragraph_selector) if self.paragraph_selector else [b]:
                    p = Paragraph(
                        lang=self.lang, content='', source={'url' if '://' in fn else 'file': truncate_path(fn)}, pagenum=1,
                        dataset=self.name, outline=outline,
                        keywords=[]
                    )

                    for field_name, field_path in self.fields.items():
                        if '//' in field_path:
                            field_path, field_attr = field_path.rsplit('//', 1)
                        else:
                            field_attr = 'text'
                        els = para.select(field_path) if field_path else [para]
                        value = []
                        for el in els:
                            if field_attr == 'text':
                                value.append(el.text)
                            elif field_attr == 'html':
                                value.append(str(el))
                            elif field_attr in el.attrs:
                                value.append(el.attrs[field_attr])
                        setattr(p, field_name, value)

                    yield p

                del b

            for fn in expand_patterns(self.files):
                self.logger('reading from', fn)
                ol = ''
                if '#' in fn:
                    fpath, ol = fn.split('#', 1)
                yield from import_html_src(fpath, safe_open(fn), ol)


class TextDataSource(DataSourceStage):
    """从文本文件中读取语段
    """
    class _Implementation(DataSourceStage._Implementation):

        def __init__(self, dataset_name, lang, content):
            """
            Args:
                dataset_name (DATASET): 数据集名称
                lang (LANG): 语言标识
                content (str): HTML或包含HTML的ZIP压缩包文件列表
            """
            super().__init__()
            self.name = dataset_name
            self.lang = lang
            self.content = content.split('\n')

        def fetch(self):
            for fn in expand_patterns(self.content):
                for i, l in enumerate(safe_open(fn)):
                    yield Paragraph(content=codecs.decode(l), 
                            source={'url' if '://' in fn else 'file': truncate_path(fn)}, 
                            dataset=self.name, lang=self.lang, outline=f'{i+1:06d}')


class LinesDataSource(DataSourceStage):
    """从直接输入的文本中获得语段，每行一个语段
    """
    class _Implementation(DataSourceStage._Implementation):

        def __init__(self, dataset_name, lang="auto", content=""):
            """
            Args:
                dataset_name (DATASET): 数据集名称
                lang (LANG): 语言标识
                content (str): 一行一个语段
            """
            super().__init__()
            self.name = dataset_name
            self.lang = lang
            self.lines = content.split('\n')

        def fetch(self):
            return map(lambda x: Paragraph(content=x, lang=self.lang, dataset=self.name), self.lines)


class WebPageListingDataSource(DataSourceStage):
    """从网页列表中导入语段"""
    class _Implementation(DataSourceStage._Implementation):

        def __init__(self, dataset, patterns,
                     mongocollection='', lang='auto', detail_link='', list_link='', proxy='', list_depth=1, tags='',
                     img_pattern='img[src]|[zoomfile]|[data-original]|[data-src]|[file]|[data-echo]'.replace('|', '\n')) -> None:
            """
            Args:
                patterns (str): 列表页面模式
                list_depth (int): 列表深度
                proxy (str): 代理服务器
                tags (str): 标签，一行一个
                detail_link (str): 详情页面正则表达式
                list_link (str): 列表页面正则表达式
                dataset (str): 数据集名称
                lang (str): 语言标记
                img_pattern (str): 图像检索标记，一行一个
                mongocollection (str): 数据库集合名
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
            self.image_patterns = img_pattern.split('\n')
            self.mongocollection = mongocollection
            self.convert = Paragraph.get_converter(mongocollection)

        def read_html(self, url):
            html = safe_open(url, proxies=self.proxies).read()
            if not html:
                self.logger('Cannot read from', url)
                return

            b = B(html.decode('utf-8'), 'lxml')
            return b

        def get_text(self, el):
            if el and el.text:
                return re.sub(r'\s+', ' ', el.text)
            return ''

        def parse_detail(self, url):
            b = self.read_html(url)
            return
            
            p = Paragraph(source={'url': url},
                          dataset=self.dataset, lang=self.lang)
            p.content = self.get_text(b)
            p.keywords = self.tags
            p.title = self.get_text(b.find('title'))

            for imgp in self.image_patterns:
                for i in b.select(imgp):
                    attr_m = re.search(r'\[(.+)\]', imgp)
                    if attr_m:
                        attr = i[attr_m.group(1)]
                    else:
                        attr = self.get_text(i)
                    u = urljoin(url, attr)
                    if ImageItem.first({'source.url': urljoin(url, attr)}):
                        continue
                    p.images.append(ImageItem(source={'url': u}))

            self.logger(f'Found {len(p.images)} images in {url}')
            return p

        def parse_list(self, url):

            b = self.read_html(url)
            if not b:
                self.logger(f'Cannot read list from {url}')
                return []

            links = {
                urljoin(url, a['href'])
                for a in b.select('a[href]')
            }

            return [u for u in links if self.list_link.search(u) or self.detail_link.search(u)]

        def fetch(self):

            queue = [(p, 1) for p in expand_patterns(self.patterns)]
            visited = set()

            def _do_parse(q_tup):
                url, level = q_tup
                if url in visited or Paragraph.get_coll(self.mongocollection).first({'source.url': url}):
                    return

                visited.add(url)

                if level <= self.list_depth:
                    for u in self.parse_list(url):
                        yield u, level

                if level == self.list_depth or self.detail_link.search(url):
                    r = self.parse_detail(url)
                    if r:
                        yield r, level

            while queue:
                self.logger(f'Queuing {len(queue)} urls')
                tpe = ThreadPoolExecutor(5)
                for riter in tpe.map(_do_parse, queue[:5]):
                    for res, level in riter:
                        if isinstance(res, Paragraph):
                            yield self.convert(res)
                        else:
                            queue.append((res, level))
                queue = queue[5:]


class BiblioDataSource(DataSourceStage):
    """从 endnote 文献条目产生语段
    """
    class _Implementation(DataSourceStage._Implementation):

        def __init__(self, content, dataset, lang='chs', format='endnote') -> None:
            """
            Args:
                content (str): 文件名或通配符，一行一个
                lang (LANG): 语言标识
                dataset (DATASET): 集合名称
                format (endnote|bibtex): 文献条目信息格式
            """
            super().__init__()
            if not hasattr(self, format):
                raise NotImplemented()

            self.method = getattr(self, format)
            self.files = expand_patterns(content)
            self.dataset = dataset
            self.lang = lang

        def endnote(self, lines):
            d = {
                'content': '',
                'authors': []
            }
            field = ''
            for l in lines:
                if not l.strip():
                    if d:
                        yield Paragraph(dataset=self.dataset, lang=self.lang, **d)
                    d = {
                        'content': '',
                        'authors': []
                    }
                else:
                    l = l.decode('utf-8').strip()
                    if ' ' not in l:
                        value = l
                    else:
                        field, value = l.split(' ', 1)
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
                        }.get(field.upper(), f'content')

                    if ';' in value and field != 'content':
                        value = [_ for _ in value.split(';') if _]
                    if field in d:
                        if field == 'content':
                            d[field] += value
                        else:
                            if not isinstance(d[field], list):
                                d[field] = [d[field]]
                            if isinstance(value, list):
                                d[field] += value
                            else:
                                d[field].append(value)
                    else:
                        d[field] = value
            if d:
                yield Paragraph(dataset=self.dataset, **d)

        def fetch(self):
            for f, _ in self.files:
                yield from self.method(f)
