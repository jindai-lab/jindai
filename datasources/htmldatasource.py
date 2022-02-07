"""来自网页或文本文件
"""
import codecs
from bs4 import BeautifulSoup as B
from models import Paragraph, try_download, parser
from datasource import DataSource
from .utils import *
from urllib.parse import urljoin


class HTMLDataSource(DataSource):
    """从HTML网页中读取语段，每个网页计作一个语段
    """

    def __init__(self, collection_name, lang, files, fields='content="//text"', paragraph_selector=''):
        """
        Args:
            collection_name (COLLECTION): 集合名称
            lang(LANG): 语言标识
            files (str): HTML或包含HTML的ZIP压缩包文件列表
            paragraph_selector (str): 确定段落的 CSS 选择器，为空则整个网页作为一个段落
            fields (str): 字段与搜索字符串的关系，形如 field=".css-selector//attribute"
        """
        super().__init__()
        self.name = collection_name
        self.lang = lang
        self.files = files.split('\n')
        self.fields = parser.eval(fields)
        self.paragraph_selector = paragraph_selector

    def fetch(self):
        def import_html_src(fname, html, outline=''):
            b = B(html, 'lxml')

            for para in b.select(self.paragraph_selector) if self.paragraph_selector else [b]:
                p = Paragraph(
                    lang=self.lang, content='', source={'url' if '://' in fn else 'file': fname}, pagenum=1,
                    collection=self.name, outline=outline,
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

        for fp, fn in expand_file_patterns(self.files):
            self.logger('reading from', fn)
            ol = ''
            if '#' in fn: fn, ol = fn.split('#', 1)
            yield from import_html_src(fn, fp, ol)
            

class TextDataSource(DataSource):
    """从文本文件中读取语段
    """

    def __init__(self, collection_name, lang, files):
        """
        Args:
            collection_name (COLLECTION): 集合名称
            lang(LANG): 语言标识
            files (str): HTML或包含HTML的ZIP压缩包文件列表
        """
        super().__init__()
        self.name = collection_name
        self.lang = lang
        self.files = files.split('\n')

    def fetch(self):
        for fp, fn in expand_file_patterns(self.files):
            for i, l in enumerate(fp):
                yield Paragraph(content=codecs.decode(l), source={'url' if '://' in fn else 'file': fn}, collection=self.name, lang=self.lang, outline=f'{i+1:06d}')



class LinesDataSource(DataSource):
    """从直接输入的文本中获得语段，每行一个语段
    """

    def __init__(self, collection_name, lang, lines):
        """
        Args:
            collection_name (COLLECTION): 集合名称
            lang(LANG): 语言标识
            lines (str): 一行一个语段
        """
        super().__init__()
        self.name = collection_name
        self.lang = lang
        self.lines = lines.split('\n')

    def fetch(self):
        return map(lambda x: Paragraph(content=x, lang=self.lang), self.lines)


class HTMLImageDataSource(DataSource):
    """从网页中获得图像
    """

    def __init__(self, files : str, collection : str):
        """
        Args:
            files (str): 网址列表
            collection (COLLECTION): 数据集名称
        """
        super().__init__()
        self.collection = collection
        self.files = files.split('\n')

    def fetch(self):
        imgset = set()
        for buf, url in expand_file_patterns(self.files):
            self.logger('fetching from', url)
            html = buf.read()
            
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
                    yield Paragraph(source={'url': url}, collection=self.collection, content=title, keywords=title.split())
                    imgset.add(imgurl)
    