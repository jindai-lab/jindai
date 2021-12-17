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

    def __init__(self, collection_name, lang, files, fields='content="//text"'):
        """
        Args:
            collection_name (str): 集合名称
            lang (简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
            files (str): HTML或包含HTML的ZIP压缩包文件列表
            fields (str): 字段与搜索字符串的关系，形如 field=".css-selector//attribute"
        """
        super().__init__()
        self.name = collection_name
        self.lang = lang
        self.files = files.split('\n')
        self.fields = parser.eval(fields)

    def fetch(self):
        def import_html_src(fname, html, outline=''):
            b = B(html, 'lxml')
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
                els = [b]
                if field_path: els = b.select(field_path)
                value = ''
                for el in els:
                    if field_attr == 'text':
                        value += el.text + '\n'
                    elif field_attr == 'html':
                        value += str(el) + '\n'
                    elif field_attr in el.attrs:
                        value += el.attrs[field_attr] + '\n'
                setattr(p, field_name, value)

            del b
            return p

        for fp, fn in expand_file_patterns(self.files):
            ol = ''
            if '#' in fn: fn, ol = fn.split('#', 1)
            yield import_html_src(fn, fp, ol)
            

class TextDataSource(DataSource):
    """从文本文件中读取语段
    """

    def __init__(self, collection_name, lang, files):
        """
        Args:
            collection_name (str): 集合名称
            lang (简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
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
            collection_name (str): 集合名称
            lang (简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
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

    def __init__(self, urls_or_patterns : str, iterate : str, collection : str):
        """
        Args:
            urls_or_patterns (str): 网址列表
            iterate (str): 形如 start-end 格式的范围，用以匹配网址模式 * 的范围
            collection (str): 数据集名称
        """
        super().__init__()
        self.collection = collection
        self.urls = []
        for l in urls_or_patterns.split('\n'):
            if '*' in l and iterate:
                start,end=map(int,iterate.split('-'))
                for i in range(start , end+1):
                    self.urls.append(l.replace('*', str(i)))

    def fetch(self):
        imgset = set()
        for url in self.urls:
            html = try_download(url)
            
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
    