"""来自网页或文本文件
"""
import codecs
from bs4 import BeautifulSoup as B
from models import Paragraph
from datasource import DataSource
from .utils import *
from urllib.parse import urljoin


class HTMLDataSource(DataSource):
    """从HTML网页中读取语段，每个文件计作一个语段
    """

    def __init__(self, collection_name, lang, files):
        """
        Args:
            collection_name (str): 集合名称
            lang (chs|cht|en|de|fr|ru|es|ja|kr|vn): 语言标识
            files (str): HTML或包含HTML的ZIP压缩包文件列表
        """
        self.name = collection_name
        self.lang = lang
        self.files = files.split('\n')

    def fetch(self):
        def import_html_src(fname, html, outline=''):
            b = B(html, 'lxml')
            p = Paragraph(
                lang=self.lang, content=b.text.strip(), source={'file': fname}, pagenum=1,
                collection=self.name, outline=outline
            )
            p.content = str(b.find('body'))
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
            lang (chs|cht|en|de|fr|ru|es|ja|kr|vn): 语言标识
            files (str): HTML或包含HTML的ZIP压缩包文件列表
        """
        self.name = collection_name
        self.lang = lang
        self.files = files.split('\n')

    def fetch(self):
        for fp, fn in expand_file_patterns(self.files):
            for i, l in enumerate(fp):
                yield Paragraph(content=codecs.decode(l), source={'file': fn}, collection=self.name, lang=self.lang, outline=f'{i+1:06d}')



class LinesDataSource(DataSource):
    """从直接输入的文本中获得语段，每行一个语段
    """

    def __init__(self, collection_name, lang, lines):
        """
        Args:
            collection_name (str): 集合名称
            lang (chs|cht|en|de|fr|ru|es|ja|kr|vn): 语言标识
            lines (str): 一行一个语段
        """
        self.name = collection_name
        self.lang = lang
        self.lines = lines.split('\n')

    def fetch(self):
        return self.lines


class HtmlImageDataSource(DataSource):
    """从网页中获得图像
    """

    def __init__(self, urls_or_patterns : str, iterate : str, collection : str):
        """
        Args:
            urls_or_patterns (str): 网址列表
            iterate (str): 形如 start-end 格式的范围，用以匹配网址模式 * 的范围
            collection (str): 数据集名称
        """
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
                    yield Paragraph(url=url, collection=self.collection, content=title, keywords=title.split(), image_source=imgurl, source={'url': url})
                    imgset.add(imgurl)
    