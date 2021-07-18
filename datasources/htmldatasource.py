"""来自网页或文本文件
"""
import codecs
import zipfile
from bs4 import BeautifulSoup as B
from models import Paragraph
from datasource import DataSource
from .utils import *


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
                lang=self.lang, content=b.text.strip(), pdffile=fname, pdfpage=0, pagenum=1,
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
                yield Paragraph(content=codecs.decode(l), pdffile=fn, collection=self.name, lang=self.lang, outline=f'{i+1:06d}')



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
