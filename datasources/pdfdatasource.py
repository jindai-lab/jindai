"""来自PDF
"""

import fitz
from pdf2image import convert_from_path
import re, logging
import glob
from models import Paragraph
from PyMongoWrapper import F, Fn, Var
from datasource import DataSource
from .utils import *


class PDFDataSource(DataSource):
    """从PDF中导入语段
    """

    def __init__(self, collection_name, lang, files_or_patterns):
        """
        Args:
            collection_name (str): 集合名称
            lang (chs|cht|en|de|fr|ru|es|ja|kr|vn): 语言标识
            files_or_patterns (str): PDF文件列表
        """

        self.name = collection_name
        self.lang = lang
        self.files = expand_file_patterns(files_or_patterns.split('\n'))

    def fetch(self):

        def __startswith(heap, needles):
            for n in needles:
                if heap.startswith(n):
                    return True
            return False

        for _, pdffile in self.files:
            pdffile_ = pdffile
            if pdffile.startswith('sources/'):
                pdffile = pdffile[len('sources/'):]
            for a in Paragraph.aggregator.match(F.pdffile == pdffile).group(_id=1, pages=Fn.max(Var.pdfpage)).perform(raw=True):
                min_page = a['pages'] + 1
                break
            else:
                min_page = 0
            
            doc = fitz.open(pdffile_)
            pages = doc.pageCount
            
            para = ''
            for p in range(min_page, pages):
                lines = doc[p].getText().split('\n')
                for para in merge_lines(lines, self.lang):
                    try:
                        yield Paragraph(lang=self.lang, content=para.encode('utf-8', errors='ignore').decode('utf-8'), pdffile=pdffile,
                            pdfpage=p, pagenum=p+1, collection=self.name)
                    except Exception as e:
                        print(pdffile, p+1, e)


class PDFImageDataSource(DataSource):
    """从PDF中获得图像
    """

    def __init__(self, files_or_patterns, limit=None):
        """
        Args:
            files_or_patterns (str): PDF文件列表
            limit (int): 最多返回的图像数量
        """
        self.files = expand_file_patterns(files_or_patterns.split('\n'))
        self.limit = limit

    def fetch(self):
        for _, pdffile in self.files:
            logging.info('processing', pdffile)
            images = convert_from_path(pdffile, 300, first_page=0, last_page=self.limit)
            for page, i in enumerate(images):
                yield Paragraph(image=i, pdffile=pdffile, pdfpage=page)
                