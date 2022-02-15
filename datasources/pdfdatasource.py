"""来自PDF
"""

from genericpath import exists
import fitz
from pdf2image import convert_from_path
from models import Paragraph
from PyMongoWrapper import F, Fn, Var
from datasource import DataSource
from .utils import *


class PDFDataSource(DataSource):
    """从PDF中导入语段
    """

    def __init__(self, dataset_name, lang, files_or_patterns, mongocollection=''):
        """
        Args:
            dataset_name (DATASET): 数据集名称
            lang (LANG): 语言标识
            files_or_patterns (str): PDF文件列表
            mongocollection (str): 查询的数据集名
            skip_existed (bool): 直接跳过已存在于数据集中的文件
        """
        super().__init__()
        self.name = dataset_name
        self.lang = lang
        self.files = expand_file_patterns(files_or_patterns.split('\n'))
        self.mongocollection = mongocollection

    def fetch(self):
        para_coll = get_dbo(self.mongocollection)
        existent = {
            a['_id']: a['pages']
            for a in para_coll.aggregator.match(F.dataset == self.name).group(_id=Var['source.file'], pages=Fn.max(Var['source.page']))
        }
        
        for _, pdf in self.files:
            pdffile = pdf
            if pdf.startswith('sources/'):
                pdf = pdf[len('sources/'):]
            
            min_page = existent.get(pdf)
            min_page = 0 if min_page is None else (min_page + 1)
            
            doc = fitz.open(pdffile)
            pages = doc.pageCount
            self.logger('importing', pdf, 'from page', min_page)

            lang = self.lang
            
            for p in range(min_page, pages):
                label = doc[p].get_label()
                lines = doc[p].getText()
                try:
                    yield para_coll(lang=lang, content=lines.encode('utf-8', errors='ignore').decode('utf-8'), source={'file': pdf,
                        'page': p}, pagenum=label or (p+1), dataset=self.name)
                except Exception as e:
                    self.logger(pdffile, p+1, e)


class PDFImageDataSource(DataSource):
    """从PDF中获得图像
    """

    def __init__(self, files_or_patterns, limit=None):
        """
        Args:
            files_or_patterns (str): PDF文件列表
            limit (int): 最多返回的图像数量
        """
        super().__init__()
        self.files = expand_file_patterns(files_or_patterns.split('\n'))
        self.limit = limit

    def fetch(self):
        for _, pdffile in self.files:
            self.logger('processing', pdffile)
            images = convert_from_path(pdffile, 300, first_page=0, last_page=self.limit)
            for page, i in enumerate(images):
                yield Paragraph(image=i, source={'file': pdffile, 'page': page})
                