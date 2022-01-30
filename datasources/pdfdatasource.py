"""来自PDF
"""

import fitz
from pdf2image import convert_from_path
from rdflib import query
from models import Paragraph
from PyMongoWrapper import F, Fn, Var
from datasource import DataSource
from .utils import *


class PDFDataSource(DataSource):
    """从PDF中导入语段
    """

    def __init__(self, collection_name, lang, files_or_patterns, mongocollection=''):
        """
        Args:
            collection_name (str): 集合名称
            lang (自动:auto|简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|葡萄牙文:pt|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
            files_or_patterns (str): PDF文件列表
            mongocollection (str): 查询的数据集名
        """
        super().__init__()
        self.name = collection_name
        self.lang = lang
        self.files = expand_file_patterns(files_or_patterns.split('\n'))
        self.mongocollection = mongocollection

    def fetch(self):
        if self.mongocollection:
            class _TempClass(Paragraph):
                _collection = self.mongocollection
            para_coll = _TempClass
        else:
            para_coll = Paragraph
        
        for _, pdf in self.files:
            pdffile = pdf
            if pdf.startswith('sources/'):
                pdf = pdf[len('sources/'):]
            for a in para_coll.aggregator.match(F['source.file'] == pdf).group(_id=1, pages=Fn.max(Var['source.page'])).perform(raw=True):
                min_page = a['pages'] + 1
                break
            else:
                min_page = 0
            
            doc = fitz.open(pdffile)
            pages = doc.pageCount
            self.logger('importing from', pdf)

            if self.lang == 'auto':
                lang = lang_detect(os.path.basename(pdf).rsplit('.', 1)[0])
                if lang == 'zh-cn': lang = 'chs'
                elif lang.startswith('zh-'): lang = 'cht'
                else: lang = lang.split('-')[0]
            else:
                lang = self.lang
            
            para = ''
            for p in range(min_page, pages):
                label = doc[p].get_label()
                lines = doc[p].getText().split('\n')
                for para in merge_lines(lines, lang):
                    try:
                        yield para_coll(lang=lang, content=para.encode('utf-8', errors='ignore').decode('utf-8'), source={'file': pdf,
                            'page': p}, pagenum=label or (p+1), collection=self.name)
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
                