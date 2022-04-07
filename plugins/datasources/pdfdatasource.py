"""来自PDF
"""

import fitz
from jindai import expand_patterns, truncate_path
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage
from PyMongoWrapper import F, Fn, Var


class PDFDataSource(DataSourceStage):
    """从PDF中导入语段
    """

    class Implementation(DataSourceStage.Implementation):
        
        def __init__(self, dataset_name, lang, content, mongocollection=''):
            """
            Args:
                dataset_name (DATASET): 数据集名称
                lang (LANG): 语言标识
                content (str): PDF文件列表
                mongocollection (str): 数据库集合名
                skip_existed (bool): 直接跳过已存在于数据集中的文件
            """
            super().__init__()
            self.name = dataset_name
            self.lang = lang
            self.files = expand_patterns(content)
            self.mongocollection = mongocollection

        def fetch(self):
            para_coll = Paragraph.get_coll(self.mongocollection)
            existent = {
                a['_id']: a['pages']
                for a in para_coll.aggregator.match(F.dataset == self.name).group(_id=Var['source.file'], pages=Fn.max(Var['source.page']))
            }
            
            for pdf in self.files:
                path = truncate_path(pdf)
                min_page = existent.get(path)
                min_page = 0 if min_page is None else (min_page + 1)
                
                doc = fitz.open(pdf)
                pages = doc.pageCount
                self.logger('importing', pdf, 'as', path, 'from page', min_page)

                lang = self.lang
                
                for p in range(min_page, pages):
                    label = doc[p].get_label()
                    lines = doc[p].getText()
                    try:
                        yield para_coll(
                            lang=lang, content=lines.encode('utf-8', errors='ignore').decode('utf-8'),
                            source={'file': path, 'page': p}, pagenum=label or (p+1), dataset=self.name)
                    except Exception as e:
                        self.logger(pdf, p+1, e)
