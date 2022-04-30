"""Import from PDF
@chs 来自PDF
"""

import fitz
from PyMongoWrapper import F, Fn, Var

from jindai import expand_patterns, truncate_path
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage


class PDFDataSource(DataSourceStage):
    """
    Import paragraphs from PDF
    @chs 从PDF中导入语段
    """

    class Implementation(DataSourceStage.Implementation):
        """datasource implementation"""

        def __init__(self, dataset_name, lang, content, mongocollection='', skip_existed=True):
            """
            Args:
                dataset_name (DATASET):
                    Dataset name
                    @chs 数据集名称
                lang (LANG):
                    Language
                    @chs 语言标识
                content (str):
                    Paths
                    @chs PDF文件列表
                mongocollection (str):
                    MongoDB collection name
                    @chs 数据库集合名
                skip_existed (bool):
                    Skip existed pages and files
                    @chs 直接跳过已存在于数据集中的文件
            """
            super().__init__()
            self.name = dataset_name
            self.lang = lang
            self.files = expand_patterns(content)
            self.mongocollection = mongocollection
            self.skip_existed = skip_existed

        def fetch(self):
            para_coll = Paragraph.get_coll(self.mongocollection)
            if self.skip_existed:
                existent = {
                    a['_id']: a['pages']
                    for a in para_coll.aggregator.match(
                        F.dataset == self.name
                    ).group(
                        _id=Var['source.file'], pages=Fn.max(
                            Var['source.page'])
                    )
                }
            else:
                existent = {}

            for pdf in self.files:
                path = truncate_path(pdf)
                min_page = existent.get(path)
                min_page = 0 if min_page is None else (min_page + 1)

                doc = fitz.open(pdf)
                pages = doc.pageCount
                self.logger('importing', pdf, 'as',
                            path, 'from page', min_page)

                lang = self.lang

                for page in range(min_page, pages):
                    label = doc[page].get_label()
                    lines = doc[page].getText()
                    try:
                        yield para_coll(
                            lang=lang, content=lines.encode(
                                'utf-8', errors='ignore').decode('utf-8'),
                            source={'file': path, 'page': page},
                            pagenum=label or (page+1),
                            dataset=self.name
                        )
                    except Exception as ex:
                        self.logger(pdf, page+1, ex)
