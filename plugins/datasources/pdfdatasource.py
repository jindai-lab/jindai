"""Import from PDF
@chs 从 PDF 导入
"""

import re
import fitz
from PyMongoWrapper import F, Fn, Var

from jindai import storage
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage


def resolve_range(page_range: str):
    """Resolve page range string

    :param page_range: page range, e.g. 1-3; 1,3,5-6,23
    :type page_range: str
    :return: range
    :rtype: Iterable, or None when error
    """
    ranges = (page_range or '').split(',')
    for rng in ranges:
        if '-' in rng:
            try:
                start, end = map(int, rng.split('-', 1))
                yield from range(start-1, end)
            except ValueError:
                pass
        elif rng and re.match(r'\d+', rng):
            yield int(rng)-1


class PDFDataSource(DataSourceStage):
    """
    Import paragraphs from PDF
    @chs 从PDF中导入语段
    """
    
    def apply_params(self, dataset_name='', lang='auto', content='', mongocollection='', skip_existed=True, page_range=''):
        """
        Args:
            dataset_name (DATASET):
                Dataset name
                @chs 数据集名称
            lang (LANG):
                Language
                @chs 语言标识
            content (LINES):
                Paths
                @chs PDF文件列表
            mongocollection (str):
                MongoDB collection name
                @chs 数据库集合名
            skip_existed (bool):
                Skip existed pages and files
                @chs 直接跳过已存在于数据集中的文件
            page_range (str):
                Page range, e.g. 1-3
                @chs 页码范围，例如 1-3
        """
        self.name = dataset_name
        self.lang = lang
        self.files = storage.globs(content)
        self.mongocollection = mongocollection
        self.skip_existed = skip_existed
        self.page_range = sorted(resolve_range(page_range))

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

        for filepath in self.files:
            short_path = storage.truncate_path(filepath)
            stream = storage.open(filepath, 'rb')
            if hasattr(stream, 'name'):
                doc = fitz.open(stream.name)
            else:
                doc = fitz.open('pdf', stream)
            self.logger('importing', short_path)
            page_range = self.page_range
            if not page_range:
                min_page = existent.get(short_path)
                min_page = 0 if min_page is None else (min_page + 1)
                self.logger('... from page', min_page)
                page_range = range(min_page, doc.page_count)

            lang = self.lang

            for page in page_range:
                if page >= doc.page_count:
                    break

                try:
                    label = doc[page].get_label()
                except RuntimeError:
                    label = ''
                try:
                    lines = doc[page].get_text()
                    yield para_coll(
                        lang=lang, content=lines.encode(
                            'utf-8', errors='ignore').decode('utf-8'),
                        source={'file': short_path, 'page': page},
                        pagenum=label or (page+1),
                        dataset=self.name
                    )
                except Exception as ex:
                    self.logger(filepath, page+1, ex)
