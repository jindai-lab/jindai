"""Import from PDF
@zhs 从 PDF 导入
"""

from io import BytesIO
import re
import fitz
import requests

from jindai import storage
from jindai.helpers import safe_import
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage


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
    @zhs 从 PDF 中导入语段
    """

    def apply_params(self, dataset_name='', lang='auto', content='', mongocollection='',
                     skip_existed=True, page_range='', nougat_endpoint='', rapid_ocr=False):
        """
        Args:
            dataset_name (DATASET):
                Dataset name
                @zhs 数据集名称
            lang (LANG):
                Language
                @zhs 语言标识
            content (FILES):
                Paths
                @zhs PDF文件列表
            mongocollection (str):
                MongoDB collection name
                @zhs 数据库集合名
            skip_existed (bool):
                Skip existed pages and files
                @zhs 直接跳过已存在于数据集中的文件
            page_range (str):
                Page range, e.g. 1-3
                @zhs 页码范围，例如 1-3
        """
        self.name = dataset_name
        self.lang = lang
        self.mongocollection = mongocollection
        self.skip_existed = skip_existed
        self.page_range = sorted(resolve_range(page_range))
        self.files = PipelineStage.parse_paths(content)
    
    def fetch(self):
        para_coll = Paragraph.get_coll(self.mongocollection)
        lang = self.lang
        
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
            imported_pages = 0
            short_path = storage.truncate_path(filepath)
            self.log('importing', short_path)
            
            stream = storage.open(filepath, 'rb')
            if hasattr(stream, 'name'):
                doc = fitz.open(stream.name)
            else:
                doc = fitz.open('pdf', stream)
            
            page_range = self.page_range
            if not page_range:
                min_page = existent.get(short_path)
                min_page = 0 if min_page is None else (min_page + 1)
                self.log('... from page', min_page)
                page_range = range(min_page, doc.page_count)

            for page in page_range:
                if page >= doc.page_count:
                    break

                try:
                    label = doc[page].get_label()
                except (RuntimeError, TypeError):
                    label = ''

                try:
                    content = doc[page].get_text().encode(
                        'utf-8', errors='ignore').decode('utf-8')
                except Exception as ex:
                    self.log(filepath, page+1, ex)
                    content = ''

                if len(content) > 10:
                    imported_pages += 1
                
                yield para_coll(
                    lang=lang, content=content,
                    source={'file': short_path, 'page': page},
                    pagenum=label or (page+1),
                    dataset=self.name
                )

            if not existent.get(short_path) and imported_pages == 0:
                self.log(filepath, 'no sufficient texts found.')
