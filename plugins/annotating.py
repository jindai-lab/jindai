"""
Annotations
@chs 标注
"""
import re
import fitz

from PyMongoWrapper import F, ObjectId
from jindai import Plugin, storage
from jindai.pipeline import DataSourceStage
from jindai.models import Paragraph
from jindai.helpers import logined, rest


class Annotation(Paragraph):
    """Annotation for a Paragraph object"""

    paragraph_id = ObjectId
    paragraph_collection = str

    @classmethod
    def on_initialize(cls):
        """Ensure index on paragraph_id"""
        super().on_initialize()
        cls.ensure_index('paragraph_id')

    @property
    def paragraph(self):
        """Get linked paragraph object"""
        return Paragraph.get_coll(self.paragraph_collection).first(F.id == self.paragraph_id)

    @staticmethod
    def query_by_paragraph(para: Paragraph):
        """Get annotations by paragraph"""
        return Annotation.query(F.paragraph_id == para.id,
                                F.paragraph_collection == (para.mongocollection or ''))


def remove_spaces(text):
    """Adjust spacing in annotations"""
    text = re.sub(r'[\s\n]+', ' ', text)
    text = re.sub(
        r'([\u4e00-\u9fa5，]{1})\s+([\u4e00-\u9fa5，]{1})', '\1\2', text)
    return text


def parse_annotation_in_page(ann, page):
    """Get underlined or highlighted text"""
    if not ann.vertices:
        return []
    texts = []
    for i in range(0, len(ann.vertices), 4):
        vertices = ann.vertices[i:i+4]
        quad = (
            min(map(lambda c: c[0], vertices)),
            min(map(lambda c: c[1], vertices)),
            max(map(lambda c: c[0], vertices)),
            max(map(lambda c: c[1], vertices)),
        )
        # print(quad)
        texts.append(' '.join([w[4] for w in sorted(page.get_text_words(
        ), key=lambda c:(c[1]//10, c[0])) if fitz.Rect(w[:4]).intersects(quad)]))
    return texts


def extract_annotation(filename):
    """Extract annotations from PDF"""
    doc = fitz.Document(filename)
    tpath = storage.truncate_path(filename)
    for pnum, page in enumerate(doc):
        label = page.get_label() or (pnum+1)
        annots = list(page.annots())
        if annots:
            yield label, {'file': tpath, 'page': pnum}
        for ann in annots:
            if ann.type[1] in ('Highlight', 'Underline'):
                texts = parse_annotation_in_page(ann, page)
                yield remove_spaces('\n'.join(texts)), {'file': tpath, 'page': pnum}
            yield ann.get_text(), {'file': tpath, 'page': pnum}


class AnnotationsFromPDF(DataSourceStage):
    """
    Import annotations from PDF
    @chs 导入 PDF 中的批注
    """

    class Implementation(DataSourceStage.Implementation):
        """Implementing annotation fetching"""

        def __init__(self, content, mongocollection=''):
            """
            Args:
                content (str):
                    Import from files
                    @chs 要导入的 PDF
                mongocollection (str):
                    Mongo Collection name to lookup
                    @chs 对应的语段所在数据库
            """
            super().__init__()
            self.files = storage.expand_patterns(content)
            self.mongocollection = mongocollection

        def fetch(self):
            for file in self.files:
                for annotate, para_source in extract_annotation(file):
                    para = Paragraph.get_coll(self.mongocollection).first(F.source == para_source) \
                        or Paragraph()
                    yield Annotation(
                        paragraph_id=para.id, paragraph_collection=self.mongocollection,
                        content=annotate, lang='auto',
                        source=para_source)


Paragraph.annotations = property(Annotation.query_by_paragraph)


class AnnotatingPlugin(Plugin):
    """Plugin for annotating"""

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
        app = self.pmanager.app

        @app.route('/api/plugins/annotations')
        @rest(mapping={'id': 'para_id'})
        def annoatations(para_id, collection):
            return Annotation.query(F.user == logined(), F.paragraph_id == id, F.paragraph_colletion == collection)
