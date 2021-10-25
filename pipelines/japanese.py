from models import Paragraph
from pipeline import PipelineStage


class JapaneseCut(PipelineStage):
    """日文分词并转写
    """

    def __init__(self):
        import pykakasi
        self.kks = pykakasi.kakasi()

    def resolve(self, p: Paragraph) -> Paragraph:
        p.tokens = []
        for i in self.kks.convert(p.content):
            p.tokens.append(i['orig'])
            p.tokens.append(i['hepburn'])
        return p
