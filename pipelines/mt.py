"""机器翻译
"""
from models import Paragraph
from pipeline import PipelineStage
from opencc import OpenCC


class MachineTranslation(PipelineStage):
    """机器翻译"""

    def __init__(self, to_lang='chs', model='opus-mt') -> None:
        """
        Args:
            to_lang(LANG): 目标语言标识
            model (较快速度:opus-mt|较高准确度:mbart50_m2m): 机器翻译所使用的模型
        """
        super().__init__()
        
        from easynmt import EasyNMT
        self.model = EasyNMT(model)

        self.cc = None
        if to_lang == 'chs':
            to_lang = 'zh'
        elif to_lang == 'cht':
            to_lang = 'zh'
            self.cc = OpenCC('s2t')

        self.to_lang = to_lang        

    def resolve(self, p: Paragraph) -> Paragraph:
        t = self.model.translate(p.content, source_lang=p.lang if p.lang not in ('chs', 'cht') else 'zh', target_lang=self.to_lang)
        if self.cc:
            t = self.cc.convert(t)
        p.content = t
        if p.id:
            p._id = None
            del p._orig['_id']
        return p
    