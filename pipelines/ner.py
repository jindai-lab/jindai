"""命名实体识别
"""
from pipeline import PipelineStage
from models import Paragraph
from .hanlpmul import HanLPNER_JA, HanLPNER_ZH


class SpaCyNER(PipelineStage):
    """使用 SpaCy 的预训练模型进行中英德法日俄命名实体识别（NER）
    """

    def __init__(self, lang) -> None:
        """ 提取的结果格式为 [名称, 类型]
        Args:
            lang(LANG): 语言
        """
        import spacy
        model = {
            'en': 'en_core_web_sm',
            'chs': 'zh_core_web_sm',
            'de': 'de_core_news_sm',
            'ja': 'ja_core_news_sm',
            'fr': 'fr_core_news_sm',
            'ru': 'ru_core_news_sm',
        }.get(lang, 'en_core_web_sm')
        self.model = spacy.load(model)

    def resolve(self, p: Paragraph) -> Paragraph:
        p.ner = [(e.text, {'ORG': 'ORGANIZATION'}.get(e.label_, e.label_)) for e in self.model(p.content).ents]
        return p


class AutoNER(PipelineStage):
    """根据输入语段自动进行对应语言的命名实体识别
    """

    def __init__(self, ifchs='hanlp', ifja='hanlp') -> None:
        """
        Args:
            ifchs (hanlp|spacy): 中文命名实体识别的实现
            ifja (hanlp|spacy): 日文命名实体识别的实现
        """
        self.models_meta = {
            'chs': HanLPNER_ZH if ifchs == 'hanlp' else lambda: SpaCyNER('chs'),
            'ja': HanLPNER_JA if ifja == 'hanlp' else lambda: SpaCyNER('ja'),
            'de': lambda: SpaCyNER('de'),
            'fr': lambda: SpaCyNER('fr'),
            'ru': lambda: SpaCyNER('ru'),
            'en': lambda: SpaCyNER('en')
        }
        self.models = {}

    def resolve(self, p: Paragraph) -> Paragraph:
        if p.lang not in self.models and p.lang in self.models_meta:
            self.models[p.lang] = self.models_meta[p.lang]()
        return self.models[p.lang].resolve(p)


class NERAsTokens(PipelineStage):
    """将命名实体识别的结果作为词袋
    """

    def resolve(self, p: Paragraph) -> Paragraph:
        p.tokens = [_[0] + '/' + _[1] for _ in p.ner] if hasattr(p, 'ner') else []
        return p
