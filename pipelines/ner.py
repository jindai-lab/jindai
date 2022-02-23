"""命名实体识别
"""
from helpers import safe_import
from pipeline import PipelineStage
from models import Paragraph
import re

hanlp = safe_import('hanlp')


class HanLPModel(PipelineStage):
    """使用 HanLP 的预训练模型进行处理
    """

    def __init__(self, pretrained, result=''):
        """
        Args:
            result (tok|ner|srl|sdp/dm|sdp/pas|sdp/psd|con|lem|pos|fea|dep): 要提取的结果
            pretrained (CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH|NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA|UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE): 预训练模型选择
        """
        import hanlp
        self.result = result
        self.model = hanlp.load(getattr(hanlp.pretrained.mtl, pretrained))

    def resolve(self, p : Paragraph) -> Paragraph:
        setattr(p, self.result, self.model(p.content, tasks=self.result)[self.result])
        return p


class HanLPNER_ZH(PipelineStage):
    """使用 HanLP 的预训练模型进行现代汉语命名实体识别（NER）
    """

    def __init__(self) -> None:
        """ 提取的结果格式为 [名称, 类型]
        """
        import hanlp
        self.model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

    def resolve(self, p: Paragraph) -> Paragraph:
        p.ner = self.model(p.content, tasks='ner').get('ner/msra', [])
        return p


class HanLPNER_JA(PipelineStage):
    """使用 HanLP 的预训练模型进行日语命名实体识别（NER）
    """

    def __init__(self) -> None:        
        import hanlp
        self.model = hanlp.load(hanlp.pretrained.mtl.hanlp.pretrained.mtl.NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA)

    def resolve(self, p: Paragraph) -> Paragraph:
        p.ner = self.model(re.sub(r'\s', '', p.content), tasks='ner').get('ner', [])
        return p


class SpaCyNER(PipelineStage):
    """使用 SpaCy 的预训练模型进行中英德法日俄命名实体识别（NER）
    """

    def __init__(self, lang) -> None:
        """ 提取的结果格式为 [名称, 类型]
        Args:
            lang (LANG): 语言
        """
        safe_import('spacy')
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
