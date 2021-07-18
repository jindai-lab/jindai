"""基于 HanLP 的处理
"""
from pipeline import PipelineStage
from models import Paragraph
import re


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
