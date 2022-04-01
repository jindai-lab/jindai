"""情感分析
"""

from jindai import PipelineStage
from jindai.helpers import safe_import
from jindai.models import Paragraph


class AutoSentimentAnalysis(PipelineStage):
    """根据语种标识自动进行情感分析
    """

    def resolve(self, p: Paragraph) -> Paragraph:
        safe_import('snownlp')
        from snownlp import SnowNLP
        p.sentiment = (SnowNLP(p.content).sentiments-0.5) * 2
        return p
