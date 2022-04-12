"""情感分析
"""

from jindai import PipelineStage, Plugin
from jindai.helpers import safe_import
from jindai.models import Paragraph


class AutoSentimentAnalysis(PipelineStage):
    """根据语种标识自动进行情感分析
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        snow = safe_import('snownlp').SnowNLP
        paragraph.sentiment = (snow(paragraph.content).sentiments-0.5) * 2
        return paragraph


class SentimentAnalysisPlugin(Plugin):
    """情感分析插件"""

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
