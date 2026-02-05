"""
Sentiment Analysis
@zhs 情感分析
"""

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.helpers import safe_import
from jindai.models import Paragraph


class AutoSentimentAnalysis(PipelineStage):
    """
    Automatic Sentiment Analysis
    @zhs 根据语种标识自动进行情感分析
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        snow = safe_import('snownlp').SnowNLP
        paragraph.extdata['sentiment'] = (snow(paragraph.content).sentiments-0.5) * 2
        return paragraph


class SentimentAnalysisPlugin(Plugin):

    def __init__(self, pmanager, **config) -> None:
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
