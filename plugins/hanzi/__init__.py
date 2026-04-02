"""Chinese character decomposition processing"""

import os
import pickle

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.models import Paragraph


class HanziChaizi(PipelineStage):
    """Chinese character decomposition, write to chaizi field
    Data source: https://github.com/howl-anderson/hanzi_chaizi/
    """

    def __init__(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'chaizi.pkl'), 'rb') as fi:
            self.dict = pickle.load(fi)
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.extdata['chaizi'] = []
        for c in paragraph.content:
            c = self.dict.get(c)
            if c:
                paragraph.extdata['chaizi'].append(c)
        return paragraph


class ChaiziPlugin(Plugin):
    """Chinese character radical decomposition plugin"""

    def __init__(self, pmanager, **config) -> None:
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
