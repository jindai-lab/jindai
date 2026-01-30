"""汉字拆字处理"""

import os
import pickle

from jindai import PipelineStage, Plugin
from jindai.models import Paragraph


class HanziChaizi(PipelineStage):
    """汉字拆字，写入 chaizi 字段
    数据来源：https://github.com/howl-anderson/hanzi_chaizi/
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
    """汉字部首拆字插件"""

    def __init__(self, pmanager, **config) -> None:
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
