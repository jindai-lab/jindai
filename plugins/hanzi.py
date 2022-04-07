"""汉字拆字处理"""

import pickle

from jindai import PipelineStage, Plugin, safe_open
from jindai.models import Paragraph


class HanziChaizi(PipelineStage):
    """汉字拆字，写入 chaizi 字段
    数据来源：https://github.com/howl-anderson/hanzi_chaizi/
    """

    def __init__(self) -> None:
        with safe_open('models_data/chaizi.pkl', 'rb') as fi:
            self.dict = pickle.load(fi)
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.chaizi = []
        for c in paragraph.content:
            c = self.dict.get(c)
            if c:
                paragraph.chaizi.append(c)
        return paragraph
    

class ChaiziPlugin(Plugin):
    """汉字部首拆字插件"""

    def __init__(self, app, **config):
        super().__init__(app, **config)
        self.register_pipelines(globals())
    