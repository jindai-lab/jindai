"""汉字处理"""

from models import Paragraph
from pipeline import PipelineStage
import pickle


class HanziChaizi(PipelineStage):
    """汉字拆字，写入 chaizi 字段
    数据来源：https://github.com/howl-anderson/hanzi_chaizi/
    """

    def __init__(self) -> None:
        with open('models_data/chaizi.pkl', 'rb') as fi:
            self.dict = pickle.load(fi)

    def resolve(self, p: Paragraph) -> Paragraph:
        p.chaizi = []
        for c in p.content:
            c = self.dict.get(c)
            if c:
                p.chaizi.append(c)
        return p
    