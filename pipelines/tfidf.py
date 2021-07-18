from collections import defaultdict
from .basics import Counter
from pipeline import *
from models import Paragraph

import math

class TermFreq(PipelineStage):
    """词频统计
    """

    def __init__(self):
        self.tf = Counter()
        self.result = {}

    def resolve(self, p : Paragraph):
        for w in p.tokens:
            self.tf[w].inc()

    def summarize(self, returned):
        final_words = list(sorted(self.tf.as_dict().items(), key=lambda x: x[1], reverse=True))
        return final_words


class TFIDFWordFetch(PipelineStage):
    """基于TFIDF的词汇抽取，需要先分词或产生 tokens 字段
    """

    def __init__(self, min_df=1e-3):
        """初始化

        Args:
            min_df (float, optional): 最小的文档频率. 
        """
        self.min_df = min_df
        self.df = Counter()
        self.tf = Counter()
        self.result = {}

    def resolve(self, p : Paragraph):
        self.df[''].inc()
        for w in p.tokens:
            self.tf[w].inc()
        for w in set(p.tokens):
            self.df[w].inc()
    
    def summarize(self, returned):
        self.df, self.tf = self.df.as_dict(), self.tf.as_dict()
        num_docs = self.df['']
        min_df = self.min_df * num_docs
        tfidf = defaultdict(float)
        for w in self.df:
            if not w: continue
            tf, df = self.tf[w], self.df[w]
            if df < min_df: continue
            tfidf[w] = tf * math.log2(num_docs / df)

        final_words = list(sorted(tfidf.items(), key=lambda x: x[1], reverse=True))

        self.result['tfidf_words'] = final_words
        return final_words
