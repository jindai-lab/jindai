from collections import defaultdict
from .basics import Counter
from pipeline import *
from models import Paragraph

import math

class TermFreq(PipelineStage):
    """词频统计
    """

    def __init__(self):
        self.tf = defaultdict(list)
        self.result = {}

    def resolve(self, p : Paragraph):
        for w in p.tokens:
            self.tf[w].append(p)

    def summarize(self, returned):
        final_words = sorted([{'word': k, 'count': len(v), 'paragraphs': v} for k, v in self.tf.items()], key=lambda x: x['count'], reverse=True)
        return final_words


class TFIDFWordFetch(PipelineStage):
    """基于TFIDF的词汇抽取，需要先分词或产生 tokens 字段
    """

    def __init__(self, min_df=1e-3):
        """初始化

        Args:
            min_df (float, optional): 最小的文档频率
        """
        self.min_df = min_df
        self.df = defaultdict(list)
        self.docs = Counter()
        self.tf = Counter()
        self.result = {}

    def resolve(self, p : Paragraph):
        self.docs[''].inc()
        for w in p.tokens:
            self.tf[w].inc()
        for w in set(p.tokens):
            self.df[w].append(p)
    
    def summarize(self, returned):
        self.tf = self.tf.as_dict()
        num_docs = self.docs.as_dict()['']
        min_df = self.min_df * num_docs
        tfidf = defaultdict(float)
        for w in self.df:
            tf, df = self.tf[w], len(self.df[w])
            if df < min_df: continue
            tfidf[w] = tf * math.log2(num_docs / df)

        final_words = sorted([{'word': k, 'tfidf': v, 'paragraphs': self.df[k][:50]} for k, v in tfidf.items()], key=lambda x: x['tfidf'], reverse=True)

        self.result['tfidf_words'] = final_words
        return final_words
