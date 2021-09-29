"""话题与分类
"""

from pipeline import PipelineStage
from models import Paragraph
from collections import defaultdict
import numpy as np
import many_stop_words
import os
from .basics import AccumulateParagraphs


class FilterStopWords(PipelineStage):
    """过滤停用词
    """

    _lang_stopwords = {
        'en': many_stop_words.get_stop_words('en'),
        'fr': many_stop_words.get_stop_words('fr'),
        'de': many_stop_words.get_stop_words('de'),
        'ru': many_stop_words.get_stop_words('ru'),
        'ja': many_stop_words.get_stop_words('ja')
    }

    def __init__(self, stopwords=''):
        """
        Args:
            stopwords (str): 额外的停用词表，用空格分割
        """
        self.stopwords = set(stopwords.split())
        
    def resolve(self, p):
        p.tokens = [_ for _ in p.tokens if _ not in self.stopwords and _ not in FilterStopWords._lang_stopwords.get(p.lang, [])]
        return p


class LDA(PipelineStage):
    """基于 LDA 的话题模型
    """

    def __init__(self, num_topics):
        """
        Args:
            num_topics (int): 话题数
        """
        self.num_topics = num_topics
        self.dict = {}
        self.mat = defaultdict(list)
        super().__init__()

    def resolve(self, p : Paragraph) -> Paragraph:
        self.mat[str(p.id)] = p.tokens

    def summarize(self, *args):
        from gensim.models.ldamodel import LdaModel

        for s in self.mat.values():
            for w in s:
                if w not in self.dict:
                    self.dict[w] = len(self.dict)

        self.array = np.zeros((len(self.mat), len(self.dict)), dtype=int)
        
        for _i, s in enumerate(self.mat.values()):
            for w in s:
                self.array[_i, self.dict[w]] += 1
               
        id2word = {v: k for k, v in self.dict.items()}
        lda = LdaModel(corpus=[list(zip(range(len(self.dict)), a)) for a in self.array],
                id2word=id2word, num_topics=self.num_topics, passes=1)
        self.result = {}
        for _id, vec in zip(self.mat, self.array):
            self.result[_id] = int(np.argmax(lda[list(zip(range(len(self.dict)), vec))]))
        return {
            'labels': self.result,
            'topics': lda.show_topics()
        }


class Word2Vec(PipelineStage):
    """根据不同语言自动进行 Word2Vec 向量化
    （调用 transformers 的 paraphrase-multilingual-MiniLM-L12-v2 模型）
    """
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        '''
        Args:
            model_name (str): 模型名称，默认为多语言小模型
        '''
        import text2vec
        self.bert = text2vec.SBert(model_name)

    def resolve(self, p : Paragraph) -> Paragraph:
        p.vec = self.bert.encode(p.content)
        return p
    
    
class WordsBagVec(PipelineStage):
    """使用词袋模型进行 0/1 编码的向量化
    """
    def __init__(self, dims=100000) -> None:
        """
        Args:
            dims (int, optional): 维数
        """
        self.words = {}
        self.dims = dims
    
    def resolve(self, p: Paragraph) -> Paragraph:
        p.vec = np.zeros(self.dims, float)
        for t in p.tokens:
            if t not in self.words:
                if len(self.words) < self.dims:
                    self.words[t] = len(self.words)
                    p.vec[self.words[t]] = 1
            else:
                p.vec[self.words[t]] += 1
        return p
    

class CosSimFSClassifier(AccumulateParagraphs):
    """基于余弦相似度的小样本分类
    """

    def __init__(self, label_field, auto_update=False):
        '''
        Args:
            label_field (str): 标签字段
            auto_update (bool): 将先前识别结果用于之后的样本
        '''
        super().__init__()
        self.label_field = label_field
        self.auto_update = auto_update
        self.vecs = {}
        self.vecs_cnt = defaultdict(int)

    def resolve(self, p : Paragraph) -> Paragraph:
        super().resolve(p)
        label = getattr(p, self.label_field, '')
        if label:
            if label in self.vecs:
                self.vecs[label] += p.vec
            else:
                self.vecs[label] = p.vec
            self.vecs_cnt[label] += 1
        return p

    def _infer(self, vec):
        vec = np.array(vec)
        vec = vec / np.linalg.norm(vec)
        r, s = '', 0
        for l, v in self.vecs.items():
            ls = np.dot(v, vec)
            if ls > s:
                s = ls
                r = l
        l = r
        if self.auto_update:
            self.vecs[l] = self.vecs[l] * self.vecs_cnt[l] + vec
            self.vecs[l] = self.vecs[l] / np.linalg.norm(self.vecs[l])
            self.vecs_cnt[l] += 1
        return l

    def summarize(self, *args): 
        for l in self.vecs:
            self.vecs[l] = self.vecs[l] / np.linalg.norm(self.vecs[l])
        for p in self.paragraphs:
            if not hasattr(p, self.label_field):
                setattr(p, self.label_field, self._infer(p.vec))
        return self.paragraphs


class CosSimClustering(AccumulateParagraphs):

    def __init__(self, min_community_size=10, threshold=0.75, label_field='label'):
        '''
        Args:
            min_community_size (int): 最少的聚类数量
            threshold (float): 相似度阈值
            label_field (str): 生成的标签字段名
        '''
        self.min_community_size = 10
        self.threshold = threshold
        self.vecs = []
        self.label_field = label_field

    def resolve(self, p):
        super().resolve(p)
        self.vecs.append(p.vec)
        return p

    def summarize(self, *args):
        from sentence_transformers import util
        vecs = np.array(self.vecs)
        clusters = util.community_detection(vecs, min_community_size=self.min_community_size, threshold=self.threshold)
        for i, c in enumerate(clusters):
            for idx in c:
                setattr(self.paragraphs[idx], self.label_field, i+1)
        return self.paragraphs
