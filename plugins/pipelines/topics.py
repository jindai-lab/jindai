"""话题与分类
"""

from collections import defaultdict
import numpy as np

from jindai import PipelineStage
from jindai.helpers import safe_import
from jindai.models import Paragraph

from .basics import AccumulateParagraphs

many_stop_words = safe_import('many_stop_words')


class FilterStopWords(PipelineStage):
    """过滤停用词
    """

    _lang_stopwords = {
        l: many_stop_words.get_stop_words(l) for l in ['en', 'fr', 'de', 'ru', 'ja']
    }

    def __init__(self, stopwords=''):
        """
        Args:
            stopwords (str): 额外的停用词表，用空格分割
        """
        self.stopwords = set(stopwords.split())
        super().__init__()
        
    def resolve(self, paragraph):
        paragraph.tokens = [_ for _ in paragraph.tokens if _ not in self.stopwords and _ not in FilterStopWords._lang_stopwords.get(paragraph.lang, [])]
        return paragraph


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
        self.array = None
        self.result = {}
        self.mat = defaultdict(list)
        super().__init__()

    def resolve(self, paragraph : Paragraph) -> Paragraph:
        self.mat[str(paragraph.id)] = paragraph.tokens

    def summarize(self, *args):
        model = safe_import('gensim.models_ldamodel').LdaModel

        for sent_vec in self.mat.values():
            for word_vec in sent_vec:
                if word_vec not in self.dict:
                    self.dict[word_vec] = len(self.dict)

        self.array = np.zeros((len(self.mat), len(self.dict)), dtype=int)
        
        for _i, sent_vec in enumerate(self.mat.values()):
            for word_vec in sent_vec:
                self.array[_i, self.dict[word_vec]] += 1
               
        id2word = {v: k for k, v in self.dict.items()}
        lda = model(corpus=[list(zip(range(len(self.dict)), a)) for a in self.array],
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
        text2vec = safe_import('text2vec')
        self.bert = text2vec.SBert(model_name)
        super().__init__()

    def resolve(self, paragraph : Paragraph) -> Paragraph:
        paragraph.vec = self.bert.encode(paragraph.content)
        return paragraph
    
    
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
        super().__init__()
    
    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.vec = np.zeros(self.dims, float)
        for t in paragraph.tokens:
            if t not in self.words:
                if len(self.words) < self.dims:
                    self.words[t] = len(self.words)
                    paragraph.vec[self.words[t]] = 1
            else:
                paragraph.vec[self.words[t]] += 1
        return paragraph
    

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

    def resolve(self, paragraph : Paragraph) -> Paragraph:
        super().resolve(paragraph)
        label = getattr(paragraph, self.label_field, '')
        if label:
            if label in self.vecs:
                self.vecs[label] += paragraph.vec
            else:
                self.vecs[label] = paragraph.vec
            self.vecs_cnt[label] += 1
        return paragraph

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
        for dim in self.vecs:
            self.vecs[dim] = self.vecs[dim] / np.linalg.norm(self.vecs[dim])
        for para in self.paragraphs:
            if not hasattr(para, self.label_field):
                setattr(para, self.label_field, self._infer(para.vec))
        return self.paragraphs


class CosSimClustering(AccumulateParagraphs):
    """余弦相似度聚类"""

    def __init__(self, min_community_size=10, threshold=0.75, label_field='label'):
        '''
        Args:
            min_community_size (int): 最少的聚类数量
            threshold (float): 相似度阈值
            label_field (str): 生成的标签字段名
        '''
        super().__init__()
        self.min_community_size = min_community_size
        self.threshold = threshold
        self.vecs = []
        self.label_field = label_field
        safe_import('sentence_transformers')


    def resolve(self, paragraph):
        super().resolve(paragraph)
        self.vecs.append(paragraph.vec)
        return paragraph

    def summarize(self, *args):
        util = safe_import('sentence_transformers.util')
        vecs = np.array(self.vecs)
        clusters = util.community_detection(vecs, min_community_size=self.min_community_size, threshold=self.threshold)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                setattr(self.paragraphs[idx], self.label_field, i+1)
        return self.paragraphs
