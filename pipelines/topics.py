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
    注意：对于中文文本，进行的是字符级别的向量化，因此无需分词。对其他语种，需要事先进行分词。
    """

    _lang_dicts = {}

    def __init__(self, ch_classical=False):
        '''
        Args:
            ch_classical (bool): 中文内容为文言文
        '''
        self.ch_classical = ch_classical

    @staticmethod
    def _load_vec(datafile):
        datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../vectors_data', datafile)
        d = {}
        with open(datafile) as f:
            for l in f:
                r = l.split()
                d[r[0]] = np.array([float(_) for _ in r[1:]])
        return d

    @staticmethod
    def _get_lang_config(lang, ch_classical):
        if lang in Word2Vec._lang_dicts:
            return Word2Vec._lang_dicts[lang]
        d = {}
        if lang == 'chs':
            d = Word2Vec._load_vec('ch_classical_chars300.txt') if ch_classical else Word2Vec._load_vec('ch_modern_chars300.txt')
        elif lang in ('en', 'de', 'fr', 'ru', 'ja'):
            d = Word2Vec._load_vec(lang + '-vec.txt')
        Word2Vec._lang_dicts[lang] = d
        return d

    def resolve(self, p : Paragraph) -> Paragraph:
        arr = list(filter(lambda w: w is not None, 
                        [Word2Vec._get_lang_config(p.lang, self.ch_classical).get(t) for t in (p.content if p.lang.startswith('ch') else p.tokens)]
                    ))
        if len(arr) > 0:
            p.vec = np.average(arr, axis=0)
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
