"""Topics, clustering and classification
@chs 话题与分类
"""

from collections import defaultdict
from io import BytesIO

import numpy as np
from jindai import PipelineStage
from jindai.helpers import safe_import
from jindai.models import Paragraph
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .basics import AccumulateParagraphs, Counter

many_stop_words = safe_import('many_stop_words')


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


class FilterStopWords(PipelineStage):
    """Filter stop words
    @chs 过滤停用词
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
        paragraph.tokens = [
            _ for _ in paragraph.tokens if _ not in self.stopwords and _ not in FilterStopWords._lang_stopwords.get(paragraph.lang, [])]
        return paragraph


class LDA(PipelineStage):
    """Topic model based on LDA
    @chs 基于 LDA 的话题模型
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

    def resolve(self, paragraph: Paragraph) -> Paragraph:
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
            self.result[_id] = int(
                np.argmax(lda[list(zip(range(len(self.dict)), vec))]))
        return {
            'labels': self.result,
            'topics': lda.show_topics()
        }


class Word2Vec(PipelineStage):
    """Vectorize text, multilingual support
    @chs 根据不同语言自动进行 Word2Vec 向量化
    @chs （调用 transformers 的 paraphrase-multilingual-MiniLM-L12-v2 模型）
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        '''
        Args:
            model_name (str): Model name
                @chs 模型名称，默认为多语言小模型
        '''
        text2vec = safe_import('text2vec')
        self.bert = text2vec.SBert(model_name)
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.vec = self.bert.encode(paragraph.content)
        return paragraph


class WordsBagVec(PipelineStage):
    """Vectorize with one-shot words bag
    @chs 使用词袋模型进行 0/1 编码的向量化
    """

    def __init__(self, dims=100000) -> None:
        """
        Args:
            dims (int, optional): Dimensions
                @chs 维数
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
    """Classification with cosine similarity
    @chs 基于余弦相似度的小样本分类
    """

    def __init__(self, label_field, auto_update=False):
        '''
        Args:
            label_field (str): Field for labels
                @chs 标签字段
            auto_update (bool): Update exemplars
                @chs 将先前识别结果用于之后的样本
        '''
        super().__init__()
        self.label_field = label_field
        self.auto_update = auto_update
        self.vecs = {}
        self.vecs_cnt = defaultdict(int)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
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
    """Clustering with cosine similarity
    @chs 余弦相似度聚类"""

    def __init__(self, min_community_size=10, threshold=0.75, label_field='label'):
        '''
        Args:
            min_community_size (int): Min size for clustering community
                @chs 最少的聚类数量
            threshold (float): Threshold for similarity
                @chs 相似度阈值
            label_field (str): Label field name
                @chs 生成的标签字段名
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
        clusters = util.community_detection(
            vecs, min_community_size=self.min_community_size, threshold=self.threshold)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                setattr(self.paragraphs[idx], self.label_field, i+1)
        return self.paragraphs


class KmeansClustering(PipelineStage):
    """Clustering with K-Means
    @chs 使用 K-Means 方法聚类
    """

    def __init__(self, k, vector_field='vec'):
        """
        Args:
            k (int): Clusters
                @chs 聚类的数量
            vector_field (str): Vector field
                @chs 向量字段名
        """
        self.kmeans = KMeans(n_clusters=k)
        self.vector_field = vector_field
        self.paragraphs = []
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        self.paragraphs.append(paragraph)
        return paragraph

    def _get_vec(self, i):
        vec = getattr(self.paragraphs[i], self.vector_field)
        return vec

    def summarize(self, _):
        if len(self.paragraphs) == 0:
            return []
        mat = np.zeros((len(self.paragraphs), len(self._get_vec(0))), 'float')
        pca = PCA(n_components=2)
        for i in range(len(self.paragraphs)):
            mat[i, :] = self._get_vec(i)
        Xn = pca.fit_transform(mat)
        self.kmeans.fit(mat)
        for p, l, coords in zip(self.paragraphs, self.kmeans.labels_, Xn):
            p.label = l
            p.coords = coords
        return self.paragraphs


class DrawClusters(PipelineStage):
    """Draw clusters
    @chs 画出二维点图
    """

    def __init__(self, label_field='label', coordinates_field='coords', notation_field='content', notation_length=5):
        '''
        Args:
            label_field (str): Label field
                @chs 聚类标签字段名
            notation_field (str): Notation field
                @chs 标记文本字段名
            notation_length (int): Max notation text length
                @chs 标记文本最大长度
        '''
        self.label_field = label_field
        self.notation_field = notation_field
        self.notation_length = notation_length
        self.coordinates_field = coordinates_field
        self.fig, self.axis = plt.subplots()
        super().__init__()

    def _get_label_coords_notation(self, paragraph: Paragraph):
        return getattr(paragraph, self.label_field), getattr(paragraph, self.coordinates_field), getattr(paragraph, self.notation_field, '')[:self.notation_length]

    def summarize(self, result):
        buf = BytesIO()
        clusters = defaultdict(list)

        for p in result:
            label, coords, notation = self._get_label_coords_notation(p)
            clusters[label].append(coords)
            self.axis.annotate(xy=coords, text=notation)

        for label in clusters:
            mat = np.array(clusters[label])
            self.axis.scatter(mat[:, 0], mat[:, 1], label=f'{label}')

        self.fig.savefig(buf, format='png')

        return {
            '__file_ext__': 'png',
            'data': buf.getvalue()
        }


class GenerateCooccurance(PipelineStage):
    """Generate cooccurance matrix
    @chs 生成共现矩阵
    """

    def __init__(self, weighted_by='vec_cos') -> None:
        """
        Args:
            weighted_by (vec_cos|token): Weight by
            @chs 权重方式，vec_cos 表示*语段间*向量余弦相似度，token 表示语段内各词的共现次数
        """
        super().__init__()
        safe_import('sentence_transformers')
        self.method = weighted_by
        self.counter = Counter()

    def token(self, p):
        """Count from tokens"""
        for i, t in enumerate(p.tokens):
            for m in p.tokens[i+1:]:
                self.counter[f'{t}\n{m}'].inc()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if self.method == 'token':
            self.token(paragraph)
        return paragraph

    def summarize(self, _):
        if self.method == 'vec_cos':
            from sentence_transformers.util import cos_sim
            c = {}
            for i, p in enumerate(self.paragraphs):
                for q in self.paragraphs[i+1:]:
                    c[f'{p.content[:10]}\n{q.content[:10]}'] = float(
                        cos_sim(p.vec, q.vec)[0][0])
        else:
            c = self.counter.as_dict()
        for k, v in c.items():
            if '\n' in k:
                yield k.split('\n', 1), v


class GraphicClustering(PipelineStage):
    """Clustering by undirectional graph
    @chs 按无向图进行聚类"""

    def __init__(self, topk=1000) -> None:
        """
        Args:
            topk (int, optional): Max edges count
                @chs 最多绘制边的数量
        """
        self.topk = topk
        self.paragraphs = []
        super().__init__()

    def summarize(self, result):
        nx = safe_import('networkx')
        g = nx.Graph()
        for (a, b), v in sorted(result, key=lambda x: x[1], reverse=True)[:self.topk]:
            for i in (a, b):
                if i not in g.nodes:
                    g.add_node(i)
            g.add_edge(a, b, weight=v)
            self.logger(a, b, v)

        meta = ''
        for i, cluster in enumerate(nx.connected_components(g)):
            meta += f'{i+1}\t' + '; '.join(cluster) + '\n'

        self.logger(meta)

        nx.draw(g, with_labels=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')

        return {
            '__file_ext__': 'png',
            'data': buf.getvalue(),
            'meta': meta
        }
