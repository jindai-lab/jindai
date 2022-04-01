from collections import defaultdict
from io import BytesIO

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from jindai import PipelineStage
from jindai.helpers import safe_import
from jindai.models import Paragraph
from .basics import Counter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


class KmeansClustering(PipelineStage):
    """使用 K-Means 方法聚类
    """

    def __init__(self, k, vector_field='vec'):
        """
        Args:
            k (int): 聚类的数量
            vector_field (str): 向量字段名
        """
        self.kmeans = KMeans(n_clusters=k)
        self.vector_field = vector_field
        self.paragraphs = []

    def resolve(self, p: Paragraph) -> Paragraph:
        self.paragraphs.append(p)
        return p

    def _get_vec(self, i):
        vec = getattr(self.paragraphs[i], self.vector_field)
        return vec

    def summarize(self, *args):
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
    """画出二维点图
    """

    def __init__(self, label_field='label', coordinates_field='coords', notation_field='content', notation_length=5):
        '''
        Args:
            label_field (str): 聚类标签字段名
            notation_field (str): 标记文本字段名
            notation_length (int): 标记文本最大长度
        '''
        self.label_field = label_field
        self.notation_field = notation_field
        self.notation_length = notation_length
        self.coordinates_field = coordinates_field
        self.fig, self.ax = plt.subplots()

    def _get_label_coords_notation(self, p: Paragraph):
        return getattr(p, self.label_field), getattr(p, self.coordinates_field), getattr(p, self.notation_field, '')[:self.notation_length]

    def summarize(self, returned):
        buf = BytesIO()
        clusters = defaultdict(list)

        for p in returned:
            label, coords, notation = self._get_label_coords_notation(p)
            clusters[label].append(coords)
            self.ax.annotate(xy=coords, text=notation)

        for label in clusters:
            mat = np.array(clusters[label])
            self.ax.scatter(mat[:, 0], mat[:, 1], label=f'{label}')

        self.fig.savefig(buf, format='png')

        return {
            '__file_ext__': 'png',
            'data': buf.getvalue()
        }


class GenerateCooccurance(PipelineStage):
    """生成共现矩阵
    """

    def __init__(self, weighted_by='vec_cos') -> None:
        """
        Args:
            weighted_by (vec_cos|token): 权重方式，vec_cos 表示*语段间*向量余弦相似度，token 表示语段内各词的共现次数

        Returns:
            共现矩阵，需要 GraphicClustering 绘图
        """
        safe_import('sentence_transformers')

        self.method = weighted_by
        self.counter = Counter()

    def token(self, p):
        for i, t in enumerate(p.tokens):
            for m in p.tokens[i+1:]:
                self.counter[f'{t}\n{m}'].inc()

    def resolve(self, p: Paragraph) -> Paragraph:
        if self.method == 'token':
            self.token(p)
        return p

    def summarize(self, r):
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

    def __init__(self, topk=1000) -> None:
        """
        Args:
            topk (int, optional): 最多绘制边的数量
        """
        self.topk = topk
        self.paragraphs = []

    def summarize(self, returned):
        import networkx as nx
        g = nx.Graph()
        for (a, b), v in sorted(returned, key=lambda x: x[1], reverse=True)[:self.topk]:
            for i in (a, b):
                if i not in g.nodes:
                    g.add_node(i)
            g.add_edge(a, b, weight=v)
            self.logger(a, b, v)

        meta = ''
        for i, c in enumerate(nx.connected_components(g)):
            meta += f'{i+1}\t' + '; '.join(c) + '\n'

        self.logger(meta)

        nx.draw(g, with_labels=True)
        buf = BytesIO()
        plt.savefig(buf, format='png')

        return {
            '__file_ext__': 'png',
            'data': buf.getvalue(),
            'meta': meta
        }
