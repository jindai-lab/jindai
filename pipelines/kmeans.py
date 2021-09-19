import numpy as np
from io import BytesIO
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pipeline import PipelineStage
from models import Paragraph

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
    
    def resolve(self, p : Paragraph) -> Paragraph:
        self.paragraphs.append(p)
        return p

    def _get_vec(self, i):
        vec = getattr(self.paragraphs[i], self.vector_field)
        return vec

    def summarize(self):
        if len(self.paragraphs) == 0: return
        mat = np.zeros((len(self.paragraphs), len(self._get_vec(0))), 'float')
        pca = PCA(n_components=2)
        Xn = PCA.fit_transform(mat)
        for i in range(len(self.paragraphs)):
            mat[i, :] = self._get_vec(i)
        self.kmeans.fit(mat)
        for p, l, coords in zip(self.paragraphs, self.kmeans.labels_, Xn):
            p.label = l
            p.coords = coords


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
        self.pl = plt.plot()

    def _get_label_coords_notation(self, p : Paragraph):
        return getattr(p, self.label_field), getattr(p, self.coordinates_field), getattr(p, self.notation_field, '')[:self.notation_length]

    def resolve(self, p : Paragraph) -> Paragraph:
        label, coords, notation = self._get_label_coords_notation(p)
        self.pl.scatter(coords, label=label)
        self.pl.annotate(coords, notation)
        return p

    def summarize(self):
        buf = BytesIO()
        self.pl.savefig(buf, format='png')
        return {
            '__file_ext__': 'png',
            'data': buf.getvalue()
        }
