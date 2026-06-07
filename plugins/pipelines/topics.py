"""Topics, clustering and classification
@zhs 话题与分类
"""

from collections import defaultdict
from io import BytesIO
from typing import Dict

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity

from jindai.models import Paragraph, TextEmbeddings
from jindai.pipeline import PipelineStage

from .basics import AccumulateParagraphs, Counter


def community_detection(vecs, min_community_size=1, threshold=0.75):
    """
    基于余弦相似度阈值的社区检测。
    
    参数:
        vecs: list of array-like 或 numpy 数组，形状 (n_samples, n_features)
        min_community_size: int，最小社区大小，小于此值的社区将被丢弃（标签置为 -1）
        threshold: float，相似度阈值，大于等于此值的两个向量属于同一社区
    
    返回:
        labels: numpy 数组，形状 (n_samples,)，社区标签（-1 表示不属于任何有效社区）
    """
    # 转换为 numpy 数组并确保是 float 类型
    vecs = np.asarray(vecs, dtype=np.float32)
    n = len(vecs)
    
    if n == 0:
        return np.array([], dtype=int)
    
    # 计算余弦相似度矩阵（仅保留上三角以节省内存，但 full 矩阵更方便）
    sim_matrix = cosine_similarity(vecs)
    
    # 构建邻接矩阵：相似度 >= threshold 的节点之间有一条边
    adjacency = (sim_matrix >= threshold).astype(int)
    # 确保对角线为 0（节点不与自己连接，不影响连通分量）
    np.fill_diagonal(adjacency, 0)
    
    # 计算连通分量（无向图）
    n_components, labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)
    
    # 统计每个社区的大小
    unique, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique, counts))
    
    # 将大小小于 min_community_size 的社区标记为 -1
    filtered_labels = np.array([
        lab if size_map[lab] >= min_community_size else -1
        for lab in labels
    ])
    
    return filtered_labels


def import_plt():
    """safe import matplotlib.pyplot"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    return plt


def import_sklearn_kmeans_pca() -> tuple:
    """safe import sklearn kmeans and pca, in respective order
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    return KMeans, PCA


class LDA(PipelineStage):
    """Topic model based on LDA
    @zhs 基于 LDA 的话题模型
    """

    def __init__(self, num_topics) -> None:
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

    async def summarize(self, *_) -> dict[str, dict]:
        import gensim.models.ldamodel
        model = gensim.models.ldamodel.LdaModel

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


class TextEmbedding(PipelineStage):
    """Vectorize text, multilingual support
    @zhs 根据不同语言自动进行 TextEmbedding 向量化
    """

    def __init__(self) -> None:
        pass
        
    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.extdata['vec'] = TextEmbeddings.get_embedding_sync(paragraph.content)
        return paragraph


class WordsBagVectorize(PipelineStage):
    """Vectorize with one-shot words bag
    @zhs 使用词袋模型进行 0/1 编码的向量化
    """

    def __init__(self, dims=100000) -> None:
        """
        Args:
            dims (int, optional): Dimensions
                @zhs 维数
        """
        self.words = {}
        self.dims = dims
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.vec = np.zeros(self.dims, float)
        for token in paragraph.tokens:
            if token not in self.words:
                if len(self.words) < self.dims:
                    self.words[token] = len(self.words)
                    paragraph.vec[self.words[token]] = 1
            else:
                paragraph.vec[self.words[token]] += 1
        return paragraph


class CosSimFSClassifier(AccumulateParagraphs):
    """Classification with cosine similarity
    @zhs 基于余弦相似度的小样本分类
    """

    def __init__(self, label_field, auto_update=False) -> None:
        '''
        Args:
            label_field (str): Field for labels
                @zhs 标签字段
            auto_update (bool): Update exemplars
                @zhs 将先前识别结果用于之后的样本
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

    def _infer(self, vec) -> str:
        vec = np.array(vec)
        vec = vec / np.linalg.norm(vec)
        result, sim = '', 0
        for label, vec_ in self.vecs.items():
            label_sim = np.dot(vec_, vec)
            if label_sim > sim:
                sim = label_sim
                result = label
        label = result
        if self.auto_update:
            self.vecs[label] = self.vecs[label] * self.vecs_cnt[label] + vec
            self.vecs[label] = self.vecs[label] / \
                np.linalg.norm(self.vecs[label])
            self.vecs_cnt[label] += 1
        return label

    async def summarize(self, *_):
        for dim in self.vecs:
            self.vecs[dim] = self.vecs[dim] / np.linalg.norm(self.vecs[dim])
        for para in self.paragraphs:
            if not hasattr(para, self.label_field):
                setattr(para, self.label_field, self._infer(para.vec))
        return self.paragraphs


class CosSimClustering(AccumulateParagraphs):
    """Clustering with cosine similarity
    @zhs 余弦相似度聚类"""

    def __init__(self, min_community_size=10, threshold=0.75, label_field='label') -> None:
        '''
        Args:
            min_community_size (int): Min size for clustering community
                @zhs 最少的聚类数量
            threshold (float): Threshold for similarity
                @zhs 相似度阈值
            label_field (str): Label field name
                @zhs 生成的标签字段名
        '''
        super().__init__()
        self.min_community_size = min_community_size
        self.threshold = threshold
        self.vecs = []
        self.label_field = label_field

    def resolve(self, paragraph):
        super().resolve(paragraph)
        self.vecs.append(paragraph.vec)
        return paragraph

    async def summarize(self, *_):
        vecs = np.array(self.vecs)
        clusters = community_detection(
            vecs, min_community_size=self.min_community_size, threshold=self.threshold)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                setattr(self.paragraphs[idx], self.label_field, i+1)
        return self.paragraphs


class KmeansClustering(PipelineStage):
    """Clustering with K-Means
    @zhs 使用 K-Means 方法聚类
    """

    def __init__(self, k, vector_field='vec') -> None:
        """
        Args:
            k (int): Clusters
                @zhs 聚类的数量
            vector_field (str): Vector field
                @zhs 向量字段名
        """
        KMeans, _ = import_sklearn_kmeans_pca()
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

    async def summarize(self, _) -> list:
        if len(self.paragraphs) == 0:
            return []
        mat = np.zeros((len(self.paragraphs), len(self._get_vec(0))), 'float')
        _, PCA_cls = import_sklearn_kmeans_pca()
        pca = PCA_cls(n_components=2)
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
    @zhs 画出二维点图
    """

    def __init__(self,
                 label_field='label', coordinates_field='coords',
                 notation_field='content', notation_length=5) -> None:
        '''
        Args:
            label_field (str): Label field
                @zhs 聚类标签字段名
            notation_field (str): Notation field
                @zhs 标记文本字段名
            notation_length (int): Max notation text length
                @zhs 标记文本最大长度
        '''
        self.label_field = label_field
        self.notation_field = notation_field
        self.notation_length = notation_length
        self.coordinates_field = coordinates_field
        self.fig, self.axis = import_plt().subplots()
        super().__init__()

    def _get_label_coords_notation(self, paragraph: Paragraph) -> tuple:
        return (getattr(paragraph, self.label_field),
                getattr(paragraph, self.coordinates_field),
                getattr(paragraph, self.notation_field, '')[:self.notation_length])

    async def summarize(self, result) -> dict:
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

        return PipelineStage.return_file('png', buf.getvalue())


from typing import Dict


class GenerateCooccurance(PipelineStage):
    """Generate cooccurance matrix
    @zhs 生成共现矩阵
    """

    def __init__(self, weighted_by='vec_cos') -> None:
        """
        Args:
            weighted_by (vec_cos|token): Weight by
            @zhs 权重方式，vec_cos 表示*语段间*向量余弦相似度，token 表示语段内各词的共现次数
        """
        super().__init__()
        self.method = weighted_by
        self.counter = Counter()

    def token(self, paragraph) -> None:
        """Count from tokens"""
        for i, t in enumerate(paragraph.tokens):
            for m in paragraph.tokens[i+1:]:
                self.counter[f'{t}\n{m}'].inc()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if self.method == 'token':
            self.token(paragraph)
        return paragraph

    async def summarize(self, _) -> Dict:
        if self.method == 'vec_cos':
            sim_result = {}
            for i, para in enumerate(self.paragraphs):
                for another in self.paragraphs[i+1:]:
                    sim_result[f'{para.content[:10]}\n{another.content[:10]}'] = float(
                        cosine_similarity(para.vec, another.vec)[0][0])
        else:
            sim_result = self.counter.as_dict()
        for k, v in sim_result.items():
            if '\n' in k:
                yield k.split('\n', 1), v


class GraphicClustering(PipelineStage):
    """Clustering by undirectional graph
    @zhs 按无向图进行聚类"""

    def __init__(self, topk=1000) -> None:
        """
        Args:
            topk (int, optional): Max edges count
                @zhs 最多绘制边的数量
        """
        self.topk = topk
        self.paragraphs = []
        super().__init__()

    async def summarize(self, result) -> dict:
        import networkx as nx
        graph = nx.Graph()
        for (node_a, node_b), val in sorted(result, key=lambda x: x[1], reverse=True)[:self.topk]:
            for i in (node_a, node_b):
                if i not in graph.nodes:
                    graph.add_node(i)
            graph.add_edge(node_a, node_b, weight=val)
            self.log(node_a, node_b, val)

        meta = ''
        for i, cluster in enumerate(nx.connected_components(graph)):
            meta += f'{i+1}\t' + '; '.join(cluster) + '\n'

        self.log(meta)

        nx.draw(graph, with_labels=True)
        buf = BytesIO()
        import_plt().savefig(buf, format='png')

        return PipelineStage.return_file('png', buf.getvalue(), meta=meta)
    