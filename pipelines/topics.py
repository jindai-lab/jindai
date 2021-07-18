"""话题与分类
"""

from pipeline import PipelineStage
from models import Paragraph
from collections import defaultdict
import numpy as np
import many_stop_words
import os
from .basics import AccumulateParagraphs

STOPWORDS = '''——— 》）， ）÷（１－ ”， ）、 ＝（ : → ℃ & * 一一 ~~~~ ’ . 『 .一 ./ -- 』 ＝″ 【 ［＊］ ｝＞ ［⑤］］ ［①Ｄ］ ｃ］ ｎｇ昉 ＊ // ［ ］ ［②ｅ］ ［②ｇ］ ＝｛ } ，也 ‘ Ａ ［①⑥］ ［②Ｂ］ ［①ａ］
 ［④ａ］ ［①③］ ［③ｈ］ ③］ １． －－ ［②ｂ］ ’‘ ××× ［①⑧］ ０：２ ＝［ ［⑤ｂ］ ［②ｃ］ ［④ｂ］ ［②③］ ［③ａ］ ［④ｃ］ ［①⑤］ ［①⑦］ ［①ｇ］ ∈［ ［①⑨］ ［①④］ ［①ｃ］ ［②ｆ］ ［②⑧］
［②①］ ［①Ｃ］ ［③ｃ］ ［③ｇ］ ［②⑤］ ［②②］ 一. ［①ｈ］ .数 ［］ ［①Ｂ］ 数/ ［①ｉ］ ［③ｅ］ ［①①］ ［④ｄ］ ［④ｅ］ ［③ｂ］ ［⑤ａ］ ［①Ａ］ ［②⑧］ ［②⑦］ ［①ｄ］ ［②ｊ］ 〕〔 ］［ :// ′∈
［②④ ［⑤ｅ］ １２％ ｂ］ ... ................... …………………………………………………③ ＺＸＦＩＴＬ ［③Ｆ］ 」 ［①ｏ］ ］∧′＝［ ∪φ∈ ′｜ ｛－ ②ｃ ｝ ［③①］ Ｒ．Ｌ． ［①Ｅ］ Ψ －［＊］－ ↑ .日 ［②ｄ
］ ［② ［②⑦］ ［②②］ ［③ｅ］ ［①ｉ］ ［①Ｂ］ ［①ｈ］ ［①ｄ］ ［①ｇ］ ［①②］ ［②ａ］ ｆ］ ［⑩］ ａ］ ［①ｅ］ ［②ｈ］ ［②⑥］ ［③ｄ］ ［②⑩］ ｅ］ 〉 】 元／吨 ［②⑩］ ２．３％ ５：０ ［①］ :: ［②
］ ［③］ ［④］ ［⑤］ ［⑥］ ［⑦］ ［⑧］ ［⑨］ …… —— ? 、 。 “ ” 《 》 ！ ， ： ； ？ ． , ． \' ? · ——— ── ? — < > （ ） 〔 〕 [ ] ( ) - + ～ × ／ / ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨ ⑩ Ⅲ В " ; # @ γ μ
φ φ． × Δ ■ ▲ sub exp sup sub Lex ＃ ％ ＆ ＇ ＋ ＋ξ ＋＋ － －β ＜ ＜± ＜Δ ＜λ ＜φ ＜＜ = ＝ ＝☆ ＝－ ＞ ＞λ ＿ ～± ～＋ ［⑤ｆ］ ［⑤ｄ］ ［②ｉ］ ≈ ［②Ｇ］ ［①ｆ］ ＬＩ ㈧ ［－ ...... 〉 ［③⑩］
第二 一番 一直 一个 一些 许多 种 有的是 也就是说 末##末 啊 阿 哎 哎呀 哎哟 唉 俺 俺们 按 按照 吧 吧哒 把 罢了 被 本 本着 比 比方 比如 鄙人 彼 彼此 边 别 别的 别说 并 并且 不比 不成 不单 不但 不独 不管 不光 不过 不仅
不拘 不论 不怕 不然 不如 不特 不惟 不问 不只 朝 朝着 趁 趁着 乘 冲 除 除此之外 除非 除了 此 此间 此外 从 从而 打 待 但 但是 当 当着 到 得 的 的话 等 等等 地 第 叮咚 对 对于 多 多少 而 而况 而且 而是 而外 而言 而已 尔
后 反过来 反过来说 反之 非但 非徒 否则 嘎 嘎登 该 赶 个 各 各个 各位 各种 各自 给 根据 跟 故 故此 固然 关于 管 归 果然 果真 过 哈 哈哈 呵 和 何 何处 何况 何时 嘿 哼 哼唷 呼哧 乎 哗 还是 还有 换句话说 换言之 或 或是 或
者 极了 及 及其 及至 即 即便 即或 即令 即若 即使 几 几时 己 既 既然 既是 继而 加之 假如 假若 假使 鉴于 将 较 较之 叫 接着 结果 借 紧接着 进而 尽 尽管 经 经过 就 就是 就是说 据 具体地说 具体说来 开始 开外 靠 咳 可 可见
 可是 可以 况且 啦 来 来着 离 例如 哩 连 连同 两者 了 临 另 另外 另一方面 论 嘛 吗 慢说 漫说 冒 么 每 每当 们 莫若 某 某个 某些 拿 哪 哪边 哪儿 哪个 哪里 哪年 哪怕 哪天 哪些 哪样 那 那边 那儿 那个 那会儿 那里 那么 那
么些 那么样 那时 那些 那样 乃 乃至 呢 能 你 你们 您 宁 宁可 宁肯 宁愿 哦 呕 啪达 旁人 呸 凭 凭借 其 其次 其二 其他 其它 其一 其余 其中 起 起见 起见 岂但 恰恰相反 前后 前者 且 然而 然后 然则 让 人家 任 任何 任凭 如 如
此 如果 如何 如其 如若 如上所述 若 若非 若是 啥 上下 尚且 设若 设使 甚而 甚么 甚至 省得 时候 什么 什么样 使得 是 是的 首先 谁 谁知 顺 顺着 似的 虽 虽然 虽说 虽则 随 随着 所 所以 他 他们 他人 它 它们 她 她们 倘 倘或 倘
然 倘若 倘使 腾 替 通过 同 同时 哇 万一 往 望 为 为何 为了 为什么 为着 喂 嗡嗡 我 我们 呜 呜呼 乌乎 无论 无宁 毋宁 嘻 吓 相对而言 像 向 向着 嘘 呀 焉 沿 沿着 要 要不 要不然 要不是 要么 要是 也 也罢 也好 一 一般 一旦
一方面 一来 一切 一样 一则 依 依照 矣 以 以便 以及 以免 以至 以至于 以致 抑或 因 因此 因而 因为 哟 用 由 由此可见 由于 有 有的 有关 有些 又 于 于是 于是乎 与 与此同时 与否 与其 越是 云云 哉 再说 再者 在 在下 咱 咱们
则 怎 怎么 怎么办 怎么样 怎样 咋 照 照着 者 这 这边 这儿 这个 这会儿 这就是说 这里 这么 这么点儿 这么些 这么样 这时 这些 这样 正如 吱 之 之类 之所以 之一 只是 只限 只要 只有 至 至于 诸位 着 着呢 自 自从 自个儿 自各儿
自己 自家 自身 综上所述 总的来看 总的来说 总的说来 总而言之 总之 纵 纵令 纵然 纵使 遵照 作为 兮 呃 呗 咚 咦 喏 啐 喔唷 嗬 嗯 嗳'''


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

    def __init__(self, stopwords=STOPWORDS):
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
