import json
import os
import re
import threading
from collections import defaultdict
from io import BytesIO
from itertools import count as iter_count
from bson import ObjectId
import jieba
import numpy as np
import pandas
from models import Paragraph
from opencc import OpenCC
from pipeline import *
from PyMongoWrapper.dbo import DbObject


class Passthrough(PipelineStage):
    """直接通过
    """

    def resolve(self, p : Paragraph) -> Paragraph:
        return p


class TradToSimpChinese(PipelineStage):
    """繁体中文转为简体中文
    """

    t2s = OpenCC('t2s')

    def resolve(self, p: Paragraph) -> Paragraph:
        p.content = TradToSimpChinese.t2s.convert(p.content)
        if p.lang == 'cht': p.lang = 'chs'
        return p


class JiebaCut(PipelineStage):
    """使用结巴分词生成检索词
    """

    def __init__(self, for_search=False, **kwargs):
        """
        Args:
            for_search (bool): 是否用于搜索（会产生冗余分词结果）
        """
        self.for_search = for_search

    def resolve(self, p: Paragraph) -> Paragraph:
        p.tokens = list(jieba.cut_for_search(p.content) if self.for_search else jieba.cut(p.content))
        return p


class WordCut(PipelineStage):
    """西文分词
    """
    
    def resolve(self, p: Paragraph) -> Paragraph:
        p.tokens = [_.lower() for _ in re.split(r'[^\w]', p.content)]
        return p


class KeywordsFromTokens(PipelineStage):
    """将词袋中的分词结果加入到检索词中并删除词袋
    """
    
    def resolve(self, p: Paragraph) -> Paragraph:
        for w in set(p.tokens):
            p.keywords.append(w)
        delattr(p, 'tokens')
        p.save()
        return p


class FilterPunctuations(PipelineStage):
    """过滤标点符号
    """
    
    re_punctuations = re.compile(r'[，。「」·；□■•●『』［］【】（）\s\(\)、“”‘’《》——\-！？\.\?\!\,\'\"：\/\\\n\u3000…]')

    def resolve(self, p: Paragraph) -> Paragraph:
        p.content = FilterPunctuations.re_punctuations.sub('', p.content)
        return p


class AccumulateParagraphs(PipelineStage):
    """将遍历的段落保存起来以备下一步骤使用（通常用于导出）
    """

    def __init__(self):
        self.paragraphs = []
        self._lock = threading.Lock()

    def resolve(self, p : Paragraph):
        with self._lock:
            self.paragraphs.append(p)

    def summarize(self, *args):
        return self.paragraphs


class Export(PipelineStage):
    """结果导出为文件
    """

    def __init__(self, format='xlsx', limit=0) -> None:
        """导出结果

        Args:
            format (xlsx|json|csv): 输出格式。
            limit (int, optional): 最多导出的记录数量，0表示无限制。
        """
        self.format = format
        self.limit = limit

    def summarize(self, r):

        def json_dump(v):
            try:
                return json.dump(v)
            except:
                return str(v)

        r = [_.as_dict() if isinstance(_, DbObject) else _ for _ in r ]

        if self.format == 'json':
            return {
                '__file_ext__': 'json',
                'data': json_dump(r).encode('utf-8')
            }

        elif self.format == 'csv':
            b = BytesIO()
            pandas.DataFrame(r).to_csv(b)
            return {
                '__file_ext__': 'csv',
                'data': b.get_value()
            }

        elif self.format == 'xlsx':
            b = BytesIO()
            pandas.DataFrame(r).to_excel(b, engine='xlsxwriter')
            return {
                '__file_ext__': 'xlsx',
                'data': b.getvalue()
            }


class AutoSummary(PipelineStage):
    """中文自动摘要
    """
    def __init__(self, count) -> None:
        """
        Args:
            count (int): 摘要中的句子数量
        """
        self.count = count

    def resolve(self, p: Paragraph) -> Paragraph:
        from textrank4zh import TextRank4Keyword, TextRank4Sentence
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=p.content, lower=True, source='all_filters')
        p.summary = '\n'.join([
            item.sentence
            for item in tr4s.get_key_sentences(num=self.count)
        ])
        return p


class Coocurrence(PipelineStage):
    """从 tokens 生成共现词对
    """
    
    def resolve(self, p: Paragraph) -> Paragraph:
        tokens = list(set(p.tokens))
        for i, t in enumerate(tokens):
            for q in tokens[i+1:]:
                p.tokens.append(f'{t}::{q}')


class Counter:

    class _CounterNum:

        def __init__(self):
            self._number_of_read = 0
            self._counter = iter_count()
            
        def value(self):
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
            return value

        def inc(self, d=1):
            for i in range(d):
                next(self._counter)

    def __init__(self) -> None:
        self._d = defaultdict(Counter._CounterNum)

    def __getitem__(self, key):
        return self._d[key]

    def as_dict(self):
        return {
            k: v.value() for k, v in self._d.items()
        }


class NgramCounter(PipelineStage):
    """N-Gram 计数
    """

    def __init__(self, n : int, lr=False):
        """ N-Gram

        Args:
            n (int): 最大字串长度
            lr (bool): 是否同时记录左右字符计数
        """
        if lr: n += 2
        self.n = n
        self.lr = lr
        self.ngrams = Counter()
        self.ngrams_lefts = defaultdict(Counter)
        self.ngrams_rights = defaultdict(Counter)
    
    def resolve(self, p : Paragraph) -> Paragraph:
        ngrams = [' ' * i for i in range(self.n)]
        for c in p.content:
            for i in range(self.n):
                ngrams[i] = (ngrams[i] + c)[-i-1:]
                self.ngrams[ngrams[i]].inc()
            if self.lr:
                for i in range(2, self.n):
                    left, word, right = ngrams[i][0], ngrams[i][1:-1], ngrams[i][-1]
                    self.ngrams_lefts[word][left].inc()
                    self.ngrams_rights[word][right].inc()

    def summarize(self, returned):
        self.ngrams = self.ngrams.as_dict()
        self.ngrams_lefts = {k: v.as_dict() for k, v in self.ngrams_lefts.items()}
        self.ngrams_rights = {k: v.as_dict() for k, v in self.ngrams_rights.items()}
