f"""基本操作"""

import json
import re
import statistics
import threading
from collections import defaultdict, deque
from io import BytesIO
from itertools import count as iter_count
from bson import ObjectId
from chardet import detect
import jieba
import pandas
import json
import itertools
from opencc import OpenCC
import pykakasi
from transliterate import translit

from PyMongoWrapper import F, QueryExprParser
from PyMongoWrapper.dbo import DbObject, DbObjectCollection
from .utils import execute_query_expr, language_iso639
from helpers import get_converter

from models import Paragraph, Dataset, AutoTag, parser, db
from pipeline import Pipeline, PipelineStage


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


class LanguageDetect(PipelineStage):    
    """简易语言检测
    使用正则表达式和 hanzidentifier 弥补 langdetect 在检测中日韩文字时准确率低的问题，返回 ISO 两字母代码或 chs 或 cht。
    """

    def resolve(self, p : Paragraph) -> Paragraph:
        if p.lang and p.lang != 'auto':
            return p
        
        if p.content:
            p.lang = self.detect(p.content)
            if p.lang in ('zh-cn', 'zh-sg'): p.lang = 'chs'
            elif p.lang.startswith('zh-'): p.lang = 'cht'
            elif '-' in p.lang: p.lang = p.lang.split('-')[0]
            return p

    def detect(self, s):
        import hanzidentifier, langdetect

        s = re.sub('[0-9]', '', s).strip()
        
        if re.search(r"[\uac00-\ud7ff]+", s):
            return 'ko'

        if re.search(r"[\u30a0-\u30ff\u3040-\u309f]+", s):
            return 'ja'
        
        if hanzidentifier.has_chinese(s):
            if hanzidentifier.is_simplified(s):
                return 'chs'
            else:
                return 'cht'
        
        return langdetect.detect(s)


class JiebaCut(PipelineStage):
    """使用结巴分词生成检索词
    """

    def resolve(self, p: Paragraph) -> Paragraph:
        p.tokens = list(jieba.cut_for_search(p.content) if self.for_search else jieba.cut(p.content))
        return p


class WordStemmer(PipelineStage):
    """附加词干到 tokens 中（需要先进行切词）
    """

    _language_stemmers = {}

    @staticmethod
    def get_stemmer(lang):
        from nltk.stem.snowball import SnowballStemmer 
        if lang not in WordStemmer._language_stemmers:
            stemmer = SnowballStemmer(language_iso639.get(lang, lang).lower())
            WordStemmer._language_stemmers[lang] = stemmer
        return WordStemmer._language_stemmers[lang]

    def __init__(self, append=True):
        """
        Args:
            append (bool): 将词干添加到结尾，否则直接覆盖
        """
        self.append = append

    def resolve(self, p : Paragraph) -> Paragraph:
        tokens = [WordStemmer.get_stemmer(p.lang).stem(_) for _ in p.tokens]
        if self.append:
            p.tokens += tokens
        else:
            p.tokens = tokens
        return p


class WordCut(PipelineStage):
    """多语种分词
    """

    t2s = OpenCC('t2s')
    kks = pykakasi.kakasi()
    stmr = WordStemmer(append=True)

    def __init__(self, for_search=False, **kwargs):
        """
        Args:
            for_search (bool): 是否用于搜索（添加冗余分词结果或词干/转写）
        """
        self.for_search = for_search
    
    def resolve(self, p: Paragraph) -> Paragraph:
        p.tokens = []

        if p.lang == 'cht':
            p.content = WordCut.t2s.convert(p.content)

        if p.lang in ('chs', 'cht'):
            p.tokens = list(jieba.cut_for_search(p.content) if self.for_search else jieba.cut(p.content))
        elif p.lang == 'ja':
            p.tokens = list(set(p.content))
            for i in WordCut.kks.convert(p.content):
                p.tokens.append(i['orig'])
                if self.for_search: p.tokens.append(i['hepburn'])
        else:
            p.tokens = [_.lower() for _ in re.split(r'[^\w]', p.content)]
            if self.for_search:
                WordCut.stmr.resolve(p)

        if self.for_search and p.lang not in ('chs', 'cht', 'en', 'ja'):
            p.tokens += [translit(_, p.lang, reversed=True).lower() for _ in p.tokens]

        return p


class KeywordsFromTokens(PipelineStage):
    """将检索词设为分词结果并删除词串字段
    """
    
    def resolve(self, p: Paragraph) -> Paragraph:
        p.keywords = list(set(p.tokens))
        del p.tokens
        p.save()
        return p


class FilterPunctuations(PipelineStage):
    """过滤标点符号
    """
    
    re_punctuations = re.compile(r'[，。「」·；□■•●『』［］【】（）\s\(\)、“”‘’《》——\-！？\.\?\!\,\'\"：\/\\\n\u3000…]')

    def resolve(self, p: Paragraph) -> Paragraph:
        p.content = FilterPunctuations.re_punctuations.sub('', p.content)
        return p


class Reparagraph(PipelineStage):
    """重新分段"""

    def resolve(self, p : Paragraph) -> Paragraph:
        lang = p.lang
        lines = p.content.split('\n')
        
        def paragraph_finished(t):
            return t.endswith(tuple('.!?…\"。！？…—：”）'))

        def merge_lines():
            lens = [len(_) for _ in lines]
            if len(lens) < 3:
                yield ('' if lang[:2] in ('ch', 'ja') else ' ').join(lines)
                return

            std = abs(statistics.stdev(lens))
            maxl = max(lens)
            t = ''
            last_line = '1'
            for l in lines:
                l = l.strip()
                if not l:
                    continue
                if re.search(r'^[①-⑩]', l):
                    break

                if lang[:2] != 'ch':
                    t += ' '
                t += l
                if len(l) < maxl - std:
                    if paragraph_finished(t) or not last_line:
                        yield t
                        t = ''
                last_line = l.strip()

            if t:
                yield t

        pd = p.as_dict()
        del pd['content']
        for t in merge_lines():
            yield Paragraph(content=t, **pd)


class AccumulateParagraphs(PipelineStage):
    """将遍历的段落保存起来以备下一步骤使用（通常用于导出）
    """

    def __init__(self):
        self.paragraphs = deque()

    def resolve(self, p : Paragraph):
        self.paragraphs.append(p)

    def summarize(self, *args):
        return list(self.paragraphs)


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
        from textrank4zh import TextRank4Sentence
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=p.content, lower=True, source='all_filters')
        p.summary = '\n'.join([
            item.sentence
            for item in tr4s.get_key_sentences(num=self.count)
        ])
        return p


class ArrayField(PipelineStage):
    """操作数组字段"""

    def __init__(self, field, pull=True, elements='') -> None:
        """
        Args:
            field (str): 字段名
            pull (bool): 添加（true）或删除（false）
            elements (str): 添加或删除的元素；使用分号分隔，作为表达式运算
        """
        self.field = field
        self.elements = parser.eval(elements)
        self.pull = pull

    def resolve(self, p: Paragraph) -> Paragraph:
        if not isinstance(p[self.field], (list, DbObjectCollection)): return p
        for ele in self.elements:
            if self.pull:
                p[self.field].append(ele)
            else:
                if ele in p[self.field]:
                    p[self.field].remove(ele)
        return p
        

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


class Limit(PipelineStage):
    """限制返回的结果数量
    """
    
    def __init__(self, limit):
        """
        Args:
            limit (int): 要返回的最大结果数量，0则不返回
        """
        self.limit = limit
        self.counter = itertools.count()

    def resolve(self, p : Paragraph):
        v = next(self.counter)
        if v < self.limit:
            return p
        
        
class FilterDuplication(PipelineStage):
    """过滤已经存储在指定数据库中的段落
    """
    
    def __init__(self, field, mongocollection='paragraph') -> None:
        """
        Args:
            mongocollection (str): 数据库集合名
            field (str): 要去重的字段值
        """
        self.mongocollection = mongocollection or 'paragraph'
        self.field = field
        
    def resolve(self, p: Paragraph) -> Paragraph:
        for _ in db[self.mongocollection].find({self.field : getattr(p, self.field)}):
            return
        return p


class RegexReplace(PipelineStage):
    """正则表达式匹配并替换
    """

    def __init__(self, pattern, replacement='', plain=False):
        """
        Args:
            pattern (str): 正则表达式
            replacement (str): 要替换成的字符串
        """
        if plain:
            pattern = re.escape(pattern)
        self.re = re.compile(pattern)
        self.replacement = replacement

    def resolve(self, p : Paragraph) -> Paragraph:
        p.content = self.re.sub(self.replacement, p.content)
        return p


class RegexFilter(PipelineStage):
    """正则表达式匹配并提取到字段中
    """

    def __init__(self, pattern, target, source='content', continuous=False, filter_out=False):
        """
        Args:
            pattern (str): 正则表达式
            source (str): 匹配的字段，默认为内容
            target (str): 要提取入的字段名称
            filter_out (bool): 过滤未匹配到的语段
            continuous (bool): 未匹配到的语段自动使用上次的值
        """
        self.re = re.compile(pattern)
        self.source = source
        self.target = target
        self.value = ''
        self.continuous = continuous
        self.filter_out = filter_out
        
    def resolve(self, p : Paragraph) -> Paragraph:
        match = self.re.search(str(getattr(p, self.source, '')))
        if match:
            self.value = match.group(1) or match.group(0)
        if match or self.continuous:
            setattr(p, self.target, self.value)
        else:
            return
        return p


class FieldAssignment(PipelineStage):
    """将某一个字段的值或输入值保存到另一个字段
    """

    def __init__(self, field, value):
        """
        Args:
            field (str): 新的字段名
            value (str): 以 $ 开头的字段名，或常数值（类型将自动匹配），或 $$oid 表示一个新的 ObjectId
        """
        self.field = field
        self.specials = {
            '$$oid': lambda: ObjectId()
        }
        if value in self.specials:
            self.valueLiteral = self.specials[value]
            self.valueField = None
        elif value.startswith('$'):
            self.valueField = value[1:]
            self.valueLiteral = None
        else:
            self.valueLiteral = parser.parse_literal(value)
            self.valueField = None

    def value(self, p : Paragraph):
        if self.valueField is None:
            if hasattr(self.valueLiteral, '__call__'):
                return self.valueLiteral()
            return self.valueLiteral
        else:
            return getattr(p, self.valueField, '')

    def resolve(self, p : Paragraph) -> Paragraph:
        setattr(p, self.field, self.value(p))
        return p


class FilterArrayField(PipelineStage):
    """过滤列表字段的值
    """

    def __init__(self, field, cond) -> None:
        """
        Args:
            field (str): 字段名称
            cond (QUERY): 条件式，用 iter 表示被判断的项目，或用省略形式
        """
        self.field = field
        self.cond = QueryExprParser(allow_spacing=True, abbrev_prefixes={None: 'iter='}).eval(cond)

    def resolve(self, p: Paragraph) -> Paragraph:
        v = getattr(p, self.field, [])
        if not isinstance(v, list):
            return p
        
        nv = []
        for iter in v:
            p.iter = iter
            if execute_query_expr(self.cond, p):
                nv.append(iter)
        
        if hasattr(p, 'iter'): delattr(p, 'iter')
        setattr(p, self.field, nv)
        return p


class SaveParagraph(PipelineStage):
    """保存
    """

    def __init__(self, mongocollection='paragraph'):
        '''
        Args:
            mongocollection (str): 数据库目标数据集名称
        '''
        self.mongocollection = mongocollection
        self.datasets = defaultdict(set)
        self.convert = get_converter(mongocollection)

    def resolve(self, p : Paragraph):
        self.convert(p).save()
        if 'file' in p.source:
            self.datasets[p.dataset].add(p.source['file'])
        return p

    def summarize(self, returned):
        for c, sources in self.datasets.items():
            coll = Dataset.first(F.name == c) \
                    or Dataset(name=c, sources=[], mongocollection=self.mongocollection, order_weight=999)
            for s in sources:
                if s not in coll.sources:
                    coll.sources.append(s)
            coll.save()


class FieldIncresement(PipelineStage):
    """对字段进行自增操作
    """

    def __init__(self, field, inc_value):
        '''
        Args:
            field (str): 字段名称
            inc_value (str): 自增的值，或以 $ 开头的另一字段名
        '''
        self.field = field
        if inc_value.startswith('$'):
            self.inc_field = inc_value[1:]
            self.inc_value = ''
        else:
            self.inc_value = parser.parse_literal(inc_value)
            self.inc_field = ''

    def resolve(self, p : Paragraph):
        v = getattr(p, self.field)
        v += self.inc_value if self.inc_value else getattr(self, self.inc_field)
        setattr(p, v)
        return p


class OutlineFilter(PipelineStage):
    """中英文大纲序号识别
    """

    chnum = '[一二三四五六七八九十首甲乙丙丁戊己庚辛壬癸]'
    romannum = (',I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,').split(',')
    
    def __init__(self):
        super().__init__()
        self.nums = ['00', '00', '00']

    def roman(self, x):
        if '.' in x:
            x = x[:x.find('.')]
        return OutlineFilter.romannum.index(x) if x in OutlineFilter.romannum else 99

    def dechnum(self, x):
        ns = [(OutlineFilter.chnum+_).find(_) for _ in x]
        if len(ns) == 1:
            if ns[0] > 10:
                return ns[0] - 11
            else:
                return ns[0]
        elif len(ns) == 2:
            if ns[0] == 10:
                return ns[0] + ns[1]
            else:
                return -1
        else:
            return ns[0]*ns[1]+ns[2]

    def check_outline(self, p: Paragraph):
        lang, content = p.lang, p. content
        outline = ''

        if lang == 'eng':
            lev, num, _ = (content + "  ").split(" ", 2)
            if lev == '§':
                outline = 'sect {:02}'.format(int(num[:-1]))
            elif lev.upper() == 'CHAPTER':
                outline = 'chap {:02}'.format(self.roman(num))
            elif lev.upper() == 'BOOK':
                outline = 'book {:02}'.format(self.roman(num))
            elif lev.upper().startswith('INTRODUCTION'):
                outline = 'book 00'
        else:
            if re.match('^' + OutlineFilter.chnum + '+、', content):
                outline = 'sect {:02}'.format(
                    self.deOutlineFilter.chnum(content[:content.find('、')]))
            elif re.match('^第' + OutlineFilter.chnum + '+章', content):
                outline = 'chap {:02}'.format(
                    self.deOutlineFilter.chnum(content[1:content.find('章')]))
            elif re.match('^第' + OutlineFilter.chnum + '+節', content):
                outline = 'sect {:02}'.format(
                    self.deOutlineFilter.chnum(content[1:content.find('節')]))
            elif re.match('^第' + OutlineFilter.chnum + '卷', content):
                outline = 'book {:02}'.format(
                    self.deOutlineFilter.chnum(content[1:content.find('卷')]))
            elif re.match('^篇' + OutlineFilter.chnum, content):
                outline = 'chap {:02}'.format(self.deOutlineFilter.chnum(content[1]))
            elif re.match('^部' + OutlineFilter.chnum, content):
                outline = 'book {:02}'.format(self.deOutlineFilter.chnum(content[1]))

        return outline

    def resolve(self, p : Paragraph):
        p.content = p.content.strip()
        if not p.content:
            return p

        outline = self.check_outline(p)

        if outline and outline[5] != '-':
            # self.logger(content[:20], outline)
            if outline.startswith('book '):
                nnums = [outline[5:], '00', '00']
            elif outline.startswith('chap '):
                nnums = [self.nums[0], outline[5:], '00']
            else:
                nnums = [self.nums[0], self.nums[1], outline[5:]]
            if '.'.join(nnums) > '.'.join(self.nums):
                self.nums = nnums

        p.outline = '.'.join(self.nums)
        p.save()
        return p


class LatinTransliterate(PipelineStage):
    """转写为拉丁字母的单词（需要先进行切词）
    """
    def __init__(self, append=True):
        """
        Args:
            append (bool): 是添加到结尾还是覆盖
        """
        self.append = append
        
    def resolve(self, p : Paragraph) -> Paragraph:
        from transliterate import translit
        tokens = [translit(_, 'ru', reversed=True).lower() for _ in p.tokens]
        if self.append:
            p.tokens += tokens
        else:
            p.tokens = tokens
        return p


class ApplyAutoTags(PipelineStage):
    """应用自动标签设置
    """
    
    def __init__(self) -> None:
        self.ats = list(AutoTag.query({}))
    
    def resolve(self, p):
        for i in self.ats:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in p.keywords) or (pattern and re.search(pattern, p.source['url'])):
                if tag not in p.keywords:
                    p.keywords.append(tag)
                if tag.startswith('@'):
                    p.author = tag
        p.save()
        return p

