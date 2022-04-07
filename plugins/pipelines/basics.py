"""基本操作"""

import json
import re
import statistics
from collections import defaultdict, deque
from io import BytesIO
from itertools import chain
from itertools import count as iter_count

from PyMongoWrapper import F, QueryExprParser, ObjectId
from PyMongoWrapper.dbo import DbObject, DbObjectCollection
from jindai import PipelineStage
from jindai.helpers import execute_query_expr, language_iso639, safe_import
from jindai.models import Dataset, Paragraph, db, parser


class Passthrough(PipelineStage):
    """直接通过
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        return paragraph


class TradToSimpChinese(PipelineStage):
    """繁体中文转为简体中文
    """

    t2s = safe_import('opencc', 'opencc-python-reimplementation').OpenCC('t2s')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = TradToSimpChinese.t2s.convert(paragraph.content)
        if paragraph.lang == 'cht':
            paragraph.lang = 'chs'
        return paragraph


class LanguageDetect(PipelineStage):
    """简易语言检测
    使用正则表达式和 hanzidentifier 弥补 langdetect 在检测中日韩文字时准确率低的问题，返回 ISO 两字母代码或 chs 或 cht。
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph.lang and paragraph.lang != 'auto':
            return paragraph

        if paragraph.content:
            paragraph.lang = self.detect(paragraph.content)
            if paragraph.lang in ('zh-cn', 'zh-sg'):
                paragraph.lang = 'chs'
            elif paragraph.lang.startswith('zh-'):
                paragraph.lang = 'cht'
            elif '-' in paragraph.lang:
                paragraph.lang = paragraph.lang.split('-')[0]
            return paragraph

    def detect(self, sentence):
        """检测语言"""

        hanzidentifier = safe_import('hanzidentifier')
        langdetect = safe_import('langdetect')

        sentence = re.sub('[0-9]', '', sentence).strip()

        if re.search(r"[\uac00-\ud7ff]+", sentence):
            return 'ko'

        if re.search(r"[\u30a0-\u30ff\u3040-\u309f]+", sentence):
            return 'ja'

        if hanzidentifier.has_chinese(sentence):
            if hanzidentifier.is_simplified(sentence):
                return 'chs'
            else:
                return 'cht'

        return langdetect.detect(sentence)


class WordStemmer(PipelineStage):
    """附加词干到 tokens 中（需要先进行切词）
    """

    _language_stemmers = {}

    @staticmethod
    def get_stemmer(lang):
        """Get stemmer for language"""
        safe_import('nltk')
        stemmer = safe_import('nltk.stem.snowball').SnowballStemmer
        if lang not in WordStemmer._language_stemmers:
            lang = language_iso639.get(lang, lang).lower()
            if lang not in stemmer.languages:
                return WordStemmer.get_stemmer('en')
            stemmer = stemmer(lang)
            WordStemmer._language_stemmers[lang] = stemmer
        return WordStemmer._language_stemmers[lang]

    def __init__(self, append=True):
        """
        Args:
            append (bool): 将词干添加到结尾，否则直接覆盖
        """
        super().__init__()
        self.append = append

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        tokens = [WordStemmer.get_stemmer(
            paragraph.lang).stem(_) for _ in paragraph.tokens]
        if self.append:
            paragraph.tokens += tokens
        else:
            paragraph.tokens = tokens
        return paragraph


class LatinTransliterate(PipelineStage):
    """转写为拉丁字母的单词（需要先进行切词）
    """

    def __init__(self, append=True):
        """
        Args:
            append (bool): 是添加到结尾还是覆盖
        """
        super().__init__()
        self.append = append
        transliterate = safe_import('transliterate')
        self.supported_languages = transliterate.get_available_language_codes()
        self.translit = transliterate.translit

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph.lang in self.supported_languages:
            tokens = [self.translit(
                _, paragraph.lang, reversed=True).lower() for _ in paragraph.tokens]
            if self.append:
                paragraph.tokens += tokens
            else:
                paragraph.tokens = tokens
        return paragraph


class WordCut(PipelineStage):
    """多语种分词
    """

    t2s = safe_import('opencc', 'opencc-python-reimplementation').OpenCC('t2s')
    kks = safe_import('pykakasi').kakasi()
    jieba = safe_import('jieba')
    stmr = WordStemmer(append=True)
    trlit = LatinTransliterate(append=True)

    def __init__(self, for_search=False, **_):
        """
        Args:
            for_search (bool): 是否用于搜索（添加冗余分词结果或词干/转写）
        """
        super().__init__()
        self.for_search = for_search

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.tokens = []

        if paragraph.lang == 'cht':
            paragraph.content = WordCut.t2s.convert(paragraph.content)

        if paragraph.lang in ('chs', 'cht'):
            paragraph.tokens = list(WordCut.jieba.cut_for_search(paragraph.content)
                                    if self.for_search else WordCut.jieba.cut(paragraph.content))
        elif paragraph.lang == 'ja':
            paragraph.tokens = list(set(paragraph.content))
            for i in WordCut.kks.convert(paragraph.content):
                paragraph.tokens.append(i['orig'])
                if self.for_search:
                    paragraph.tokens.append(i['hepburn'])
        else:
            paragraph.tokens = [_.lower()
                                for _ in re.split(r'[^\w]', paragraph.content)]
            if self.for_search:
                WordCut.stmr.resolve(paragraph)

        if self.for_search:
            WordCut.trlit.resolve(paragraph)

        return paragraph


class KeywordsFromTokens(PipelineStage):
    """将检索词设为分词结果并删除词串字段
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.keywords = list(set(paragraph.tokens))
        del paragraph.tokens
        paragraph.save()
        return paragraph


class FilterPunctuations(PipelineStage):
    """过滤标点符号
    """

    re_punctuations = re.compile(
        r'[，。「」·；□■•●『』［］【】（）\s\(\)、“”‘’《》——\-！？\.\?\!\,\'\"：\/\\\n\u3000…]')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = FilterPunctuations.re_punctuations.sub(
            '', paragraph.content)
        return paragraph


class Reparagraph(PipelineStage):
    """重新分段"""

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        lang = paragraph.lang
        lines = paragraph.content.split('\n')

        def paragraph_finished(text):
            return text.endswith(tuple('.!?…\"。！？…—：”）'))

        def merge_lines():
            lens = [len(_) for _ in lines]
            if len(lens) < 3:
                yield ('' if lang[:2] in ('ch', 'ja') else ' ').join(lines)
                return

            std = abs(statistics.stdev(lens))
            maxl = max(lens)
            text = ''
            last_line = '1'
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if re.search(r'^[①-⑩]', line):
                    break

                if lang[:2] != 'ch':
                    text += ' '
                text += line
                if len(line) < maxl - std:
                    if paragraph_finished(text) or not last_line:
                        yield text
                        text = ''
                last_line = line.strip()

            if text:
                yield text

        data = paragraph.as_dict()
        del data['content']
        for text in merge_lines():
            yield Paragraph(content=text, **data)


class SplitParagraph(PipelineStage):
    """拆分语段
    """

    def __init__(self, delimeter='\n'):
        """
        Args:
            delimeter (str): 拆分的分隔符
        """
        super().__init__()
        self.delimeter = delimeter

    def resolve(self, paragraph: Paragraph):
        for content in paragraph.content.split(self.delimeter):
            if content:
                new_paragraph = Paragraph(paragraph)
                new_paragraph.content = content
                yield new_paragraph


class AccumulateParagraphs(PipelineStage):
    """将遍历的段落保存起来以备下一步骤使用（通常用于导出）
    """

    def __init__(self):
        super().__init__()
        self.paragraphs = deque()

    def resolve(self, paragraph: Paragraph):
        self.paragraphs.append(paragraph)

    def summarize(self, *_):
        return list(self.paragraphs)


class Export(PipelineStage):
    """结果导出为文件
    """

    def __init__(self, output_format='xlsx', limit=0) -> None:
        """导出结果

        Args:
            format (xlsx|json|csv): 输出格式。
            limit (int, optional): 最多导出的记录数量，0表示无限制。
        """
        super().__init__()
        self.format = output_format
        self.limit = limit

    def summarize(self, result):
        safe_import('xlsxwriter')
        pandas = safe_import('pandas')

        def json_dump(val):
            try:
                return json.dumps(val)
            except Exception:
                return str(val)

        result = [_.as_dict() if isinstance(_, DbObject) else _ for _ in result]

        if self.format == 'json':
            return {
                '__file_ext__': 'json',
                'data': json_dump(result).encode('utf-8')
            }

        elif self.format == 'csv':
            buf = BytesIO()
            pandas.DataFrame(result).to_csv(buf)
            return {
                '__file_ext__': 'csv',
                'data': buf.getvalue()
            }

        elif self.format == 'xlsx':
            buf = BytesIO()
            pandas.DataFrame(result).to_excel(buf, engine='xlsxwriter')
            return {
                '__file_ext__': 'xlsx',
                'data': buf.getvalue()
            }


class AutoSummary(PipelineStage):
    """中文自动摘要
    """

    def __init__(self, count) -> None:
        """
        Args:
            count (int): 摘要中的句子数量
        """
        super().__init__()
        self.count = count

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        tr4s = safe_import('textrank4zh').TextRank4Sentence()
        tr4s.analyze(text=paragraph.content, lower=True, source='all_filters')
        paragraph.summary = '\n'.join([
            item.sentence
            for item in tr4s.get_key_sentences(num=self.count)
        ])
        return paragraph


class ArrayField(PipelineStage):
    """操作数组字段"""

    def __init__(self, field, push=True, elements='') -> None:
        """
        Args:
            field (str): 字段名
            push (bool): 添加（true）或删除（false）
            elements (str): 添加或删除的元素，每行一个 $ 开头的字段名或常量
        """
        super().__init__()
        self.field = field
        self.elements = []
        try:
            elements = parser.eval(elements)
            assert isinstance(elements, list)
            self.elements = elements
        except Exception:
            for ele in elements.split('\n'):
                ele = parser.eval(ele)
                self.elements.append(ele)
        self.push = push

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph[self.field] is None and self.push:
            paragraph[self.field] = []
        if not isinstance(paragraph[self.field], (list, DbObjectCollection)):
            return paragraph
        for ele in self.elements:
            if isinstance(ele, str) and ele.startswith('$'):
                ele = getattr(paragraph, ele[1:], '')
                if ele is None:
                    continue
            if self.push:
                paragraph[self.field].append(ele)
            else:
                if ele in paragraph[self.field]:
                    paragraph[self.field].remove(ele)
        return paragraph


class ArrayAggregation(PipelineStage):
    """减少一层数组嵌套层级"""

    def __init__(self, field, new_field='') -> None:
        """
        Args:
            field (str): 数组字段名
            new_field (str): 新的字段名，留空表示替换原数组字段
        """
        super().__init__()
        self.field = field
        self.new_field = new_field or field

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        setattr(paragraph, self.new_field, list(chain(*paragraph[self.field])))
        return paragraph


class Counter:
    """线程安全的计数器"""

    class _CounterNum:

        def __init__(self):
            self._number_of_read = 0
            self._counter = iter_count()

        def value(self):
            """Get the current value of the counter"""
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
            return value

        def inc(self, inc_val=1):
            """Increment the counter by a given value (default = 1)"""
            for _ in range(inc_val):
                next(self._counter)

    def __init__(self) -> None:
        super().__init__()
        self._d = defaultdict(Counter._CounterNum)

    def __getitem__(self, key):
        return self._d[key]

    def as_dict(self):
        """Get the dictionary representation of the counter"""
        return {
            k: v.value() for k, v in self._d.items()
        }


class NgramCounter(PipelineStage):
    """N-Gram 计数
    """

    def __init__(self, n: int, lr=False):
        """ N-Gram

        Args:
            n (int): 最大字串长度
            lr (bool): 是否同时记录左右字符计数
        """
        super().__init__()
        if lr:
            n += 2
        self.n = n
        self.lr = lr
        self.ngrams = Counter()
        self.ngrams_lefts = defaultdict(Counter)
        self.ngrams_rights = defaultdict(Counter)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        ngrams = [' ' * i for i in range(self.n)]
        for content in paragraph.content:
            for i in range(self.n):
                ngrams[i] = (ngrams[i] + content)[-i-1:]
                self.ngrams[ngrams[i]].inc()
            if self.lr:
                for i in range(2, self.n):
                    left, word, right = ngrams[i][0], ngrams[i][1:-
                                                                1], ngrams[i][-1]
                    self.ngrams_lefts[word][left].inc()
                    self.ngrams_rights[word][right].inc()

    def summarize(self, _):
        self.ngrams = self.ngrams.as_dict()
        self.ngrams_lefts = {k: v.as_dict()
                             for k, v in self.ngrams_lefts.items()}
        self.ngrams_rights = {k: v.as_dict()
                              for k, v in self.ngrams_rights.items()}


class Limit(PipelineStage):
    """限制返回的结果数量
    """

    def __init__(self, limit):
        """
        Args:
            limit (int): 要返回的最大结果数量，0则不返回
        """
        super().__init__()
        self.limit = limit
        self.counter = iter_count()

    def resolve(self, paragraph: Paragraph):
        val = next(self.counter)
        if val < self.limit:
            return paragraph


class FilterDuplication(PipelineStage):
    """过滤已经存储在指定数据库中的段落
    """

    def __init__(self, field, mongocollection='paragraph') -> None:
        """
        Args:
            mongocollection (str): 数据库集合名
            field (str): 要去重的字段值
        """
        super().__init__()
        self.mongocollection = mongocollection or 'paragraph'
        self.field = field

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        for _ in db[self.mongocollection].find({self.field: getattr(paragraph, self.field)}):
            return
        return paragraph


class RegexReplace(PipelineStage):
    """正则表达式匹配并替换
    """

    def __init__(self, pattern, replacement='', plain=False):
        """
        Args:
            pattern (str): 正则表达式
            replacement (str): 要替换成的字符串
        """
        super().__init__()
        if plain:
            pattern = re.escape(pattern)
        self.regexp = re.compile(pattern)
        self.replacement = replacement

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = self.regexp.sub(self.replacement, paragraph.content)
        return paragraph


class RegexFilter(PipelineStage):
    """正则表达式匹配并提取到字段中
    """

    def __init__(self, pattern, target, source='content', match='{0}',
                 continuous=False, filter_out=False):
        """
        Args:
            pattern (str): 正则表达式
            source (str): 匹配的字段，默认为内容
            target (str): 要提取入的字段名称
            match (str): 填入的值，如 '{1}{2}' 等形式，默认为全部匹配到的文本
            filter_out (bool): 过滤未匹配到的语段
            continuous (bool): 未匹配到的语段自动使用上次的值
        """
        super().__init__()
        self.regexp = re.compile(pattern)
        self.source = source
        self.target = target
        self.value = ''
        self.match = match
        self.continuous = continuous
        self.filter_out = filter_out

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        match = self.regexp.search(str(getattr(paragraph, self.source, '')))
        if match:
            val = self.match.format(match.group(
                0), *[_ if _ is not None else '' for _ in match.groups()])
            setattr(paragraph, self.target, val)
            self.value = val
        elif self.continuous:
            setattr(paragraph, self.target, self.value)
        elif self.filter_out:
            return
        return paragraph


class FieldAssignment(PipelineStage):
    """将某一个字段的值或输入值保存到另一个字段
    """

    def __init__(self, field, value):
        """
        Args:
            field (str): 新的字段名
            value (str): 以 $ 开头的字段名，或常数值（类型将自动匹配），或 $$oid 表示一个新的 ObjectId
        """
        super().__init__()
        self.field = field
        self.specials = {
            '$$oid': ObjectId
        }
        if value in self.specials:
            self.value_literal = self.specials[value]
            self.value_field = None
        elif value.startswith('$'):
            self.value_field = value[1:]
            self.value_literal = None
        else:
            self.value_literal = parser.parse_literal(value)
            self.value_field = None

    def value(self, paragraph: Paragraph):
        """Get value from paragraph object"""
        if self.value_field is None:
            if hasattr(self.value_literal, '__call__'):
                return self.value_literal()
            return self.value_literal
        else:
            return getattr(paragraph, self.value_field, '')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        setattr(paragraph, self.field, self.value(paragraph))
        return paragraph


class FilterArrayField(PipelineStage):
    """过滤列表字段的值
    """

    def __init__(self, field, cond) -> None:
        """
        Args:
            field (str): 字段名称
            cond (QUERY): 条件式，用 iter 表示被判断的项目，或用省略形式
        """
        super().__init__()
        self.field = field
        self.cond = QueryExprParser(allow_spacing=True, abbrev_prefixes={
                                    None: 'iter='}).eval(cond)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        vals = getattr(paragraph, self.field, [])
        if not isinstance(vals, list):
            return paragraph

        new_vals = []
        for val in vals:
            paragraph.iter = val
            if execute_query_expr(self.cond, paragraph):
                new_vals.append(val)

        if hasattr(paragraph, 'iter'):
            delattr(paragraph, 'iter')
        setattr(paragraph, self.field, new_vals)
        return paragraph


class SaveParagraph(PipelineStage):
    """保存
    """

    def __init__(self, mongocollection='paragraph'):
        '''
        Args:
            mongocollection (str): 数据库集合名
        '''
        super().__init__()
        self.mongocollection = mongocollection
        self.datasets = defaultdict(set)
        self.convert = Paragraph.get_converter(mongocollection)

    def resolve(self, paragraph: Paragraph):
        self.convert(paragraph).save()
        if 'file' in paragraph.source and paragraph.dataset:
            self.datasets[paragraph.dataset].add(paragraph.source['file'])
        return paragraph

    def summarize(self, _):
        for name, sources in self.datasets.items():
            coll = Dataset.first(F.name == name) \
                or Dataset(name=name, sources=[],
                           mongocollection=self.mongocollection, order_weight=999)
            for source in sources:
                if source not in coll.sources:
                    coll.sources.append(source)
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
        super().__init__()
        self.field = field
        if inc_value.startswith('$'):
            self.inc_field = inc_value[1:]
            self.inc_value = ''
        else:
            self.inc_value = parser.parse_literal(inc_value)
            self.inc_field = ''

    def resolve(self, paragraph: Paragraph):
        val = getattr(paragraph, self.field)
        val += self.inc_value if self.inc_value else getattr(
            self, self.inc_field)
        setattr(paragraph, val)
        return paragraph


class OutlineFilter(PipelineStage):
    """中英文大纲序号识别
    """

    chnum = '[一二三四五六七八九十首甲乙丙丁戊己庚辛壬癸]'
    romannum = (
        ',I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,').split(',')

    def __init__(self):
        super().__init__()
        self.nums = ['00', '00', '00']

    def roman(self, text):
        """Decode Roman numbers"""
        if '.' in text:
            text = text[:text.find('.')]
        return OutlineFilter.romannum.index(text) if text in OutlineFilter.romannum else 99

    def dechnum(self, text):
        """Decode Chinese numbers"""
        vals = [(OutlineFilter.chnum+_).find(_) for _ in text]
        if len(vals) == 1:
            if vals[0] > 10:
                return vals[0] - 11
            else:
                return vals[0]
        elif len(vals) == 2:
            if vals[0] == 10:
                return vals[0] + vals[1]
            else:
                return -1
        else:
            return vals[0]*vals[1]+vals[2]

    def check_outline(self, paragraph: Paragraph):
        """Check outline"""
        lang, content = paragraph.lang, paragraph. content
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
                    self.dechnum(content[:content.find('、')]))
            elif re.match('^第' + OutlineFilter.chnum + '+章', content):
                outline = 'chap {:02}'.format(
                    self.dechnum(content[1:content.find('章')]))
            elif re.match('^第' + OutlineFilter.chnum + '+節', content):
                outline = 'sect {:02}'.format(
                    self.dechnum(content[1:content.find('節')]))
            elif re.match('^第' + OutlineFilter.chnum + '卷', content):
                outline = 'book {:02}'.format(
                    self.dechnum(content[1:content.find('卷')]))
            elif re.match('^篇' + OutlineFilter.chnum, content):
                outline = 'chap {:02}'.format(
                    self.dechnum(content[1]))
            elif re.match('^部' + OutlineFilter.chnum, content):
                outline = 'book {:02}'.format(
                    self.dechnum(content[1]))

        return outline

    def resolve(self, paragraph: Paragraph):
        paragraph.content = paragraph.content.strip()
        if not paragraph.content:
            return paragraph

        outline = self.check_outline(paragraph)

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

        paragraph.outline = '.'.join(self.nums)
        paragraph.save()
        return paragraph


class ConditionalAssignment(PipelineStage):
    """按条件赋值字段"""

    def __init__(self, cond, field):
        """
        Args:
            cond (QUERY): 一行一个检查的条件，与最终要赋予的值之间用=>连缀
            field (str): 要赋值的字段
        """
        super().__init__()
        self.cond = [parser.eval(_) for _ in cond.split('\n')]
        self.field = field

    def resolve(self, paragraph: Paragraph):
        for cond, val in self.cond:
            if execute_query_expr(cond, paragraph):
                setattr(paragraph, self.field, val if not isinstance(val, str)
                        or not val.startswith('$') else getattr(paragraph, val[1:], None))
                break
        return paragraph


class KeywordsReplacement(PipelineStage):
    """替换关键词（标签）"""

    def __init__(self, from_tag, to_tag, arr='keywords'):
        """
        Args:
            from_tag (str): 原标签
            to_tag (str): 目标标签
            arr (str): 替换的数组字段（默认为标签）
        """
        super().__init__()
        self.from_tag = from_tag
        self.to_tag = to_tag
        self.arr = arr

    def resolve(self, paragraph: Paragraph):
        arr = paragraph[self.arr]
        if self.from_tag in arr:
            arr = list(arr)
            arr.remove(self.from_tag)
            arr.append(self.to_tag)
            paragraph[self.arr] = arr
        return paragraph


class MongoCollectionBatchOper(PipelineStage):
    """数据库批处理"""

    def __init__(self, mongocollection='', updates='[]'):
        """
        Args:
            mongocollection (str): 要处理的数据库
            updates (QUERY): 要执行的更新，应表示为一个列表，其中的每个元素为 (query; update)，
                如 (keywords=test; pull(keywords=key)) 。update 可为 set, pull, unset 等，也可使用聚合
        """
        super().__init__()
        self.collection = Paragraph.get_coll(mongocollection)
        updates = parser.eval('[];' + updates)
        self.updates = []

        def _assert(cond, info=''):
            assert cond, '更新格式不正确：' + repr(info)

        _assert(isinstance(updates, list))
        for tup in updates:
            _assert(len(tup) == 2, updates)
            query, update = tup
            if isinstance(update, dict):
                assert len(update) == 1, '更新格式不正确'
                for k in update:
                    assert k.startswith('$'), '更新格式不正确'
            elif isinstance(update, list):
                pass
            else:
                assert False, '更新格式不正确'
            self.updates.append((query, update))

    def summarize(self, _):
        for query, update in self.updates:
            self.collection.db.update_many(query, update)
        return True
