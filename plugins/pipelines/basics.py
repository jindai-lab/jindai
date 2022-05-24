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
from jindai import PipelineStage, parser, safe_open
from jindai.helpers import execute_query_expr, language_iso639, safe_import, WordStemmer as _Stemmer
from jindai.models import Dataset, Paragraph, db


class Passthrough(PipelineStage):
    """
    Passthrough
    @chs 直接通过
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        return paragraph


class TradToSimpChinese(PipelineStage):
    """
    Convert Traditional Chinese to Simplified Chinese
    @chs 繁体中文转为简体中文
    @cht 繁體中文轉爲簡體中文
    """

    t2s = safe_import('opencc', 'opencc-python-reimplementation').OpenCC('t2s')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = TradToSimpChinese.t2s.convert(paragraph.content)
        if paragraph.lang == 'cht':
            paragraph.lang = 'chs'
        return paragraph


class LanguageDetect(PipelineStage):
    """Simple language detection
    @chs 简易语言检测
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
        """Detect language"""

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

        try:
            return langdetect.detect(sentence)
        except langdetect.lang_detect_exception.LangDetectException:
            return 'en'


class WordStemmer(PipelineStage):
    """
    Stemming words in tokens field
    @chs 附加词干到 tokens 中（需要先进行切词）
    """

    def __init__(self, append=True):
        """
        Args:
            append (bool):
                Append to/overwrite tokens field
                @chs 是添加到结尾还是覆盖
        """
        super().__init__()
        self.append = append
        self.stemmer = _Stemmer()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        tokens = self.stemmer.stem_tokens(paragraph.lang, paragraph.tokens)
        if self.append:
            paragraph.tokens += tokens
        else:
            paragraph.tokens = tokens
        return paragraph


class LatinTransliterate(PipelineStage):
    """
    Transliterate tokens
    @chs 转写为拉丁字母的单词（需要先进行切词）
    """

    def __init__(self, append=True):
        """
        Args:
            append (bool):
                Append to/overwrite tokens field
                @chs 是添加到结尾还是覆盖
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
    """
    Multilingual word cutting
    @chs 多语种分词
    """

    t2s = safe_import('opencc', 'opencc-python-reimplementation').OpenCC('t2s')
    kks = safe_import('pykakasi').kakasi()
    jieba = safe_import('jieba')
    stmr = WordStemmer(append=True)
    trlit = LatinTransliterate(append=True)

    def __init__(self, for_search=False, **_):
        """
        Args:
            for_search (bool): 
                Append redundant word-cutting results or stemming/transliteration
                @chs 是否用于搜索（添加冗余分词结果或词干/转写）
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
    """
    Set tokens as keywords and unset tokens field
    @chs 将检索词设为分词结果并删除词串字段
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.keywords = list(set(paragraph.tokens))
        del paragraph.tokens
        paragraph.save()
        return paragraph


class FilterPunctuations(PipelineStage):
    """
    Filter punctuations
    @chs 过滤标点符号
    """

    re_punctuations = re.compile(
        r'[，。「」·；□■•●『』［］【】（）\s\(\)、“”‘’《》——\-！？\.\?\!\,\'\"：\/\\\n\u3000…]')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = FilterPunctuations.re_punctuations.sub(
            '', paragraph.content)
        return paragraph


class Reparagraph(PipelineStage):
    """
    Reparagraphize
    @chs 重新分段"""

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
            yield type(paragraph)(content=text, **data)


class SplitParagraph(PipelineStage):
    """
    Split paragraphs
    @chs 拆分语段
    """

    def __init__(self, delimeter='\n'):
        """
        Args:
            delimeter (str):
                Delimeter
                @chs 拆分的分隔符
        """
        super().__init__()
        self.delimeter = delimeter

    def resolve(self, paragraph: Paragraph):
        for content in paragraph.content.split(self.delimeter):
            if content:
                new_paragraph = type(paragraph)(paragraph)
                new_paragraph.content = content
                yield new_paragraph


class AccumulateParagraphs(PipelineStage):
    """
    Accumulate all paragraphs iterated
    @chs 聚集段落遍历结果
    """

    def __init__(self):
        super().__init__()
        self.paragraphs = deque()

    def resolve(self, paragraph: Paragraph):
        self.paragraphs.append(paragraph)

    def summarize(self, *_):
        return list(self.paragraphs)


class Export(PipelineStage):
    """
    Export accumulative result to file
    @chs 结果导出为文件
    """

    def __init__(self, output_format='xlsx', limit=0) -> None:
        """导出结果

        Args:
            output_format (xlsx|json|csv):
                Export file foramt
                @chs 输出格式
            limit (int, optional):
                Max export records count, 0 for no limit
                @chs 最多导出的记录数量，0表示无限制。
        """
        super().__init__()
        self.format = output_format
        self.limit = limit

    def summarize(self, result):
        safe_import('xlsxwriter')  # as required for pandas to export xlsx file
        pandas = safe_import('pandas')

        def json_dump(val):
            try:
                return json.dumps(val)
            except Exception:
                return str(val)

        result = [_.as_dict() if isinstance(
            _, DbObject) else _ for _ in result]

        if self.format == 'json':
            return PipelineStage.return_file('json', json_dump(result).encode('utf-8'))

        elif self.format in ('csv', 'xlsx'):
            buf = BytesIO()
            getattr(pandas.DataFrame(result), f'to_{self.format}')(buf)
            return PipelineStage.return_file(self.format, buf.getvalue())


class AutoSummary(PipelineStage):
    """
    Auto summary for Chinese texts
    @chs 中文自动摘要
    """

    def __init__(self, count) -> None:
        """
        Args:
            count (int):
                Sentences count
                @chs 摘要中的句子数量
        """
        super().__init__()
        self.count = count

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        # textrank4zh is a package for Chinese text ranking
        tr4s = safe_import('textrank4zh').TextRank4Sentence()
        tr4s.analyze(text=paragraph.content, lower=True, source='all_filters')
        paragraph.summary = '\n'.join([
            item.sentence
            for item in tr4s.get_key_sentences(num=self.count)
        ])
        return paragraph


class ArrayField(PipelineStage):
    """
    Manipulate array field
    @chs 操作数组字段
    """

    def __init__(self, field, push=True, elements='') -> None:
        """
        Args:
            field (str): Field name
                @chs 字段名
            push (bool): push or delete
                @chs 添加或删除
            elements (str):
                Element to push or delete, use $<field> or constants
                @chs 添加或删除的元素，每行一个 $ 开头的字段名或常量
        """
        super().__init__()
        self.field = field
        self.elements = []
        try:
            elements = parser.parse(elements)
            assert isinstance(elements, list)
            self.elements = elements
        except Exception:
            for ele in elements.split('\n'):
                ele = parser.parse(ele)
                self.elements.append(ele)
        self.push = push

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph[self.field] is None and self.push:
            paragraph[self.field] = []
        if not isinstance(paragraph[self.field], (list, DbObjectCollection)):
            return paragraph
        for ele in self.elements:
            ele = execute_query_expr(ele, paragraph)
            if self.push:
                paragraph[self.field].append(ele)
            else:
                if ele in paragraph[self.field]:
                    paragraph[self.field].remove(ele)
        return paragraph


class ArrayAggregation(PipelineStage):
    """
    Concat arrays in an array field
    @chs 减少一层数组嵌套层级"""

    def __init__(self, field, new_field='') -> None:
        """
        Args:
            field (str): Field name
                @chs 字段名
            new_field (str): New field name, blank for replacement
                @chs 新的字段名，留空表示替换原数组字段
        """
        super().__init__()
        self.field = field
        self.new_field = new_field or field

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        setattr(paragraph, self.new_field, list(chain(*paragraph[self.field])))
        return paragraph


class Counter:
    """Thread-safe counter"""

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
    """N-Gram
    """

    def __init__(self, n: int, lr=False):
        """ N-Gram

        Args:
            n (int): Max string lenght
                @chs 最大字串长度
            lr (bool): Count left/right characters
                @chs 是否同时记录左右字符计数
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
    """
    Limit results count
    @chs 限制返回的结果数量
    """

    def __init__(self, limit):
        """
        Args:
            limit (int):
                Max results count
                @chs 要返回的最大结果数量，0则不返回
        """
        super().__init__()
        self.limit = limit
        self.counter = iter_count()

    def resolve(self, paragraph: Paragraph):
        val = next(self.counter)
        if val < self.limit:
            return paragraph


class FilterDuplication(PipelineStage):
    """
    Filter duplications in specified database collection
    @chs 过滤已经存储在指定数据库中的段落
    """

    def __init__(self, field, mongocollection='paragraph') -> None:
        """
        Args:
            mongocollection (str): Database collection name
                @chs 数据库集合名
            field (str): Field that mark an duplication
                @chs 要去重的字段值
        """
        super().__init__()
        self.mongocollection = mongocollection or 'paragraph'
        self.field = field

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        for _ in db[self.mongocollection].find({self.field: getattr(paragraph, self.field)}):
            return
        return paragraph


class RegexReplace(PipelineStage):
    """
    Replace with regular expression
    @chs 正则表达式匹配并替换
    """

    def __init__(self, pattern, replacement='', plain=False):
        """
        Args:
            pattern (str):
                Regular expression
                @chs 正则表达式
            replacement (str):
                Replacement string
                @chs 要替换成的字符串
        """
        super().__init__()
        if plain:
            pattern = re.escape(pattern)
        self.regexp = re.compile(pattern)
        self.replacement = replacement

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = self.regexp.sub(
            self.replacement, paragraph.content)
        return paragraph


class RegexFilter(PipelineStage):
    """
    Match regular expression and extract result to field
    @chs 正则表达式匹配并提取到字段中
    """

    def __init__(self, pattern, target, source='content', match='{0}',
                 continuous=False, filter_out=False):
        """
        Args:
            pattern (str): Regular expression
                @chs 正则表达式
            source (str): String to match, default to content
                @chs 匹配的字段，默认为内容
            target (str): Field name to fill in
                @chs 要提取入的字段名称
            match (str): Fill the target with, in forms like '{1}{2}', default to
                all matched text
                @chs 填入的值，如 '{1}{2}' 等形式，默认为全部匹配到的文本
            filter_out (bool): Filter out unmatched paragraphs
                @chs 过滤未匹配到的语段
            continuous (bool): Use previous matched value for unmatched paragraph
                @chs 未匹配到的语段自动使用上次的值
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
    """
    Assign value/field to another field
    @chs 将某一个字段的值或输入值保存到另一个字段
    """

    def __init__(self, field, value):
        """
        Args:
            field (str): Field name
                @chs 新的字段名
            value (QUERY):
                $<field> or contants
                @chs 以 $ 开头的字段名，或常数值（类型将自动匹配）
        """
        super().__init__()
        self.field = field
        self.value = value

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        setattr(paragraph, self.field, execute_query_expr(self.value, paragraph))
        return paragraph


class FilterArrayField(PipelineStage):
    """Filter array field
    @chs 过滤列表字段的值
    """

    def __init__(self, field, cond) -> None:
        """
        Args:
            field (str): Field name
                @chs 字段名称
            cond (QUERY): Conditional expression, use `iter` for the iterated item,
                or use abbreviated form like '>0' to mean 'iter>0'
                @chs 条件式，用 iter 表示被判断的项目，或用省略形式
        """
        super().__init__()
        self.field = field
        self.cond = QueryExprParser(allow_spacing=True, abbrev_prefixes={
                                    None: 'iter='}).parse(cond)

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


class DeleteParagraph(PipelineStage):
    """Delete Paragraph from Database
    @chs 从数据库删除段落
    """

    def resolve(self, paragraph: Paragraph):
        if paragraph.id:
            paragraph.delete()
        return  # no yielding paragraph anymore


class SaveParagraph(PipelineStage):
    """Save
    @chs 保存
    """

    def __init__(self, mongocollection=''):
        '''
        Args:
            mongocollection (str): Database collection name
                @chs 数据库集合名
        '''
        super().__init__()
        self.mongocollection = mongocollection
        self.datasets = {}
        self.convert = Paragraph.get_converter(
            mongocollection) if mongocollection else lambda x: x

    def resolve(self, paragraph: Paragraph):
        self.convert(paragraph).save()
        if paragraph.dataset and paragraph.dataset not in self.datasets:
            self.datasets[paragraph.dataset] = {
                'mongocollection': getattr(paragraph, '_collection', ''),
                'sources': set()
            }
        if 'file' in paragraph.source and paragraph.dataset:
            self.datasets[paragraph.dataset]['sources'].add(
                paragraph.source['file'])
        return paragraph

    def summarize(self, _):
        self.logger('datasets count:', len(self.datasets))
        for name, data in self.datasets.items():
            coll = Dataset.first(F.name == name) \
                or Dataset(name=name, sources=[],
                           mongocollection=data['mongocollection'],
                           order_weight=999)
            for source in data['sources']:
                if source not in coll.sources:
                    coll.sources.append(source)
            coll.save()


class FieldIncresement(PipelineStage):
    """
    Increment on field
    @chs 对字段进行自增操作
    """

    def __init__(self, field, inc_value):
        '''
        Args:
            field (str): Field name
                @chs 字段名称
            inc_value (str): Increment by, or $<field>
                @chs 自增的值，或以 $ 开头的另一字段名
        '''
        super().__init__()
        self.field = field
        self.inc_value = parser.parse(inc_value)

    def resolve(self, paragraph: Paragraph):
        val = getattr(paragraph, self.field)
        val += execute_query_expr(self.inc_value, paragraph)
        setattr(paragraph, val)
        return paragraph


class OutlineFilter(PipelineStage):
    """
    Identify Chinese/Roman ordinal numbers for outline
    @chs 中英文大纲序号识别
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
    """
    Conditional assignment
    @chs 按条件赋值字段"""

    def __init__(self, cond, field):
        """
        Args:
            cond (QUERY): Conditions and assignment values, in forms like:
                (<condition> => <constant>);
                @chs 检查的条件，与最终要赋予的值之间用 => 连缀，条件之间用 ; 连接
            field (str): Target field name
                @chs 要赋值的字段
        """
        super().__init__()
        self.cond = parser.parse('[];' + cond)
        self.field = field

    def resolve(self, paragraph: Paragraph):
        for cond, val in self.cond:
            if execute_query_expr(cond, paragraph):
                setattr(paragraph, self.field,
                        execute_query_expr(val, paragraph))
                break
        return paragraph


class KeywordsReplacement(PipelineStage):
    """Replace keywords/tags
    @chs 替换关键词（标签）"""

    def __init__(self, from_tag, to_tag, arr='keywords'):
        """
        Args:
            from_tag (str): Original keyword
                @chs 原标签
            to_tag (str): Target keyword
                @chs 目标标签
            arr (str): Target field, default to keywords
                @chs 替换的数组字段（默认为标签）
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
    """Batch operation on database collection
    @chs 数据库批处理"""

    def __init__(self, mongocollection='', updates='[]'):
        """
        Args:
            mongocollection (str):
                Database collection
                @chs 要处理的数据库
            updates (QUERY):
                Updates to perform, in the form of a list like [(<query>; <update>); ...]
                where <update> can be function calls to set, pull, unset, etc.
                @chs 要执行的更新，应表示为一个列表，其中的每个元素为 (query; update)，
                @chs 如 (keywords=test; pull(keywords=key)) 。update 可为 set, pull, unset 等，也可使用聚合
        """
        super().__init__()
        self.collection = Paragraph.get_coll(mongocollection)
        updates = parser.parse('[];' + updates)
        self.updates = []

        def _assert(cond, info=''):
            assert cond, 'Wrong format for update' + repr(info)

        _assert(isinstance(updates, list))
        for tup in updates:
            _assert(len(tup) == 2, updates)
            query, update = tup
            if isinstance(update, dict):
                assert len(update) == 1, 'Wrong format for update'
                for k in update:
                    assert k.startswith('$'), 'Wrong format for update'
            elif isinstance(update, list):
                pass
            else:
                assert False, 'Wrong format for update'
            self.updates.append((query, update))

    def summarize(self, _):
        for query, update in self.updates:
            self.collection.db.update_many(query, update)
        return True


class PDFUnlock(PipelineStage):
    """Unlock "secured" PDF
    @chs 解锁读保护的 PDF
    """

    def __init__(self, file: bytes):
        """
        :param file: File binary
            @chs 文件 data: URL
        :type file: file:pdf
        """
        super().__init__()
        pike = safe_import('pikepdf')
        buf = BytesIO()
        pike.open(safe_open(file, 'rb')).save(buf)
        self.data = PipelineStage.return_file('pdf', buf.getvalue())

    def summarize(self, _):
        return self.data
