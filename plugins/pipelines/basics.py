"""基本操作"""

import json
import statistics
import string
from collections import defaultdict, deque
from io import BytesIO
from itertools import chain
from itertools import count as iter_count

import many_stop_words
import pandas
import regex as re
import textrank4zh

from jindai.pipeline import PipelineStage
from jindai.app import aeval, storage
from jindai.helpers import WordStemmer as _Stemmer
from jindai.helpers import jieba, safe_import
from jindai.models import Paragraph, Terms, get_db_session


class Passthrough(PipelineStage):
    """
    Passthrough
    @zhs 直接通过
    """


class FilterOut(PipelineStage):
    """
    Filter Out
    @zhs 截止当前处理的段落
    """

    def __init__(self, cond='true') -> None:
        """
        Arg:
            cond (QUERY): Condition
                @zhs 截止条件
        """
        self.cond = cond

    def resolve(self, paragraph) -> None:
        ee = aeval(self.cond, paragraph)
        if ee:
            return
        return paragraph


class TradToSimpChinese(PipelineStage):
    """
    Convert Traditional Chinese to Simplified Chinese
    @zhs 繁体中文转为简体中文
    @zht 繁體中文轉爲簡體中文
    """
    import opencc
    t2s = opencc.OpenCC('t2s')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = TradToSimpChinese.t2s.convert(paragraph.content)
        if paragraph.lang == 'zht':
            paragraph.lang = 'zhs'
        return paragraph


class LanguageDetect(PipelineStage):
    """Simple language detection
    @zhs 简易语言检测
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph.lang and paragraph.lang != 'auto':
            return paragraph

        if paragraph.content:
            paragraph.lang = self.detect(paragraph.content)
            if paragraph.lang in ('zh-cn', 'zh-sg'):
                paragraph.lang = 'zhs'
            elif paragraph.lang.startswith('zh-'):
                paragraph.lang = 'zht'
            elif '-' in paragraph.lang:
                paragraph.lang = paragraph.lang.split('-')[0]

        return paragraph

    def detect(self, sentence) -> str:
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
                return 'zhs'
            else:
                return 'zht'

        try:
            return langdetect.detect(sentence)
        except langdetect.lang_detect_exception.LangDetectException:
            return 'en'


class WordStemmer(PipelineStage):
    """
    Stemming words in tokens field
    @zhs 附加词干到 tokens 中（需要先进行切词）
    """

    def __init__(self, append=True, field='tokens') -> None:
        """
        Args:
            append (bool):
                Append to/overwrite tokens field
                @zhs 是添加到结尾还是覆盖
            field (str):
                Field name to store cutted words
                @zhs 保存到字段名
        """
        super().__init__()
        self.append = append
        self.field = field
        self.stemmer = _Stemmer()
        
    def stem_words(self, lang, words):
        return self.stemmer.stem_tokens(lang, words)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        tokens = self.stem_words(paragraph.lang, paragraph[self.field])
        if self.append:
            paragraph[self.field] += tokens
        else:
            paragraph[self.field] = tokens
        return paragraph


class LatinTransliterate(PipelineStage):
    """
    Transliterate tokens
    @zhs 转写为拉丁字母的单词（需要先进行切词）
    """

    def __init__(self, append=True, field='tokens') -> None:
        """
        Args:
            append (bool):
                Append to/overwrite tokens field
                @zhs 是添加到结尾还是覆盖
            field (str):
                Field name to store cutted words
                @zhs 保存到字段名
        """
        super().__init__()
        self.append = append
        self.field = field
        transliterate = safe_import('transliterate')
        self.supported_languages = transliterate.get_available_language_codes()
        self.translit = transliterate.translit
        
    def transliterate(self, lang, words) -> list:
        if lang in self.supported_languages:
            return [self.translit(
                word, lang, reversed=True).lower() for word in words]
        else:
            return []

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if tokens := self.translit(paragraph.lang, paragraph[self.field] or []):
            if self.append:
                paragraph[self.field] += tokens
            else:
                paragraph[self.field] = tokens
        return paragraph


class WordCut(PipelineStage):
    """
    Multilingual word cutting
    @zhs 多语种分词
    """

    t2s = safe_import('opencc', 'opencc-python-reimplemented').OpenCC('t2s')
    kks = safe_import('pykakasi').kakasi()
    
    def __init__(self, for_search=False, field='keywords', **_) -> None:
        """
        Args:
            for_search (bool): 
                Append redundant word-cutting results or stemming/transliteration
                @zhs 是否用于搜索（添加冗余分词结果或词干/转写）
            field (str):
                Field name to store cutted words
                @zhs 保存到字段名
        """
        super().__init__()
        self.for_search = for_search
        self.field = field
        self.stmr = WordStemmer(append=True, field=field)
        self.trlit = LatinTransliterate(append=True, field=field)

    @staticmethod
    def remove_accents(input_str) -> str:
        unicodedata = safe_import('unicodedata')
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    def resolve(self, paragraph: Paragraph) -> Paragraph:
        words = []
        if paragraph.lang == 'zht':
            content = WordCut.t2s.convert(paragraph.content)
        else:
            content = paragraph.content

        if paragraph.lang in ('zhs', 'zht'):
            words = list(jieba.cut_query(content) if self.for_search else jieba.cut_text(content))
        elif paragraph.lang == 'ja':
            words = list(set(content))
            for i in WordCut.kks.convert(content):
                words.append(i['orig'])
                if self.for_search:
                    words.append(i['hepburn'])
        else:
            words = [_.lower().strip().strip(string.punctuation)
                     for _ in re.split(r'[^\w]', content)]
            if self.for_search:
                words += self.stmr.stem_words(paragraph.lang, words)
            words += [WordCut.remove_accents(word) for word in words]

        if self.for_search:
            words += self.trlit.transliterate(paragraph.lang, words)
        
        if not paragraph[self.field]:
            paragraph[self.field] = []
        paragraph[self.field] += words
        return paragraph


class FilterPunctuations(PipelineStage):
    """
    Filter punctuations
    @zhs 过滤标点符号
    """

    re_punctuations = re.compile(
        r'[，。「」·；□■•●『』［］【】（）\s\(\)、“”‘’《》——\-！？\.\?\!\,\'\"：\/\\\n\u3000…]')

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.content = FilterPunctuations.re_punctuations.sub(
            '', paragraph.content.strip(string.punctuation))
        return paragraph
    

class Reparagraph(PipelineStage):
    """
    Reparagraphize
    @zhs 重新分段"""

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        lang = paragraph.lang
        lines = paragraph.content.split('\n')

        def paragraph_finished(text):
            return text.endswith(tuple('.!?…\"。！？…—：”）'))

        def merge_lines():
            lens = [len(_) for _ in lines]
            if len(lens) < 3:
                yield ('' if lang[:2] in ('zh', 'ja') else ' ').join(lines)
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

                if lang[:2] != 'zh':
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
            data['content'] = text
            yield Paragraph.from_dict(data)


from typing import Iterator


class SplitParagraph(PipelineStage):
    """
    Split paragraphs
    @zhs 拆分语段
    """

    def __init__(self, delimeter='\n') -> None:
        """
        Args:
            delimeter (str):
                Delimeter
                @zhs 拆分的分隔符
        """
        super().__init__()
        self.delimeter = delimeter

    def resolve(self, paragraph: Paragraph) -> Iterator:
        for content in paragraph.content.split(self.delimeter):
            if content:
                new_paragraph = Paragraph.from_dict(paragraph.as_dict())
                new_paragraph.content = content
                yield new_paragraph


class AccumulateParagraphs(PipelineStage):
    """
    Accumulate all paragraphs iterated
    @zhs 聚集段落遍历结果
    """

    def __init__(self, sort='') -> None:
        """
        Args:
            sort (str): Sort by field name
                @zhs 排序字段名
        """
        super().__init__()
        self.paragraphs = deque()
        self.sort = [_.strip() for _ in sort.split(',') if _]

    def resolve(self, paragraph: Paragraph) -> None:
        self.paragraphs.append(paragraph.as_dict())
        
    def sorter(self, obj) -> list:
        class _Rev:
            def __init__(self, val):
                self.val = val
            
            def __lt__(self, other):
                return self.val > other.val
                        
        def _rev(val, reversed):
            if reversed:
                if isinstance(val, (int, float)):
                    return -val
                return _Rev(val)
            return val
                    
        return [_rev(obj.get(k.strip('-'), ''), k.startswith('-')) for k in self.sort]

    def summarize(self, _) -> list:
        results = list(self.paragraphs)
        if self.sort:
            results = sorted(results, key=self.sorter)
        return results


class Export(PipelineStage):
    """
    Export accumulated result to file
    @zhs 结果导出为文件
    """

    def __init__(self, output_format='xlsx', limit=0) -> None:
        """导出结果

        Args:
            output_format (xlsx|json|csv):
                Export file foramt
                @zhs 输出格式
            limit (int, optional):
                Max export records count, 0 for no limit
                @zhs 最多导出的记录数量，0表示无限制。
        """
        super().__init__()
        self.extension = output_format
        self.format = {'xlsx': 'excel'}.get(output_format, output_format)
        self.limit = limit

    def summarize(self, result) -> dict:
        
        def json_dump(val):
            return json.dumps(val)

        def str_repr(val, strip_brackets=False):
            if isinstance(val, str):
                return val
            elif isinstance(val, list):
                val = ', '.join(map(str_repr, val))
                if not strip_brackets:
                    val = f'[{val}]'
                return val
            elif isinstance(val, dict):
                val = ', '.join([f'{k}={str_repr(v)}' for k, v in val.items()])
                if not strip_brackets:
                    val = f'({val})'
                return val
            elif isinstance(val, bytes):
                return f'<... {len(val)} bytes>'
            else:
                return str(val)

        def flattern_dict(val):
            result = {}
            for k, v in val.items():
                if isinstance(v, dict):
                    v = flattern_dict(v)
                    for kp, vp in v.items():
                        result[f'{k}.{kp}'] = vp
                else:
                    result[k] = v
            return result

        result = [{k: str_repr(v, True) for k, v in (
            flattern_dict(r if isinstance(r, dict) else r.as_dict())).items()} for r in result]

        if self.format == 'json':
            return PipelineStage.return_file('json', json_dump(result).encode('utf-8'))

        elif self.format in ('csv', 'excel'):
            buf = BytesIO()
            getattr(pandas.DataFrame(result), f'to_{self.format}')(buf)
            return PipelineStage.return_file(self.extension, buf.getvalue())


class AutoSummary(PipelineStage):
    """
    Auto summary for Chinese texts
    @zhs 中文自动摘要
    """

    def __init__(self, count) -> None:
        """
        Args:
            count (int):
                Sentences count
                @zhs 摘要中的句子数量
        """
        super().__init__()
        self.count = count

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        tr4s = textrank4zh.TextRank4Sentence()
        tr4s.analyze(text=paragraph.content, lower=True, source='all_filters')
        paragraph.summary = '\n'.join([
            item.sentence
            for item in tr4s.get_key_sentences(num=self.count)
        ])
        return paragraph


class ArrayField(PipelineStage):
    """
    Manipulate array field
    @zhs 操作数组字段
    """

    def __init__(self, field, push=True, elements='') -> None:
        """
        Args:
            field (str): Field name
                @zhs 字段名
            push (bool): push or delete
                @zhs 添加或删除
            elements (LINES):
                Element to push or delete, use $<field> or constants
                @zhs 添加或删除的元素，每行一个 $ 开头的字段名或常量
        """
        super().__init__()
        self.field = field
        self.elements = PipelineStage.parse_lines(elements)
        self.push = push

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph[self.field] is None and self.push:
            paragraph[self.field] = []
        for ele in self.elements:
            ele = aeval(ele, paragraph)
            if self.push:
                paragraph[self.field].append(ele)
            else:
                if ele in paragraph[self.field]:
                    paragraph[self.field].remove(ele)
        return paragraph


class ArrayAggregation(PipelineStage):
    """
    Concat arrays in an array field
    @zhs 减少一层数组嵌套层级"""

    def __init__(self, field, new_field='') -> None:
        """
        Args:
            field (str): Field name
                @zhs 字段名
            new_field (str): New field name, blank for replacement
                @zhs 新的字段名，留空表示替换原数组字段
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

    def as_dict(self) -> dict:
        """Get the dictionary representation of the counter"""
        return {
            k: v.value() for k, v in self._d.items()
        }


from typing import Dict


class NgramCounter(PipelineStage):
    """N-Gram
    """

    def __init__(self, n: int, lr=False) -> None:
        """ N-Gram

        Args:
            n (int): Max string lenght
                @zhs 最大字串长度
            lr (bool): Count left/right characters
                @zhs 是否同时记录左右字符计数
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

    def summarize(self, _) -> Dict:
        self.ngrams = self.ngrams.as_dict()
        self.ngrams_lefts = {k: v.as_dict()
                             for k, v in self.ngrams_lefts.items()}
        self.ngrams_rights = {k: v.as_dict()
                              for k, v in self.ngrams_rights.items()}


class Limit(PipelineStage):
    """
    Limit results count
    @zhs 限制返回的结果数量
    """

    def __init__(self, limit) -> None:
        """
        Args:
            limit (int):
                Max results count
                @zhs 要返回的最大结果数量，0则不返回
        """
        super().__init__()
        self.limit = limit
        self.counter = iter_count()

    def resolve(self, paragraph: Paragraph):
        val = next(self.counter)
        if val < self.limit:
            return paragraph


class RegexReplace(PipelineStage):
    """
    Replace with regular expression
    @zhs 正则表达式匹配并替换
    """

    def __init__(self, pattern, replacement='', plain=False) -> None:
        """
        Args:
            pattern (str):
                Regular expression
                @zhs 正则表达式
            replacement (str):
                Replacement string
                @zhs 要替换成的字符串
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
    @zhs 正则表达式匹配并提取到字段中
    """

    def __init__(self, pattern, target, source='content', match='{0}',
                 continuous=False, filter_out=False) -> None:
        """
        Args:
            pattern (str): Regular expression
                @zhs 正则表达式
            source (str): String to match, default to content
                @zhs 匹配的字段，默认为内容
            target (str): Field name to fill in
                @zhs 要提取入的字段名称
            match (str): Fill the target with, in forms like '{1}{2}', default to
                all matched text
                @zhs 填入的值，如 '{1}{2}' 等形式，默认为全部匹配到的文本
            filter_out (bool): Filter out unmatched paragraphs
                @zhs 过滤未匹配到的语段
            continuous (bool): Use previous matched value for unmatched paragraph
                @zhs 未匹配到的语段自动使用上次的值
        """
        super().__init__()
        self.regexp = re.compile(pattern)
        self.source = source
        self.target = target
        self.value = ''
        self.match = match
        self.continuous = continuous
        self.filter_out = filter_out

    def resolve(self, paragraph: Paragraph) -> Paragraph | None:
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


class RegexMatches(PipelineStage):
    """
    Regex Match
    @zhs 正则表达式匹配
    """

    def __init__(self, regex, field='content') -> None:
        """
        Args:
            field (str): Field name to match from
                @zhs 要匹配的字段
            regex (str): Regular Expression
                @zhs 正则表达式
        """
        super().__init__()
        self.field = field
        self.regex = regex

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        for m in re.findall(self.regex, str(getattr(paragraph, self.field))):
            paragraph.extdata[self.field] = m
            yield paragraph


class FieldAssignment(PipelineStage):
    """
    Assign value/field to another field
    @zhs 将某一个字段的值或输入值保存到另一个字段
    """

    def __init__(self, field, value='', delete_field=False) -> None:
        """
        Args:
            field (str): Field name
                @zhs 新的字段名
            value (QUERY):
                $<field> or contants
                @zhs 以 $ 开头的字段名，或常数值（类型将自动匹配）
            delete_field (bool):
                Delete the field
                @zhs 删除该字段
        """
        super().__init__()
        self.field = field
        self.value = value
        self.delete_field = delete_field

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if self.delete_field:
            del paragraph.extdata[self.field]
        else:
            paragraph[self.field] = aeval(self.value, paragraph)
        return paragraph


from typing import Dict


class DeleteParagraph(PipelineStage):
    """Delete Paragraph from Database
    @zhs 从数据库删除段落
    """

    async def resolve(self, paragraph: Paragraph) -> None:
        if paragraph.id:
            async with get_db_session() as session:
                await session.delete(paragraph)
            

class SaveParagraph(PipelineStage):
    """Save
    @zhs 保存
    """

    def __init__(self) -> None:
        '''
        '''
        super().__init__()

    async def resolve(self, paragraph: Paragraph):
        if not paragraph.id:
            async with get_db_session() as session:
                session.add(paragraph)
                await Terms.store(paragraph.keywords)
        return paragraph
    

class FieldIncresement(PipelineStage):
    """
    Increment on field
    @zhs 对字段进行自增操作
    """

    def __init__(self, field, inc_value) -> None:
        '''
        Args:
            field (str): Field name
                @zhs 字段名称
            inc_value (str): Increment by, or $<field>
                @zhs 自增的值，或以 $ 开头的另一字段名
        '''
        super().__init__()
        self.field = field
        self.inc_value = inc_value

    def resolve(self, paragraph: Paragraph):
        val = getattr(paragraph, self.field)
        val += aeval(self.inc_value, paragraph)
        setattr(paragraph, val)
        return paragraph


class OutlineFilter(PipelineStage):
    """
    Identify Chinese/Roman ordinal numbers for outline
    @zhs 中英文大纲序号识别
    """

    chnum = '[一二三四五六七八九十首甲乙丙丁戊己庚辛壬癸]'
    romannum = (
        ',I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,').split(',')

    def __init__(self) -> None:
        super().__init__()
        self.nums = ['00', '00', '00']

    def roman(self, text) -> int:
        """Decode Roman numbers"""
        if '.' in text:
            text = text[:text.find('.')]
        return OutlineFilter.romannum.index(text) if text in OutlineFilter.romannum else 99

    def dechnum(self, text) -> int:
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

    def check_outline(self, paragraph: Paragraph) -> str:
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
            # self.log(content[:20], outline)
            if outline.startswith('book '):
                nnums = [outline[5:], '00', '00']
            elif outline.startswith('chap '):
                nnums = [self.nums[0], outline[5:], '00']
            else:
                nnums = [self.nums[0], self.nums[1], outline[5:]]
            if '.'.join(nnums) > '.'.join(self.nums):
                self.nums = nnums

        paragraph.outline = '.'.join(self.nums)
        try_commit()
        return paragraph


class ConditionalAssignment(PipelineStage):
    """
    Conditional assignment
    @zhs 按条件赋值字段"""

    def __init__(self, cond, field) -> None:
        """
        Args:
            cond (QUERY): Conditions and assignment values, in forms like:
                <condition> => <constant>;
                @zhs 检查的条件，与最终要赋予的值之间用 => 连缀，条件之间用 ; 连接
            field (str): Target field name
                @zhs 要赋值的字段
        """
        super().__init__()
        self.conds = aeval(cond)
        self.field = field

    def resolve(self, paragraph: Paragraph):
        for cond, val in self.conds:
            if aeval(cond, paragraph):
                setattr(paragraph, self.field,
                        aeval(val, paragraph))
                break
        return paragraph


class KeywordsReplacement(PipelineStage):
    """Replace keywords/tags
    @zhs 替换关键词（标签）"""

    def __init__(self, from_tag, to_tag, arr='keywords') -> None:
        """
        Args:
            from_tag (str): Original keyword
                @zhs 原标签
            to_tag (str): Target keyword
                @zhs 目标标签
            arr (str): Target field, default to keywords
                @zhs 替换的数组字段（默认为标签）
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


class PDFUnlock(PipelineStage):
    """Unlock "secured" PDF
    @zhs 解锁读保护的 PDF
    """

    def __init__(self, file: bytes) -> None:
        """
        :param file: File binary
            @zhs 文件 data: URL
        :type file: file:pdf
        """
        super().__init__()
        import pike
        buf = BytesIO()
        pike.open(storage.open(file, 'rb')).save(buf)
        self.data = PipelineStage.return_file('pdf', buf.getvalue())

    def summarize(self, _) -> dict:
        return self.data


from typing import Dict


class SetNamedResult(Passthrough):
    """Set name for summarization result
    @zhs 为总结阶段的结果设置名称
    """

    def __init__(self, name: str) -> None:
        """
        :param name: Name
            @zhs 名称
        :type name: str
        """
        super().__init__()
        self.name = name

    def summarize(self, result) -> Dict:
        self.gctx[self.name] = result
        return result


class LoadNamedResult(Passthrough):
    """Get named result
    @zhs 读取已命名的结果
    """

    def __init__(self, name: str) -> None:
        """
        :param name: Name
            @zhs 名称
        :type name: str
        """
        super().__init__()
        self.name = name

    def summarize(self, _) -> dict:
        if self.name == '':
            return self.gctx
        return self.gctx.get(self.name)


class FilterStopWords(PipelineStage):
    """Filter stop words
    @zhs 过滤停用词
    """
    
    _lang_stopwords = {
        l: many_stop_words.get_stop_words(l)
        for l in ['en', 'fr', 'de', 'ru', 'ja', 'zh']
    }
    
    _punctuations = re.compile(r'^[\u3000-\u303F\uFF00-\uFFEF\"\'{}()\[\]\\*&.?!,…:;@#!]$')
    
    @staticmethod
    def get(lang) -> list:
        if lang == 'chs':
            return FilterStopWords._lang_stopwords['zh']
        elif lang in FilterStopWords._lang_stopwords:
            return FilterStopWords._lang_stopwords[lang]
        else:
            return []

    def __init__(self, stopwords='') -> None:
        """
        Args:
            stopwords (str): 额外的停用词表，用空格分割
        """
        self.stopwords = set(stopwords.split())
        super().__init__()
    
    def resolve(self, paragraph):
        paragraph.keywords = [
            _ for _ in paragraph.keywords
            if _ not in self.stopwords and \
                _ not in FilterStopWords.get(paragraph.lang) and \
                not FilterStopWords._punctuations.match(_)
        ]
        return paragraph