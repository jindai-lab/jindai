import re, os
import json, yaml
import itertools
from bson import ObjectId
from models import Paragraph
from pipeline import Pipeline, PipelineStage
from PyMongoWrapper import QueryExprParser
from .utils import execute_query_expr

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
            self.value = match.group(0)
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
        self._parser = QueryExprParser()
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
            self.valueLiteral = self._parser.expand_literals(value)
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
            cond (str): 条件式，用 iter 表示被判断的项目，或用省略形式
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

    def __init__(self, mongocollection=''):
        '''
        Args:
            mongocollection (str): 数据库目标数据集名称
        '''
        self.convert = lambda x: x
        if mongocollection:
            class TempParagraph(Paragraph):
                _collection = mongocollection
                
            self.convert = lambda x: TempParagraph(**x.as_dict())

    def resolve(self, p : Paragraph):
        self.convert(p).save()
        return p


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
            self.inc_value = expand_literal(inc_value)
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
            # print(content[:20], outline)
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
