"""来自数据库
"""
import jieba
import re
from models import Paragraph, parser
from datasource import DataSource


class DBQueryDataSource(DataSource):
    """从系统自身的数据库进行查询
    """

    def __init__(self, query, mongocollection='', req={}, limit=0, skip=0, sort='', raw=False):
        """
        Args:
            query (str): 查询字符串，或以 ? 开头的查询表达式，或以 ?? 开头的聚合查询表达式
            sort (str): 排序表达式
            limit (int): 查询最多返回的结果数量，默认为0即无限制
            skip (int): 返回从第%1个开始的结果
            mongocollection (str): 数据库中其他数据集的名称
            raw (bool): 若为 False（默认值）则返回 Paragraph 对象，否则返回原始数据，仅对于聚合查询有效
        """
        super().__init__()
        self.raw = raw
        self.mongocollection = mongocollection
        if query.startswith('??'):
            self.aggregation = True
            self.querystr = query[2:]
        else:
            self.aggregation = False
            if query.startswith('?'):
                query = query[1:]
            elif re.search(r'[\,\=\|\%:]', query):
                pass
            else:
                query = ','.join([f'`{_.strip().lower()}`' for _ in jieba.cut(query) if _.strip()])
            self.querystr = query

        self.query = parser.eval(self.querystr)
        if self.aggregation and len(self.query) > 1 and isinstance(self.query[0], str) and self.query[0].startswith('from'):
            self.mongocollection = self.query[0][4:]
            self.query = self.query[1:]

        if self.aggregation and len(self.query) > 1 and '$raw' in self.query[-1]:
            self.raw = self.query[-1]['$raw']
            self.query = self.query[:-1]
        
        if req and not self.aggregation:
            self.query = {'$and': [self.query, req]}
        self.limit = limit
        self.skip = skip
        self.sort = sort.split(',') if sort else []

    def fetch(self):
        if self.mongocollection:
            class TempParagraph(Paragraph):
                _collection = self.mongocollection
            rs = TempParagraph
        else:
            rs = Paragraph
        
        if self.aggregation:
            rs = rs.aggregate(self.query if isinstance(self.query, list) else [self.query], raw=self.raw, allowDiskUse=True)
        else:
            rs = rs.query(self.query)
            if self.sort:
                rs = rs.sort(*self.sort)
            if self.skip:
                rs = rs.skip(self.skip)
            if self.limit:
                rs = rs.limit(self.limit)
        return rs

    def count(self):
        try:
            return self.fetch().count()
        except:
            return -1
