"""
Query Database
@zhs 来自数据库
"""
from jindai.app import config
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage


class DBQueryDataSource(DataSourceStage):
    """Query from Database
    @zhs 从数据库查询
    """

    def apply_params(self, query, req='', mongocollections='', limit=0, skip=0, sort='', raw=False, groups='none'):
        """
        Args:
            query (QUERY):
                Query expression, or keywords
                @zhs 查询字符串，或以 ? 开头的查询表达式，或以 ?? 开头的聚合查询表达式
            req (QUERY):
                Additional query expression
                @zhs 附加的条件
            sort (str):
                Sorting expression
                @zhs 排序表达式
            limit (int):
                Limitation for maximal results, 0 for none
                @zhs 查询最多返回的结果数量，默认为0即无限制
            skip (int):
                Skip %1 results
                @zhs 返回从第%1个开始的结果
            mongocollections (LINES):
                Name for colletion name in MongoDB, one item per line
                @zhs 数据库中其他数据集的名称
            raw (bool):
                Return dicts instead of Paragraph objects
                @zhs 若为 False（默认值）则返回 Paragraph 对象，否则返回原始数据，仅对于聚合查询有效
            groups (str):
                @choose(none|group|source|both) Groups
                @zhs @choose(无:none|按组:group|按来源:source|分组和来源:both) 分组
        """
        self.dbquery = DBQuery(
            query if not req else (query, req), mongocollections, limit, skip, sort, raw, groups)

    def fetch(self):
        return self.dbquery.fetch()
