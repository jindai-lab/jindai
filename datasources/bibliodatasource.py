"""导入常用的文献参考目录
"""

from models import Paragraph
from datasource import DataSource
from .utils import expand_file_patterns


class BiblioDataSource(DataSource):
    
    def __init__(self, bibfiles, collection, format='endnote') -> None:
        """
        Args:
            bibfiles (str): 文件名或通配符，一行一个
            collection (str): 数据集名称
            format (endnote|bibtex): 文献条目信息格式
        """
        if not hasattr(self, format):
            raise NotImplemented()
        
        self.method = getattr(self, format)
        self.files = expand_file_patterns(bibfiles.split('\n'))
        self.collection = collection
        
    def endnote(self, lines):
        d = {}
        field = ''
        for l in lines:
            if not l.strip():
                if d: yield Paragraph(collection=self.collection, **d)
                d = {}
            else:
                l = l.decode('utf-8').strip()
                if ' ' not in l:
                    value = l
                else:
                    field, value = l.split(' ', 1)
                    field = {
                        '%0': 'item_type',
                        '%A': 'authors',
                        '%+': 'institutions',
                        '%J': 'journal',
                        '%D': 'pdate',
                        '%T': 'title',
                        '%N': 'issue',
                        '%K': 'tags',
                        '%X': 'abstract',
                        '%P': 'pages',
                        '%@': 'issn',
                        '%L': 'cn_publishing_number',
                        '%W': 'catalog'
                    }.get(field.upper(), f'field_{field[1:]}')
                
                if ';' in value and field != 'abstract':
                    value = [_ for _ in value.split(';') if _]
                if field == 'authors' and not isinstance(value, list):
                    value = [value]
                if field in d:
                    if field == 'abstract':
                        d[field] += value
                    else:
                        d[field] = [d[field]]
                        if isinstance(value, list):
                            d[field] += value
                        else:
                            d[field].append(value)                        
                else:
                    d[field] = value
        if d:
            yield Paragraph(collection=self.collection, **d)
            
    def fetch(self):
        for f, _ in self.files:
            yield from self.method(f)
            