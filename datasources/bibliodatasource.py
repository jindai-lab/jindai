"""导入常用的文献参考目录
"""

from models import Paragraph
from datasource import DataSource
from .utils import expand_file_patterns


class BiblioDataSource(DataSource):
    """从 endnote 文献条目产生语段
    """
    
    def __init__(self, bibfiles, collection, lang='chs', format='endnote') -> None:
        """
        Args:
            bibfiles (str): 文件名或通配符，一行一个
            lang (简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|葡萄牙文:pt|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
            collection (str): 集合名称
            format (endnote|bibtex): 文献条目信息格式
        """
        super().__init__()
        if not hasattr(self, format):
            raise NotImplemented()
        
        self.method = getattr(self, format)
        self.files = expand_file_patterns(bibfiles.split('\n'))
        self.collection = collection
        self.lang=lang
        
    def endnote(self, lines):
        d = {
            'content': '',
            'authors': []
        }
        field = ''
        for l in lines:
            if not l.strip():
                if d:
                    yield Paragraph(collection=self.collection, lang=self.lang, **d)
                d = {
                    'content': '',
                    'authors': []
                }
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
                        '%X': 'content',
                        '%P': 'pages',
                        '%@': 'issn',
                        '%L': 'cn_publishing_number',
                        '%W': 'catalog'
                    }.get(field.upper(), f'content')
                
                if ';' in value and field != 'content':
                    value = [_ for _ in value.split(';') if _]
                if field in d:
                    if field == 'content':
                        d[field] += value
                    else:
                        if not isinstance(d[field], list):
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
            