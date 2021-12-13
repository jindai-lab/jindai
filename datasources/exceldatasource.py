"""Excel 用作数据源"""
import pandas as pd
from typing import Iterable
from models import Paragraph
from datasource import DataSource
from .utils import expand_file_patterns


class ExcelDataSource(DataSource):
    """从 Excel 导入数据"""
    
    def __init__(self, files_or_patterns, collection, lang) -> None:
        '''
        Args:
            files_or_patterns (str): 文件名或通配符，每行一个
            collection (str): 集合名称
            lang (简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
        '''
        self.files = expand_file_patterns(files_or_patterns.split('\n'))
        self.collection = collection
        self.lang = lang
        
    def fetch(self) -> Iterable[Paragraph]:
        for _, f in self.files:
            df = pd.read_excel(f)
            for _, r in df.iterrows():
                d = r.to_dict()
                if 'collection' not in d:
                    d['collection'] = self.collection
                if 'lang' not in d:
                    d['lang'] = self.lang
                yield Paragraph(**d)
        