"""Excel 用作数据源"""
import pandas as pd
from typing import Iterable
from models import Paragraph
from datasource import DataSource
from .utils import expand_file_patterns


class ExcelDataSource(DataSource):
    """从 Excel 导入数据"""
    
    def __init__(self, files_or_patterns, dataset, lang) -> None:
        '''
        Args:
            files_or_patterns (str): 文件名或通配符，每行一个
            dataset (DATASET): 集合名称
            lang (LANG): 语言标识
        '''
        super().__init__()
        self.files = expand_file_patterns(files_or_patterns.split('\n'), names_only=True)
        self.dataset = dataset
        self.lang = lang
        
    def fetch(self) -> Iterable[Paragraph]:
        for f in self.files:
            df = pd.read_excel(f)
            for _, r in df.iterrows():
                d = r.to_dict()
                if 'dataset' not in d:
                    d['dataset'] = self.dataset
                if 'lang' not in d:
                    d['lang'] = self.lang
                yield Paragraph(**d)
        