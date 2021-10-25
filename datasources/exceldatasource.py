"""Excel 用作数据源"""
import pandas as pd
from typing import Iterable
from models import Paragraph
from datasource import DataSource
from .utils import expand_file_patterns


class ExcelDataSource(DataSource):
    """从 Excel 导入数据"""
    
    def __init__(self, files_or_patterns) -> None:
        '''
        Args:
            files_or_patterns (str): 文件名或通配符，每行一个
        '''
        self.files = expand_file_patterns(files_or_patterns.split('\n'))
        
    def fetch(self) -> Iterable[Paragraph]:
        for _, f in self.files:
            df = pd.read_excel(f)
            for _, r in df.iterrows():
                d = r.to_dict()
                for k in d:
                    v = d[k]
                    if (v.startswith('[') and v.endswith(']')) or \
                       (v.startswith('{') and v.endswith('}')):
                           d[k] = eval(v)
                yield Paragraph(**d)
        