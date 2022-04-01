"""Word/Excel 用作数据源"""
import os
import subprocess
import tempfile
from typing import Iterable

import pandas as pd
from jindai import expand_patterns, truncate_path
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage


class WordDataSource(DataSourceStage):
    """从Word文档中导入语段
    """
    class Implementation(DataSourceStage.Implementation):

        def __init__(self, dataset, lang, content):
            """
            Args:
                dataset_name (DATASET): 数据集名称
                lang (LANG): 语言标识
                content (files): Word文档列表
            """
            super().__init__()
            self.name = dataset
            self.lang = lang
            self.files = expand_patterns(content)
            
        def call_abiword(self, file):
            fn = tempfile.mktemp()
            subprocess.call(['abiword', '--to', 'txt', '-o', fn, file])
            if os.path.exists(fn):
                with open(fn, encoding='utf-8') as fi:
                    res = fi.read()
                os.unlink(fn)
                return res

        def fetch(self):
            for f in self.files:
                doc = self.call_abiword(f)
                if doc:
                    p = Paragraph(
                        lang=self.lang, content=doc, source={'file': truncate_path(f)}, pagenum=1,
                        dataset=self.name, outline=''
                    )
                    yield p
                    


class ExcelDataSource(DataSourceStage):
    """从Excel文档中导入语段数据
    """
    
    class Implementation(DataSourceStage.Implementation):
        """从 Excel 导入数据"""

        def __init__(self, content, dataset, lang) -> None:
            '''
            Args:
                content (str): 文件名或通配符，每行一个
                dataset (DATASET): 集合名称
                lang (LANG): 语言标识
            '''
            super().__init__()
            self.files = expand_patterns(content)
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
