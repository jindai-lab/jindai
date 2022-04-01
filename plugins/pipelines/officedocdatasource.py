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
        """Impl"""

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
            """Call abiword to extract text from a word document"""
            filename = tempfile.mktemp()
            subprocess.call(['abiword', '--to', 'txt', '-o', filename, file])
            if os.path.exists(filename):
                with open(filename, encoding='utf-8') as input_file:
                    res = input_file.read()
                os.unlink(filename)
                return res

        def fetch(self):
            for file in self.files:
                doc = self.call_abiword(file)
                if doc:
                    para = Paragraph(
                        lang=self.lang, content=doc,
                        source={'file': truncate_path(file)}, pagenum=1,
                        dataset=self.name, outline=''
                    )
                    yield para


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
            for file in self.files:
                dataframe = pd.read_excel(file)
                for _, row in dataframe.iterrows():
                    data = row.to_dict()
                    if 'dataset' not in data:
                        data['dataset'] = self.dataset
                    if 'lang' not in data:
                        data['lang'] = self.lang
                    yield Paragraph(**data)
