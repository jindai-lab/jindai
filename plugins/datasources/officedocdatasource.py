"""
Import from Word/Excel Documents
@zhs Word/Excel 用作数据源"""
import os
import subprocess
import tempfile
from typing import Iterable

import pandas as pd
from jindai.storage import instance as storage
from jindai.models import Paragraph, Dataset
from jindai.pipeline import DataSourceStage, PipelineStage


class WordDataSource(DataSourceStage):
    """
    Import from Word documents
    @zhs 从Word文档中导入语段
    """

    def apply_params(self, dataset_name='', lang='auto', content=''):
        """
        Args:
            dataset_name (DATASET): 
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (FILES):
                Paths
                @zhs 文件列表
        """
        self.dataset = Dataset.get_by_name(dataset_name)
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

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
                    source_url=storage.relative_path(file),
                    pagenum=1,
                    dataset=self.dataset.id,
                    outline=''
                )
                yield para


class ExcelDataSource(DataSourceStage):
    """
    Import from Excel documents
    @zhs 从Word文档中导入语段
    """

    def apply_params(self, content='', dataset_name='', lang='auto') -> None:
        """
        Args:
            dataset_name (DATASET): 
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (FILES):
                Paths
                @zhs 文件列表
        """
        
        self.dataset = Dataset.get_by_name(dataset_name)
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    def fetch(self) -> Iterable[Paragraph]:
        for file in self.files:
            dataframe = pd.read_excel(file)
            for _, row in dataframe.iterrows():
                data = row.to_dict()
                if 'dataset' not in data:
                    data['dataset'] = self.dataset.id
                if 'lang' not in data:
                    data['lang'] = self.lang
                yield Paragraph(**data)
