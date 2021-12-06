"""来自Word文档
"""

import subprocess
import os
import tempfile

from models import Paragraph
from datasource import DataSource


class WordDataSource(DataSource):
    """从Word文档中导入语段
    """

    def __init__(self, collection_name, lang, *files):
        """
        Args:
            collection_name (str): 集合名称
            lang (简体中文:chs|繁体中文:cht|英文:en|德文:de|法文:fr|俄文:ru|西班牙文:es|日文:ja|韩文/朝鲜文:kr|越南文:vn): 语言标识
            files (files): Word文档列表
        """
        self.name = collection_name
        self.lang = lang
        self.files = files

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
                    lang=self.lang, content=doc, source={'file': f}, pagenum=1,
                    collection=self.name, outline=''
                )
                yield p
