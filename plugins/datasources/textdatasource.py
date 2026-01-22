"""
Data Source from File Patterns
@zhs 文件名模式匹配数据源
"""

import codecs

from jindai.models import Paragraph
from jindai.storage import instance as storage
from jindai.pipeline import DataSourceStage, PipelineStage


class FilePatternDataSource(DataSourceStage):
    """
    Data Source from File Patterns
    @zhs 文件名模式匹配数据源
    """

    def apply_params(self, content=""):
        """File Pattern Data Source

        :param content: Patterns
            @zhs 文件名模式
        :type content: FILES
        """
        self.paths = PipelineStage.parse_paths(content)

    def fetch(self):
        for path in self.paths:
            yield Paragraph(content=path)


class TextDataSource(DataSourceStage):
    """
    Read Paragraphs from text files
    @zhs 从文本文件中读取语段
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
                @zhs HTML或包含HTML的ZIP压缩包文件列表
        """
        self.name = dataset_name
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    def fetch(self):
        for path in self.files:
            for i, line in enumerate(storage.open(path)):
                yield Paragraph(content=codecs.decode(line),
                                source_url=path if '://' in path else storage.relative_path(path),
                                dataset=self.name, lang=self.lang, outline=f'{i+1:06d}')


class LinesDataSource(DataSourceStage):
    """
    Import paragraphs from lines
    @zhs 从直接输入的文本中获得语段，每行一个语段
    """

    def apply_params(self, dataset_name='', lang="auto", content="", params=None, delimiter='\n'):
        """
        Args:
            dataset_name (DATASET):
                Data name
                @zhs 数据集名称
            lang (LANG):
                Language code
                @zhs 语言标识
            content (str):
                Text contents
                @zhs 文本内容
            params (object):
                Other customizable fields
                @zhs 其他自定义字段
            delimiter (str):
                Delimiter
                @zhs 分隔符，默认为换行符
        """
        self.name = dataset_name
        self.lang = lang
        self.lines = content.split(delimiter)
        self.params = params or {}

    def fetch(self):
        self.params.pop('contnet', '')
        for line in self.lines:
            yield Paragraph(lang=self.lang, dataset=self.name, content=line)


class BiblioDataSource(DataSourceStage):
    """
    Import paragraph from EndNote bibliography items
    @zhs 从 EndNote 文献条目产生语段
    """

    def apply_params(self, content='', dataset_name='', lang='zhs', input_format='endnote') -> None:
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
            format (endnote|other, unsupported):
                Format
                @zhs 文献条目信息格式
        """
        if not hasattr(self, input_format):
            raise NotImplementedError()

        self.method = getattr(self, input_format)
        self.dataset = dataset_name
        self.lang = lang
        self.files = PipelineStage.parse_paths(content)

    def endnote(self, lines):
        """"Parse EndNote format"""

        doc = {
            'content': '',
            'authors': []
        }
        field = ''
        for line in lines:
            if not line.strip():
                if doc:
                    yield Paragraph.from_dict(dict(dataset=self.dataset, lang=self.lang, **doc))
                doc = {
                    'content': '',
                    'authors': []
                }
            else:
                line = line.decode('utf-8').strip()
                if ' ' not in line:
                    value = line
                else:
                    field, value = line.split(' ', 1)
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
                    }.get(field.upper(), 'content')

                if ';' in value and field != 'content':
                    value = [_ for _ in value.split(';') if _]
                if field in doc:
                    if field == 'content':
                        doc[field] += value
                    else:
                        if not isinstance(doc[field], list):
                            doc[field] = [doc[field]]
                        if isinstance(value, list):
                            doc[field] += value
                        else:
                            doc[field].append(value)
                else:
                    doc[field] = value
        if doc:
            yield Paragraph.from_dict(dict(dataset=self.dataset, **doc))

    def fetch(self):
        for file in self.files:
            yield from self.method(file)
