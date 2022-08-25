"""
Data Source from File Patterns
@chs 文件名模式匹配数据源
"""

from jindai.models import Paragraph
from jindai.storage import instance as storage
from jindai.pipeline import DataSourceStage


class FilePatternDataSource(DataSourceStage):
    """
    Data Source from File Patterns
    @chs 文件名模式匹配数据源
    """

    class Implementation(DataSourceStage.Implementation):
        """Implementing File Pattern Data Source"""

        def __init__(self, content=""):
            """File Pattern Data Source

            :param content: Patterns
                @chs 文件名模式
            :type content: str, optional
            """
            self.patterns = content
            super().__init__()

        def fetch(self):
            for path in storage.expand_patterns(self.patterns):
                yield Paragraph(content=path)
