"""Handling File Uploads
@chs 处理用户通过网页上传的文档
"""

import tempfile

from jindai.storage import instance as storage
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage


class FileUploadDataSource(DataSourceStage):
    """
    Use user-uploaded temporary file for input
    @chs 上传文件
    """

    class Implementation(DataSourceStage.Implementation):
        """Implementing datasource"""

        def __init__(self, file='file', field='content'):
            """
            Args:
                file (file:pdf,html,zip,txt):
                    File data-url string
                    @chs 上传的文件
                field (str):
                    Field to store temporary file name
                    @chs 文件名写入到的字段
            """
            super().__init__()
            self.files = self.parse_data_uri(file)
            self.field = field

        def parse_data_uri(self, data_uri: str) -> str:
            """
            Parse data uri and write into a temp file
            Returns:
                str: temp file path
            """
            tmp = tempfile.mktemp()
            with open(tmp, "wb") as fout:
                fout.write(storage.open(data_uri, 'rb').read())
            return tmp

        def fetch(self):
            yield Paragraph(**{self.field: self.files})
