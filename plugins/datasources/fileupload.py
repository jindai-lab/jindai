"""处理用户通过网页上传的文档"""

from urllib import request
import tempfile

from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage


class FileUploadDataSource(DataSourceStage):
    """上传文件
    """

    class Implementation(DataSourceStage.Implementation):
        """Implementing datasource"""

        def __init__(self, file='file', field='content'):
            """
            Args:
                file (file:pdf,html,zip,txt): 上传的文件
                field (str): 文件名写入到字段中
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
            with request.urlopen(data_uri) as response:
                data = response.read()
            tmp = tempfile.mktemp()
            with open(tmp, "wb") as fout:
                fout.write(data)
            return tmp

        def fetch(self):
            yield Paragraph(**{self.field: self.files})
