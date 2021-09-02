"""从文件获取图像
"""

from PIL import Image
from datasource import DataSource
from models import Paragraph
import glob

    
def expand_file_patterns(patterns):
    for pattern in patterns:
        for f in glob.glob(pattern):
            yield f


class ImageDataSource(DataSource):
    """从文件列表中获得图像
    """

    def __init__(self, files_or_patterns):
        """
        Args:
            files_or_patterns (str): 文件列表
        """
        self.files = expand_file_patterns(files_or_patterns.split('\n'))

    def fetch(self):
        for fp, imfile in self.files:
            i = Image.open(fp)
            yield Paragraph(source={'file': imfile}, _image=i)
