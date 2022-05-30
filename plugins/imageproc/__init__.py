"""图像相关插件"""

from typing import List
from PyMongoWrapper import F, ObjectId
from jindai.helpers import *
from jindai.models import Paragraph
from jindai import Plugin

from .imageproc import *
from .ocr import *


class ImageProcPlugin(Plugin):
    """图像处理插件"""

    def __init__(self, pmanager):
        super().__init__(pmanager)
        self.register_pipelines(globals())
